from __future__ import annotations

import logging
import time
from typing import Any

from agent.shared import QAState, return_with_profile
from api.answer import build_answer, no_evidence_answer
from api.config import get_settings
from api.db import SessionLocal
from api.dogv_resolver import (
    Reference,
    named_target_topic_overlap,
    parse_named_norm_target,
    parse_reference,
    reference_matches_title,
    title_num_year,
    title_primary_tipo,
)
from api.dogv_urls import build_html_url, build_pdf_url
from api.models import DogvDocument, DogvIssue
from api.query_classifiers import is_reference_query

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _map_language(intent_lang: str | None) -> str:
    if intent_lang == "ca":
        return "va_va"
    return "es_es"


def _parse_citation_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for key in ("doc_id", "document_id", "id"):
            if key in value:
                return _parse_citation_id(value.get(key))
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("doc_"):
            text = text[4:]
        if text.isdigit():
            return int(text)
    return None


def _collapse_to_principal(
    citations: list[dict[str, Any]],
    question: str,
    ondemand_doc_id: int | None,
) -> list[dict[str, Any]]:
    """For an explicit single-disposition question ("Decreto 185/2018"), surface
    only the cited disposition itself instead of a long tail of tangential docs —
    but only when that principal is actually present among the citations. If it is
    not (we couldn't ground it), leave the citations untouched."""
    if len(citations) <= 1 or not is_reference_query(question):
        return citations
    ref = parse_reference(question)
    if ref is None:
        return citations
    matches = [c for c in citations if reference_matches_title(ref, c.get("title") or "")]
    if not matches:
        return citations
    if len(matches) > 1 and ondemand_doc_id is not None:
        pinned = [c for c in matches if c.get("document_id") == ondemand_doc_id]
        if pinned:
            return pinned[:1]
    return matches[:1]


# A named target needs at least this many topic terms in the title (and this share
# of them) before we'll force it as the sole citation — guards against coincidental
# single-word overlaps citing the wrong primary norm.
_NAMED_MIN_MATCHED = 2
_NAMED_MIN_RATIO = 0.5


def _grounded_meta(state: QAState) -> dict[int, dict[str, Any]]:
    """doc_id -> {title, rank} for docs actually READ (evidence ∪ full_docs),
    titles + retrieval rank sourced from the fused candidate pool they came from.
    Restricting to the read set keeps a forced citation grounded in used evidence."""
    grounded: set[int] = set()
    for item in state.get("evidence") or []:
        did = item.get("doc_id") or item.get("document_id")
        if did is not None:
            grounded.add(int(did))
    for item in state.get("full_docs") or []:
        did = item.get("document_id") or item.get("doc_id")
        if did is not None:
            grounded.add(int(did))
    meta: dict[int, dict[str, Any]] = {}
    for rank, cand in enumerate(state.get("candidate_docs") or []):
        did = cand.get("document_id")
        if did is None:
            continue
        did = int(did)
        if did in grounded and did not in meta:
            meta[did] = {"title": cand.get("title") or "", "rank": rank}
    return meta


def _evidence_doc_meta(state: QAState) -> dict[int, dict[str, Any]]:
    """doc_id -> {title, ref, issue_date} for every retrieved candidate, so each
    evidence quote can self-identify its disposition in the synthesis prompt.
    Sourced from candidate_docs (DB-backed titles), keeping it grounded."""
    meta: dict[int, dict[str, Any]] = {}
    for cand in state.get("candidate_docs") or []:
        did = cand.get("document_id")
        if did is None:
            continue
        did = int(did)
        if did in meta:
            continue
        meta[did] = {
            "title": cand.get("title") or "",
            "ref": cand.get("ref") or "",
            "issue_date": cand.get("issue_date") or "",
        }
    return meta


def _ref_from_num_year(tipo: str | None, num_year: str | None) -> Reference | None:
    """Rebuild a Reference from a backfill-resolved {tipo, 'N/YYYY'} pair."""
    if not num_year or "/" not in num_year:
        return None
    num, _, year = num_year.partition("/")
    if not (num.isdigit() and year.isdigit()):
        return None
    return Reference(tipo=tipo, numero=int(num), anyo=int(year), raw=num_year)


def _norm_target_doc_id(state: QAState) -> int | None:
    """If the question targets one specific primary disposition and that norm is in
    the read set, return its doc id — even when synthesis cited something else.

    Numbered targets ("Decreto 185/2018") match by ref; named targets ("la Ley de
    la Función Pública Valenciana") match by leading tipo + topic-term overlap.
    Returns None when there's no norm target or the match is ambiguous/absent."""
    question = state.get("question") or ""
    meta = _grounded_meta(state)
    if not meta:
        return None

    # An inferred no-number target (e.g. "la Ley de Transparencia" -> Ley 1/2022,
    # resolved by backfill) carries no number in the question for parse_reference to
    # find; match the read set against the ref backfill handed us.
    target_ref = state.get("norm_target_ref")
    if target_ref:
        ref = _ref_from_num_year(target_ref.get("tipo"), target_ref.get("num_year"))
        if ref is not None:
            matches = sorted(
                (m["rank"], did)
                for did, m in meta.items()
                if reference_matches_title(ref, m["title"])
            )
            if matches:
                ondemand = state.get("ondemand_doc_id")
                for _, did in matches:
                    if did == ondemand:
                        return did
                return matches[0][1]

    ref = parse_reference(question)
    if ref is not None:
        matches = sorted(
            (m["rank"], did)
            for did, m in meta.items()
            if reference_matches_title(ref, m["title"])
        )
        if not matches:
            return None
        ondemand = state.get("ondemand_doc_id")
        for _, did in matches:
            if did == ondemand:
                return did
        return matches[0][1]

    named = parse_named_norm_target(question)
    if named is None:
        return None
    scored: list[tuple[int, int, int]] = []  # (matched, -rank, doc_id)
    for did, m in meta.items():
        if title_primary_tipo(m["title"]) != named.tipo:
            continue
        matched, ratio = named_target_topic_overlap(named, m["title"])
        if matched >= _NAMED_MIN_MATCHED and ratio >= _NAMED_MIN_RATIO:
            scored.append((matched, -m["rank"], did))
    if not scored:
        return None
    scored.sort(reverse=True)
    top_matched = scored[0][0]
    # Distinct norms (different N/YYYY) tied at the top -> ambiguous, don't force.
    top_norms = {
        title_num_year(meta[did]["title"])
        for matched, _, did in scored
        if matched == top_matched
    }
    top_norms.discard(None)
    if len(top_norms) > 1:
        return None
    return scored[0][2]


def _citation_for_doc(doc_id: int, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Single-citation list for `doc_id`, reusing an existing citation dict if the
    synthesis already cited it, else loading it from the DB."""
    for c in citations:
        if c.get("document_id") == doc_id:
            return [c]
    with SessionLocal() as db:
        row = (
            db.query(DogvDocument, DogvIssue)
            .join(DogvIssue)
            .filter(DogvDocument.id == doc_id)
            .first()
        )
    if row is None:
        return citations
    doc, issue = row
    return [
        {
            "document_id": doc.id,
            "title": doc.title,
            "ref": doc.ref,
            "issue_date": issue.date.isoformat() if issue.date else None,
            "pdf_url": build_pdf_url(doc.pdf_url),
            "html_url": build_html_url(doc.html_url),
        }
    ]


def answer_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    intent = state.get("intent") or {}
    language = state.get("language") or _map_language(intent.get("language"))
    evidence = state.get("evidence") or []
    full_docs = state.get("full_docs") or []
    try:
        if not evidence and not full_docs:
            elapsed = time.monotonic() - start
            logger.info("answer.skip req=%s reason=no_evidence elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "answer",
                no_evidence_answer(language),
                elapsed_seconds=round(elapsed, 3),
                skipped=True,
                reason="no_evidence",
                citations=0,
                evidence=0,
                full_docs=0,
                repair_attempts=0,
                repair_success=False,
                fallback_reason="no_evidence",
            )
        result = build_answer(
            state["question"],
            language,
            evidence,
            full_docs=full_docs,
            history=state.get("history"),
            doc_meta=_evidence_doc_meta(state),
        )
        answer = result.get("answer") or ""
        diagnostics = result.get("diagnostics") or {}
        cited_ids: set[int] = set()
        for raw in result.get("citations") or []:
            parsed = _parse_citation_id(raw)
            if parsed is not None:
                cited_ids.add(parsed)

        citations = []
        if not cited_ids and evidence:
            # The synthesis cited nothing (e.g. an abstention like "no hay evidencia"):
            # fall back to the evidence docs, but cap to the same few the abstention
            # summary shows so the citation floor's topical padding doesn't surface a
            # long citation tail under an out-of-scope answer. Grounded answers cite
            # explicitly and never reach this branch.
            fallback_max = max(1, int(getattr(settings, "answer_fallback_max_items", 3) or 3))
            for item in evidence:
                if len(cited_ids) >= fallback_max:
                    break
                parsed = _parse_citation_id(item)
                if parsed is not None:
                    cited_ids.add(parsed)
        if cited_ids:
            with SessionLocal() as db:
                docs = (
                    db.query(DogvDocument, DogvIssue)
                    .join(DogvIssue)
                    .filter(DogvDocument.id.in_(cited_ids))
                    .all()
                )
            for doc, issue in docs:
                citations.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "pdf_url": build_pdf_url(doc.pdf_url),
                        "html_url": build_html_url(doc.html_url),
                    }
                )

        if settings.answer_norm_target_citation_enabled:
            principal_id = _norm_target_doc_id(state)
            if principal_id is not None:
                citations = _citation_for_doc(principal_id, citations)
            else:
                citations = _collapse_to_principal(
                    citations, state.get("question") or "", state.get("ondemand_doc_id")
                )
        else:
            citations = _collapse_to_principal(
                citations, state.get("question") or "", state.get("ondemand_doc_id")
            )

        elapsed = time.monotonic() - start
        logger.info(
            "answer.done req=%s citations=%s evidence=%s full_docs=%s elapsed=%.2fs",
            request_id,
            len(citations),
            len(evidence),
            len(full_docs),
            elapsed,
        )
        return return_with_profile(
            state,
            "answer",
            {"answer": answer, "citations": citations},
            elapsed_seconds=round(elapsed, 3),
            skipped=False,
            citations=len(citations),
            evidence=len(evidence),
            full_docs=len(full_docs),
            validator_triggered=bool(diagnostics.get("validation_errors_initial")),
            repair_attempts=int(diagnostics.get("repair_attempts") or 0),
            repair_success=bool(diagnostics.get("repair_success")),
            fallback_reason=diagnostics.get("fallback_reason"),
            validation_errors_initial=diagnostics.get("validation_errors_initial") or [],
            validation_errors_final=diagnostics.get("validation_errors_final") or [],
            deterministic_fix_applied=bool(diagnostics.get("deterministic_fix_applied")),
            deterministic_fix_types=diagnostics.get("deterministic_fix_types") or [],
            repair_skipped_reason=diagnostics.get("repair_skipped_reason"),
            rejected_answer=diagnostics.get("rejected_answer"),
        )
    except Exception:
        logger.exception("answer.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
