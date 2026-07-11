from __future__ import annotations

import logging
import re
import time
from typing import Any

from sqlalchemy import text as sa_text

from agent.nodes.read_chunks import (
    _lexical_chunks_for_docs,
    _salient_keywords,
    _select_chunks,
    _window_chunk_text,
)
from agent.shared import (
    QAState,
    best_snippet,
    coverage_score,
    estimate_tokens,
    extract_keywords_simple,
    return_with_profile,
    rrf_margin_ratio,
)
from api.config import get_settings
from api.db import SessionLocal
from api.enumeration import is_enumeration_query
from api.intent import needs_amounts, needs_eligibility
from api.models import DogvDocument, DogvIssue
from api.reader import extract_evidence
from api.retrieval import top_chunks_for_docs
from api.taxonomy import is_base_like

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _merge_unique_ids(*id_lists: list[int]) -> list[int]:
    """Concatenate id lists keeping first occurrence order, deduped."""
    seen: set[int] = set()
    merged: list[int] = []
    for ids in id_lists:
        for doc_id in ids:
            doc_id = int(doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(doc_id)
    return merged


def _read_budget(
    question: str, doc_ids: list[int], candidate_docs: list[dict[str, Any]]
) -> tuple[list[int], int, bool]:
    """Cap the rerank-selected read set to the (possibly expanded) read budget.

    Enumeration queries ("list all dispositions of mayo 2026") must read the whole
    series, not the usual handful — widen the read budget to the enumeration cap so
    the augmented candidates actually reach the answer node. A flat RRF margin also
    widens the budget: no candidate clearly dominates, so read a few more.
    Returns (capped_doc_ids, read_max_docs, enumeration_query).
    """
    read_max_docs = getattr(settings, "ask_read_max_docs", 3)
    rerank_cap = getattr(settings, "ask_rerank_max_candidates", read_max_docs)
    enumeration_query = is_enumeration_query(question)
    if getattr(settings, "enumeration_augment_enabled", False) and enumeration_query:
        rerank_cap = getattr(settings, "ask_enumeration_max_candidates", rerank_cap)
    read_max_docs = max(read_max_docs, min(len(doc_ids), rerank_cap))
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_docs = getattr(settings, "ask_read_expand_docs", 2)
    if candidate_docs and rrf_margin_ratio(candidate_docs, probe=expand_probe) < expand_ratio:
        read_max_docs += expand_docs
    if read_max_docs > len(doc_ids):
        read_max_docs = len(doc_ids)
    return doc_ids[:read_max_docs], read_max_docs, enumeration_query


def _coverage_extras(
    question: str,
    keywords: list[str],
    doc_ids: list[int],
    candidate_docs: list[dict[str, Any]],
    top_chunks: dict[int, list[dict[str, Any]]],
    read_max_docs: int,
) -> list[int]:
    """Add unread candidates whose title/snippet covers the question keywords."""
    coverage_extra = getattr(settings, "ask_read_coverage_docs", 2)
    if coverage_extra <= 0 or not candidate_docs or not keywords:
        return doc_ids
    scored: list[tuple[int, int]] = []
    for item in candidate_docs:
        doc_id = int(item["document_id"])
        if doc_id in doc_ids:
            continue
        text = item.get("title") or ""
        chunk_list = top_chunks.get(doc_id) or []
        if chunk_list:
            text = f"{text} {best_snippet(question, chunk_list)}"
        score = coverage_score(text, keywords)
        if score > 0:
            scored.append((score, doc_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    extras = [doc_id for _, doc_id in scored[:coverage_extra]]
    if not extras:
        return doc_ids
    merged = _merge_unique_ids(doc_ids, extras)
    return merged[: min(len(merged), read_max_docs + coverage_extra)]


def _eligibility_extras(
    question: str,
    keywords: list[str],
    doc_ids: list[int],
    candidate_docs: list[dict[str, Any]],
    top_chunks: dict[int, list[dict[str, Any]]],
) -> list[int]:
    """For eligibility questions, add base/convocatoria-like docs (they carry the
    requisitos), preferring the ones whose text covers the question keywords."""
    if not needs_eligibility(question):
        return doc_ids
    eligibility_extra = getattr(settings, "ask_read_eligibility_docs", 1)
    if needs_amounts(question):
        eligibility_extra = max(eligibility_extra, 3)
    if eligibility_extra <= 0 or not candidate_docs:
        return doc_ids
    base_scored: list[tuple[int, int]] = []
    for item in candidate_docs:
        doc_id = int(item["document_id"])
        if doc_id in doc_ids:
            continue
        if is_base_like(item):
            text = item.get("title") or ""
            chunk_list = top_chunks.get(doc_id) or []
            if chunk_list:
                text = f"{text} {best_snippet(question, chunk_list)}"
            score = coverage_score(text, keywords) if keywords else 0
            base_scored.append((score, doc_id))
    base_scored.sort(key=lambda item: item[0], reverse=True)
    base_candidates = [doc_id for _, doc_id in base_scored[:eligibility_extra]]
    if not base_candidates:
        return doc_ids
    return _merge_unique_ids(doc_ids, base_candidates)


def _amount_extras(
    question: str,
    doc_ids: list[int],
    candidate_docs: list[dict[str, Any]],
    top_chunks: dict[int, list[dict[str, Any]]],
) -> list[int]:
    """For amount questions, add the unread candidates densest in numeric tokens."""
    amount_extra = getattr(settings, "ask_read_amount_docs", 1)
    if amount_extra <= 0 or not candidate_docs or not needs_amounts(question):
        return doc_ids

    def _amount_score(text: str) -> int:
        return len(re.findall(r"\b\d[\d.,]*\b", text))

    scored: list[tuple[int, int]] = []
    for item in candidate_docs:
        doc_id = int(item["document_id"])
        if doc_id in doc_ids:
            continue
        chunk_list = top_chunks.get(doc_id) or []
        text = " ".join((c.get("text") or "") for c in chunk_list)
        if not text:
            text = item.get("summary") or item.get("text") or item.get("title") or ""
        score = _amount_score(text)
        if score > 0:
            scored.append((score, doc_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    extras = [doc_id for _, doc_id in scored[:amount_extra]]
    if not extras:
        return doc_ids
    return _merge_unique_ids(doc_ids, extras)


_COMPANION_TOP_CANDIDATES = 3
# Companion kinds worth pulling in even without topic overlap: the source doc
# explicitly resolves/amends/corrects/repeals/convokes the target, so it is
# almost always the second half of a multi-doc answer.
_COMPANION_PRIORITY_KINDS = ("resuelve", "modifica", "corrige", "deroga", "convoca")


def _companion_extras(question: str, doc_ids: list[int], keywords: list[str]) -> list[int]:
    """Pull in doc_reference companions of the top-ranked candidates (e.g. the
    resolución that resolves a convocatoria the rerank alone never surfaces).

    Looks both directions (source->target and target->source) for the top
    `_COMPANION_TOP_CANDIDATES` doc_ids, since a companion may be the document
    that references the candidate or the one the candidate itself references.
    """
    if not getattr(settings, "doc_reference_expansion_enabled", True):
        return doc_ids
    max_companions = getattr(settings, "doc_reference_max_companions", 2)
    if max_companions <= 0 or not doc_ids:
        return doc_ids

    anchors = doc_ids[:_COMPANION_TOP_CANDIDATES]
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                """
                SELECT source_document_id, target_document_id, ref_kind, d.title
                FROM doc_reference r
                JOIN dogv_documents d
                  ON d.id = CASE WHEN r.source_document_id = ANY(:anchors)
                                  THEN r.target_document_id ELSE r.source_document_id END
                WHERE (source_document_id = ANY(:anchors) OR target_document_id = ANY(:anchors))
                  AND target_document_id IS NOT NULL
                """
            ),
            {"anchors": anchors},
        ).all()

    candidates: dict[int, tuple[int, int]] = {}  # companion_id -> (priority, topic_overlap)
    for source_id, target_id, ref_kind, companion_title in rows:
        companion_id = target_id if source_id in anchors else source_id
        if companion_id in doc_ids or companion_id in anchors:
            continue
        priority = 1 if ref_kind in _COMPANION_PRIORITY_KINDS else 0
        overlap = coverage_score(companion_title or "", keywords) if keywords else 0
        best = candidates.get(companion_id)
        if best is None or (priority, overlap) > best:
            candidates[companion_id] = (priority, overlap)

    if not candidates:
        return doc_ids
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    extras = [companion_id for companion_id, _ in ranked[:max_companions]]
    logger.info(
        "read.companion_expansion anchors=%s added=%s",
        anchors,
        extras,
    )
    return _merge_unique_ids(doc_ids, extras)


def _fetch_reader_inputs(
    doc_ids: list[int], question: str, query_embedding: list[float] | None
) -> tuple[
    dict[int, tuple[Any, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    list[str],
]:
    """One DB pass for everything the payload build needs.

    Returns (docs_by_id, vector-fallback chunks, document-order chunks,
    literal-keyword chunks, salient keywords).
    """
    with SessionLocal() as db:
        docs = (
            db.query(DogvDocument, DogvIssue)
            .join(DogvIssue)
            .filter(DogvDocument.id.in_(doc_ids))
            .all()
        )
        fallback_chunks: dict[int, list[dict[str, Any]]] = {}
        extra_chunks: dict[int, list[dict[str, Any]]] = {}
        if query_embedding:
            fallback_chunks = top_chunks_for_docs(
                db,
                query_embedding,
                doc_ids,
                per_doc=getattr(settings, "ask_chunks_per_doc", 4),
            )
        extra_limit = getattr(settings, "ask_chunks_per_doc", 4)
        if extra_limit > 0:
            rows = (
                db.execute(
                    sa_text(
                        """
                    SELECT document_id, chunk_index, text
                    FROM (
                        SELECT document_id, chunk_index, text,
                               ROW_NUMBER() OVER (
                                   PARTITION BY document_id
                                   ORDER BY chunk_index
                               ) AS rn
                        FROM rag_chunk
                        WHERE document_id = ANY(:doc_ids)
                    ) ranked
                    WHERE rn <= :extra_limit
                    ORDER BY document_id, chunk_index
                    """
                    ),
                    {"doc_ids": doc_ids, "extra_limit": extra_limit},
                )
                .mappings()
                .all()
            )
            for row in rows:
                extra_chunks.setdefault(int(row["document_id"]), []).append(
                    {
                        "chunk_index": int(row["chunk_index"]),
                        "text": row["text"],
                    }
                )

        salient = _salient_keywords(question)
        keyword_chunks = _lexical_chunks_for_docs(
            db,
            doc_ids,
            salient,
            per_doc=getattr(settings, "ask_chunks_per_doc", 6),
        )

    docs_by_id = {doc.id: (doc, issue) for doc, issue in docs}
    return docs_by_id, fallback_chunks, extra_chunks, keyword_chunks, salient


def _build_docs_payload(
    target_ids: list[int],
    docs_by_id: dict[int, tuple[Any, Any]],
    retrieval_map: dict[int, list[dict[str, Any]]],
    vector_map: dict[int, list[dict[str, Any]]],
    keyword_map: dict[int, list[dict[str, Any]]],
    ordered_map: dict[int, list[dict[str, Any]]],
    salient: list[str],
    enumeration_query: bool,
) -> list[dict[str, Any]]:
    """Assemble the per-doc reader payload (merged, windowed chunks + metadata)."""
    payload: list[dict[str, Any]] = []
    target_chunks = getattr(settings, "ask_chunks_per_doc", 6)
    chunk_max_chars = getattr(settings, "ask_chunk_max_chars", 1200)
    fallback_chars = getattr(settings, "ask_doc_fallback_chars", 12000)
    for doc_id in target_ids:
        row = docs_by_id.get(doc_id)
        if not row:
            continue
        doc, issue = row
        # Additive selection: keep the retrieval/BM25 chunks that already
        # carried the answer (preserve prior behaviour), THEN layer the best
        # semantic (vector) and literal-entity (keyword) chunks on top. Pure
        # superset of the old 4-retrieval-chunk payload, so it can only add
        # the missing answer chunk, never evict a working one.
        retrieval_keep = getattr(settings, "ask_read_retrieval_chunks", 4)
        merged = _select_chunks(
            [
                (retrieval_map.get(doc.id, []), retrieval_keep),
                (vector_map.get(doc.id, []), 4),
                (keyword_map.get(doc.id, []), 2),
                (ordered_map.get(doc.id, []), None),
            ],
            target_chunks,
        )
        # Enumeration answers list a series and are hypersensitive to payload
        # composition — re-windowing chunks swaps borderline series members
        # (tester #30 deterministically traded one May-2026 A1 ref for another).
        # Same guard RC4 needed for semantic anchors: keep the legacy prefix
        # cut for enumeration queries.
        if getattr(settings, "ask_chunk_window_enabled", True) and not enumeration_query:
            chunks = [_window_chunk_text(c["text"], chunk_max_chars, salient) for c in merged]
        else:
            chunks = [c["text"][:chunk_max_chars] for c in merged]
        if not chunks and doc.text:
            chunks = [doc.text[:fallback_chars]]
        payload.append(
            {
                "document_id": doc.id,
                "title": doc.title,
                "ref": doc.ref,
                "doc_kind": doc.doc_kind,
                "doc_subkind": doc.doc_subkind,
                "issue_date": issue.date.isoformat() if issue.date else None,
                "chunks": chunks,
            }
        )
    return payload


def _build_full_docs(
    target_ids: list[int], docs_by_id: dict[int, tuple[Any, Any]]
) -> list[dict[str, Any]]:
    """Full texts of the read docs, under the per-doc and total char budgets."""
    max_full_docs = int(getattr(settings, "full_doc_max_docs", 0) or 0)
    if max_full_docs <= 0:
        return []
    full_docs_local = []
    total_chars = 0
    for doc_id in target_ids:
        row = docs_by_id.get(doc_id)
        if not row:
            continue
        doc, issue = row
        if not doc.text:
            continue
        text = doc.text
        if len(text) > settings.full_doc_max_chars:
            continue
        if total_chars + len(text) > settings.full_doc_total_chars:
            break
        full_docs_local.append(
            {
                "document_id": doc.id,
                "title": doc.title,
                "ref": doc.ref,
                "issue_date": issue.date.isoformat() if issue.date else None,
                "text": text,
            }
        )
        total_chars += len(text)
        if len(full_docs_local) >= max_full_docs:
            break
    return full_docs_local


def _citation_floor_evidence(
    question: str,
    docs_payload: list[dict[str, Any]],
    payload_doc_ids: list[int],
    evidence: list[dict[str, Any]],
    max_floor: int,
) -> list[dict[str, Any]]:
    """Guarantee every doc in the reader payload is citable.

    extract_evidence drops docs with no LLM quote and no keyword coverage — for
    anchor-poor (vague/colloquial) queries and annex golds that silently discards
    the recovered gold even though it reached the reader payload with its chunks.
    For each payload doc missing from `evidence`, in confidence order (rerank-selected
    docs first, then the coverage/amount/eligibility extras) and capped at `max_floor`,
    inject its most question-relevant chunk so the answer can cite it. Pure addition:
    never reorders or drops an existing evidence item.
    """
    if max_floor <= 0 or not payload_doc_ids or not docs_payload:
        return evidence
    have: set[int] = set()
    for item in evidence:
        did = item.get("doc_id") or item.get("document_id")
        if did is not None:
            have.add(int(did))
    by_id = {int(d["document_id"]): d for d in docs_payload if d.get("document_id") is not None}
    keywords = extract_keywords_simple(question)
    added: list[dict[str, Any]] = []
    for did in payload_doc_ids:
        if len(added) >= max_floor:
            break
        did = int(did)
        if did in have:
            continue
        doc = by_id.get(did)
        if not doc:
            continue
        chunks = [c for c in (doc.get("chunks") or []) if c and c.strip()]
        if not chunks:
            continue
        best = chunks[0]
        if keywords:
            ranked = max(chunks, key=lambda c: coverage_score(c, keywords))
            if coverage_score(ranked, keywords) > 0:
                best = ranked
        have.add(did)
        added.append(
            {
                "doc_id": did,
                "quote": best.strip()[:800],
                "detail": "Extracto del documento mejor clasificado en la recuperacion.",
            }
        )
    if not added:
        return evidence
    return evidence + added


def read_docs_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    doc_ids = state.get("selected_doc_ids") or []
    # Force the backfill-pinned norm (the disposition the question names by N/YYYY,
    # found in corpus or just on-demand fetched) to the front of the read set, so the
    # norm-target citation guarantee always has it in evidence regardless of how the
    # non-deterministic rerank ordered the candidates. No-op when it was already
    # selected (dedup); only adds when the ranking would otherwise have dropped it.
    pin_ids = [int(d) for d in (state.get("norm_pin_doc_ids") or []) if d is not None]
    if pin_ids:
        doc_ids = _merge_unique_ids(pin_ids, [int(x) for x in doc_ids])
    if not doc_ids:
        elapsed = time.monotonic() - start
        logger.info("read.skip req=%s reason=no_docs elapsed=%.2fs", request_id, elapsed)
        return return_with_profile(
            state,
            "read",
            {"evidence": []},
            elapsed_seconds=round(elapsed, 3),
            skipped=True,
            reason="no_docs",
            docs=0,
            chunks=0,
            tokens_est=0,
            full_docs=0,
            full_tokens_est=0,
        )

    question = state["question"]
    keywords = extract_keywords_simple(question)
    candidate_docs = state.get("candidate_docs") or []
    top_chunks = state.get("top_chunks") or {}

    # Read set: budget-capped rerank selection + coverage/eligibility/amount extras.
    doc_ids, read_max_docs, enumeration_query = _read_budget(question, doc_ids, candidate_docs)
    doc_ids = _coverage_extras(
        question, keywords, doc_ids, candidate_docs, top_chunks, read_max_docs
    )
    doc_ids = _eligibility_extras(question, keywords, doc_ids, candidate_docs, top_chunks)
    doc_ids = _amount_extras(question, doc_ids, candidate_docs, top_chunks)
    doc_ids = _companion_extras(question, doc_ids, keywords)

    try:
        query_embedding = state.get("query_embedding")
        docs_by_id, fallback_chunks, extra_chunks, keyword_chunks, salient = _fetch_reader_inputs(
            doc_ids, question, query_embedding
        )

        confidence_min = getattr(settings, "ask_doc_confidence_min", 0.06)
        top_score = float(candidate_docs[0].get("rrf_score", 0.0)) if candidate_docs else 0.0
        high_confidence = top_score >= confidence_min

        docs_payload = _build_docs_payload(
            doc_ids,
            docs_by_id,
            top_chunks,
            fallback_chunks,
            keyword_chunks,
            extra_chunks,
            salient,
            enumeration_query,
        )
        evidence = extract_evidence(state["question"], docs_payload, full_docs=None)
        full_docs = []
        if evidence or high_confidence:
            full_docs = _build_full_docs(doc_ids, docs_by_id)
            if not evidence and full_docs:
                evidence = extract_evidence(state["question"], docs_payload, full_docs=full_docs)
                if not evidence:
                    full_docs = []
        if evidence and getattr(settings, "ask_read_citation_floor", True):
            # Convert recovered recall into a citation: make every doc that reached the
            # reader payload citable, even when the reader extracted no quote for it.
            # Walk the payload in confidence order (rerank-selected first, then the
            # coverage/amount/eligibility extras) so a gold rescued as an extra is also
            # floored. Gated on `evidence` being non-empty so a genuine no-evidence
            # abstention is never turned into a forced citation.
            evidence = _citation_floor_evidence(
                state["question"],
                docs_payload,
                doc_ids,
                evidence,
                getattr(settings, "ask_read_citation_floor_docs", 5),
            )
        chunk_count = sum(len(doc.get("chunks") or []) for doc in docs_payload)
        char_count = sum(len(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
        token_est = sum(
            estimate_tokens(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or [])
        )
        full_token_est = sum(estimate_tokens(doc.get("text") or "") for doc in full_docs)
        elapsed = time.monotonic() - start
        logger.info(
            "read.done req=%s docs=%s chunks=%s chars=%s tokens~=%s evidence=%s full_docs=%s full_chars=%s full_tokens~=%s elapsed=%.2fs",
            request_id,
            len(doc_ids),
            chunk_count,
            char_count,
            token_est,
            len(evidence),
            len(full_docs),
            sum(len(doc.get("text") or "") for doc in full_docs),
            full_token_est,
            elapsed,
        )
        return return_with_profile(
            state,
            "read",
            {"evidence": evidence, "full_docs": full_docs, "read_payload": docs_payload},
            elapsed_seconds=round(elapsed, 3),
            skipped=False,
            docs=len(doc_ids),
            chunks=chunk_count,
            chars=char_count,
            tokens_est=token_est,
            evidence=len(evidence),
            full_docs=len(full_docs),
            full_chars=sum(len(doc.get("text") or "") for doc in full_docs),
            full_tokens_est=full_token_est,
        )
    except Exception:
        logger.exception("read.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
