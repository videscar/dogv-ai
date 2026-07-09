from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any

from sqlalchemy import text as sa_text

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


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value or "") if not unicodedata.combining(ch)
    )


def _fold_preserving_length(text: str) -> str:
    """Accent-fold for offset-safe matching: every char maps to exactly one char
    (its first NFKD base char), so match positions index straight into `text`."""
    out: list[str] = []
    for ch in text.lower():
        base = "".join(
            c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)
        )
        out.append(base[0] if base else ch)
    return "".join(out)


def _window_chunk_text(text: str, cap: int, folded_keywords: list[str]) -> str:
    """Truncate a long chunk to `cap` chars keeping the question-relevant window.

    The naive prefix cut clips answer text that sits deep in a long chunk even
    though the chunk was selected FOR that content (the lexical/vector lanes match
    on the full text, the payload then shows only chars [0:cap]). When any salient
    keyword hits past the prefix half, spend half the budget on the word-aligned
    prefix (heading/context) and the other half on the window with the best
    keyword coverage. No hits past the prefix -> exact legacy prefix cut.
    """
    if len(text) <= cap:
        return text
    if not folded_keywords:
        return text[:cap]
    folded = _fold_preserving_length(text)
    half = cap // 2
    late_anchors = sorted(
        {
            match.start()
            for kw in folded_keywords
            for match in re.finditer(re.escape(kw), folded)
            if match.start() >= half
        }
    )
    if not late_anchors:
        return text[:cap]
    window = cap - half - 3  # 3 chars for the " … " joiner
    if window <= 0:
        return text[:cap]
    # Score candidate windows by the keyword kinds they add BEYOND the kept prefix
    # half. Raw kind/hit counts let a window dense in generic question words
    # (projecte, guardabosc) outrank the answer section whose distinctive keywords
    # (quantia, individualitzada) appear only once — dropping text the legacy
    # [0:cap] prefix used to show (Q7-VA: "Dotzé… 500,00" at offset 701). Marginal
    # scoring makes the prefix-redundant window lose; if no window adds any new
    # kind, keep the exact legacy prefix cut.
    prefix_kinds = {kw for kw in folded_keywords if kw in folded[:half]}
    best_score: tuple[int, int, int, int] | None = None
    best_start = half
    for anchor in late_anchors:
        start = max(half, min(anchor - 150, len(text) - window))
        segment = folded[start : start + window]
        kinds = {kw for kw in folded_keywords if kw in segment}
        new_kinds = len(kinds - prefix_kinds)
        total = sum(segment.count(kw) for kw in folded_keywords)
        score = (new_kinds, len(kinds), total, -start)
        if best_score is None or score > best_score:
            best_score = score
            best_start = start
    if best_score is None or best_score[0] == 0:
        return text[:cap]
    # Nudge both cut points back to a whitespace boundary for readability.
    prefix_end = half
    while prefix_end > half - 30 and prefix_end < len(text) and not text[prefix_end - 1].isspace():
        prefix_end -= 1
    start = best_start
    while start > best_start - 30 and start > half and not text[start - 1].isspace():
        start -= 1
    return text[:prefix_end] + " … " + text[start : start + window]


def _salient_keywords(question: str, limit: int = 8) -> list[str]:
    """Distinctive, accent-folded tokens (names, places, codes) for lexical chunk lookup."""
    out: list[str] = []
    seen: set[str] = set()
    for kw in extract_keywords_simple(question):
        if len(kw) < 5 and not any(ch.isdigit() for ch in kw):
            continue
        folded = _strip_accents(kw).lower()
        if not folded or folded in seen:
            continue
        seen.add(folded)
        out.append(folded)
        if len(out) >= limit:
            break
    return out


def _lexical_chunks_for_docs(
    db, doc_ids: list[int], patterns: list[str], per_doc: int
) -> dict[int, list[dict[str, Any]]]:
    """Per-doc chunks that literally contain the question's salient tokens.

    Accent/case-insensitive (unaccent). Catches dense annex rows (a municipality,
    a person, an expediente) that a whole-question embedding ranks too low to pass.
    """
    if not doc_ids or not patterns or per_doc <= 0:
        return {}
    params: dict[str, Any] = {"doc_ids": doc_ids, "per_doc": per_doc}
    like_parts: list[str] = []
    for idx, pat in enumerate(patterns):
        key = f"p{idx}"
        params[key] = f"%{pat}%"
        like_parts.append(f"(unaccent(lower(rc.text)) LIKE :{key})::int")
    score_sql = " + ".join(like_parts)
    where_sql = " OR ".join(f"unaccent(lower(rc.text)) LIKE :p{idx}" for idx in range(len(patterns)))
    sql = sa_text(
        f"""
        WITH ranked AS (
            SELECT rc.id AS chunk_id, rc.document_id, rc.chunk_index, rc.text,
                   ({score_sql}) AS kw_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY rc.document_id
                       ORDER BY ({score_sql}) DESC, length(rc.text) ASC, rc.id ASC
                   ) AS rn
            FROM rag_chunk rc
            WHERE rc.document_id = ANY(:doc_ids) AND ({where_sql})
        )
        SELECT chunk_id, document_id, chunk_index, text, kw_score
        FROM ranked WHERE rn <= :per_doc ORDER BY document_id, rn
        """
    )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in db.execute(sql, params).mappings().all():
        grouped.setdefault(int(row["document_id"]), []).append(dict(row))
    return grouped


def _select_chunks(sources: list[tuple[list[dict[str, Any]], int | None]], cap: int) -> list[dict[str, Any]]:
    """Merge chunk sources, deduped by chunk_index, capped at `cap`.

    Each source carries a guaranteed quota (its first `take` unique chunks are
    reserved before any source's overflow is used). This keeps the top vector-
    and keyword-matched chunks (which carry the answer) from being crowded out by
    the retrieval chunks that previously filled the whole budget, while still
    leaving room for the retrieval/BM25 chunks that explain why the doc ranked.
    """
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()

    def _key(chunk: dict[str, Any]) -> Any:
        ci = chunk.get("chunk_index")
        return ci if ci is not None else chunk.get("text")

    # Pass 1: honour each source's reserved quota, in priority order.
    for src, take in sources:
        n = 0
        for chunk in src:
            if len(out) >= cap or (take is not None and n >= take):
                break
            key = _key(chunk)
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)
            n += 1
    # Pass 2: fill any remaining budget from all sources, in priority order.
    if len(out) < cap:
        for src, _ in sources:
            for chunk in src:
                if len(out) >= cap:
                    break
                key = _key(chunk)
                if key in seen:
                    continue
                seen.add(key)
                out.append(chunk)
    return out


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
    by_id = {
        int(d["document_id"]): d
        for d in docs_payload
        if d.get("document_id") is not None
    }
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
        seen: set[int] = set()
        merged: list[int] = []
        for d in pin_ids + [int(x) for x in doc_ids]:
            if d in seen:
                continue
            seen.add(d)
            merged.append(d)
        doc_ids = merged
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
    eligibility_required = needs_eligibility(question)
    amount_extra = getattr(settings, "ask_read_amount_docs", 1)
    read_max_docs = getattr(settings, "ask_read_max_docs", 3)
    rerank_cap = getattr(settings, "ask_rerank_max_candidates", read_max_docs)
    # Enumeration queries ("list all dispositions of mayo 2026") must read the whole
    # series, not the usual handful — widen the read budget to the enumeration cap so
    # the augmented candidates actually reach the answer node.
    enumeration_query = is_enumeration_query(question)
    if getattr(settings, "enumeration_augment_enabled", False) and enumeration_query:
        rerank_cap = getattr(settings, "ask_enumeration_max_candidates", rerank_cap)
    read_max_docs = max(read_max_docs, min(len(doc_ids), rerank_cap))
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_docs = getattr(settings, "ask_read_expand_docs", 2)
    candidate_docs = state.get("candidate_docs") or []
    if candidate_docs and rrf_margin_ratio(candidate_docs, probe=expand_probe) < expand_ratio:
        read_max_docs += expand_docs
    if read_max_docs > len(doc_ids):
        read_max_docs = len(doc_ids)
    doc_ids = doc_ids[:read_max_docs]
    top_chunks = state.get("top_chunks") or {}
    coverage_extra = getattr(settings, "ask_read_coverage_docs", 2)
    if coverage_extra > 0 and candidate_docs and keywords:
        scored: list[tuple[int, int]] = []
        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            text = (item.get("title") or "")
            chunk_list = top_chunks.get(doc_id) or []
            if chunk_list:
                text = f"{text} {best_snippet(question, chunk_list)}"
            score = coverage_score(text, keywords)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        extras = [doc_id for _, doc_id in scored[:coverage_extra]]
        if extras:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + extras:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged
            max_with_extra = min(len(doc_ids), read_max_docs + coverage_extra)
            doc_ids = doc_ids[:max_with_extra]
    eligibility_extra = getattr(settings, "ask_read_eligibility_docs", 1)
    if eligibility_required and needs_amounts(question):
        eligibility_extra = max(eligibility_extra, 3)
    if eligibility_required and eligibility_extra > 0 and candidate_docs:
        base_scored: list[tuple[int, int]] = []
        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            if is_base_like(item):
                text = (item.get("title") or "")
                chunk_list = top_chunks.get(doc_id) or []
                if chunk_list:
                    text = f"{text} {best_snippet(question, chunk_list)}"
                score = coverage_score(text, keywords) if keywords else 0
                base_scored.append((score, doc_id))
        base_scored.sort(key=lambda item: item[0], reverse=True)
        base_candidates = [doc_id for _, doc_id in base_scored[:eligibility_extra]]
        if base_candidates:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + base_candidates:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged

    if amount_extra > 0 and candidate_docs and needs_amounts(question):
        scored: list[tuple[int, int]] = []

        def _amount_score(text: str) -> int:
            return len(re.findall(r"\b\d[\d.,]*\b", text))

        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            chunk_list = top_chunks.get(doc_id) or []
            text = " ".join((c.get("text") or "") for c in chunk_list)
            if not text:
                text = (item.get("summary") or item.get("text") or item.get("title") or "")
            score = _amount_score(text)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        extras = [doc_id for _, doc_id in scored[:amount_extra]]
        if extras:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + extras:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged

    try:
        top_chunks = state.get("top_chunks") or {}
        query_embedding = state.get("query_embedding")

        with SessionLocal() as db:
            docs = (
                db.query(DogvDocument, DogvIssue)
                .join(DogvIssue)
                .filter(DogvDocument.id.in_(doc_ids))
                .all()
            )
            fallback_chunks = {}
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
                rows = db.execute(
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
                ).mappings().all()
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

        def _build_payload(
            target_ids: list[int],
            retrieval_map: dict[int, list[dict[str, Any]]],
            vector_map: dict[int, list[dict[str, Any]]],
            keyword_map: dict[int, list[dict[str, Any]]],
            ordered_map: dict[int, list[dict[str, Any]]],
        ):
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

        def _build_full_docs(target_ids: list[int]) -> list[dict[str, Any]]:
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

        confidence_min = getattr(settings, "ask_doc_confidence_min", 0.06)
        top_score = float(candidate_docs[0].get("rrf_score", 0.0)) if candidate_docs else 0.0
        high_confidence = top_score >= confidence_min

        docs_payload = _build_payload(doc_ids, top_chunks, fallback_chunks, keyword_chunks, extra_chunks)
        evidence = extract_evidence(state["question"], docs_payload, full_docs=None)
        full_docs = []
        if evidence or high_confidence:
            full_docs = _build_full_docs(doc_ids)
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
        token_est = sum(estimate_tokens(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
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
