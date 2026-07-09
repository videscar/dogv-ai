"""Chunk selection and windowing for the read node.

Extracted from ``read_docs_node``: pick the chunks each document contributes to
the reader payload (retrieval / vector / literal-keyword sources under one
budget) and truncate long chunks to the question-relevant window instead of a
blind prefix cut.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from sqlalchemy import text as sa_text

from agent.shared import extract_keywords_simple


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", value or "") if not unicodedata.combining(ch)
    )


def _fold_preserving_length(text: str) -> str:
    """Accent-fold for offset-safe matching: every char maps to exactly one char
    (its first NFKD base char), so match positions index straight into `text`."""
    out: list[str] = []
    for ch in text.lower():
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
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
    where_sql = " OR ".join(
        f"unaccent(lower(rc.text)) LIKE :p{idx}" for idx in range(len(patterns))
    )
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


def _select_chunks(
    sources: list[tuple[list[dict[str, Any]], int | None]], cap: int
) -> list[dict[str, Any]]:
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
