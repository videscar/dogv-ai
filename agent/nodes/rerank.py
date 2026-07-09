from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import text as sa_text

from agent.shared import (
    QAState,
    best_snippet,
    coverage_score,
    extract_keywords_simple,
    return_with_profile,
    rrf_margin_ratio,
)
from api.config import get_settings
from api.db import SessionLocal
from api.edition_recency import suppress_stale_editions
from api.enumeration import augment_enumeration_candidates, parse_enumeration
from api.query_classifiers import guess_language
from api.rerank import prepend_recent_relevant_docs, rerank_titles
from api.retrieval import top_chunks_for_docs

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _maybe_augment_enumeration(
    state: QAState, candidates: list[dict[str, Any]], request_id: str | None
) -> list[dict[str, Any]] | None:
    """For an enumeration query, merge the month+category SQL pull into the
    candidate pool so the whole series is rerankable. Returns the widened list, or
    None when the query is not an enumeration (caller keeps the original pool)."""
    if not getattr(settings, "enumeration_augment_enabled", False):
        return None
    question = state.get("question") or ""
    spec = parse_enumeration(question)
    if spec is None:
        return None
    language = state.get("language") or guess_language(question)
    existing = {
        int(c["document_id"]) for c in candidates if c.get("document_id") is not None
    }
    cap = getattr(settings, "enumeration_augment_max", 20)
    with SessionLocal() as db:
        extra = augment_enumeration_candidates(db, spec, language, existing, cap)
    if not extra:
        return None
    logger.info(
        "rerank.enum_augment req=%s added=%s month=%s..%s groups=%s",
        request_id, len(extra), spec.date_start, spec.date_end, spec.group_codes,
    )
    return candidates + extra


def _rerank_updates(
    doc_ids: list[int],
    enumeration: list[dict[str, Any]] | None,
    all_candidates: list[dict[str, Any]],
    persist_candidates: bool = False,
) -> dict[str, Any]:
    """rerank state updates; persists the candidate pool for enumeration queries (so
    read_docs reads the augmented set) or when edition-recency pruned a stale sibling
    out of the pool (so read_docs' coverage/amount extras can't re-add it)."""
    updates: dict[str, Any] = {"selected_doc_ids": doc_ids}
    if enumeration is not None or persist_candidates:
        updates["candidate_docs"] = all_candidates
    return updates


def _apply_edition_recency(
    state: QAState,
    doc_ids: list[int],
    pool: list[dict[str, Any]],
    enumeration: list[dict[str, Any]] | None,
    request_id: str | None,
) -> tuple[list[int], list[dict[str, Any]], set[int]]:
    """RC1: drop older sibling-editions from the read set + candidate pool when the query
    has no explicit past-date target. No-op unless enabled; skipped for enumeration (which
    wants the whole series) and when a since/until filter already scopes the query to a
    period (explicit temporal intent is respected). Returns (doc_ids, pool, dropped)."""
    if not getattr(settings, "ask_edition_recency_enabled", False) or enumeration is not None:
        return doc_ids, pool, set()
    filters = state.get("filters")
    if filters is not None and (
        getattr(filters, "since_date", None) or getattr(filters, "until_date", None)
    ):
        return doc_ids, pool, set()
    if len(doc_ids) < 2:
        return doc_ids, pool, set()
    protected = [int(d) for d in (state.get("norm_pin_doc_ids") or []) if d is not None]
    with SessionLocal() as db:
        kept, dropped = suppress_stale_editions(
            db,
            doc_ids,
            pool,
            sim_threshold=float(getattr(settings, "ask_edition_recency_sim", 0.86)),
            scan_n=int(getattr(settings, "ask_edition_recency_scan_n", 12)),
            protected_ids=protected,
        )
    if not dropped:
        return doc_ids, pool, set()
    logger.info(
        "rerank.edition_recency req=%s dropped=%s kept=%s stale_ids=%s",
        request_id, len(dropped), len(kept), sorted(dropped),
    )
    pruned = [c for c in pool if int(c.get("document_id")) not in dropped]
    return kept, pruned, dropped


def _doc_similarity_scores(
    query_embedding: list[float] | None,
    doc_ids: list[int],
) -> dict[int, float]:
    if not query_embedding or not doc_ids:
        return {}
    literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
    params = {"query_embedding": literal, "doc_ids": doc_ids}
    sql = sa_text(
        """
        WITH doc_scores AS (
            SELECT document_id, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
            FROM rag_doc
            WHERE document_id = ANY(:doc_ids)
            AND embedding IS NOT NULL
        ),
        title_scores AS (
            SELECT document_id, 1 - (embedding <=> CAST(:query_embedding AS vector)) AS score
            FROM rag_title
            WHERE document_id = ANY(:doc_ids)
            AND embedding IS NOT NULL
        )
        SELECT document_id, MAX(score) AS score
        FROM (
            SELECT * FROM doc_scores
            UNION ALL
            SELECT * FROM title_scores
        ) combined
        GROUP BY document_id
        """
    )
    with SessionLocal() as db:
        rows = db.execute(sql, params).mappings().all()
    return {int(row["document_id"]): float(row["score"]) for row in rows}


def rerank_titles_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    candidates = state.get("candidate_docs") or []
    all_candidates = candidates
    try:
        question = state["question"]
        keywords = extract_keywords_simple(question)
        enumeration = _maybe_augment_enumeration(state, candidates, request_id)
        if enumeration is not None:
            candidates = enumeration
            all_candidates = enumeration  # persist the widened pool for read_docs
        max_candidates = getattr(settings, "ask_rerank_max_candidates", 10)
        top_n = getattr(settings, "ask_rerank_top_n", 5)
        read_max_docs = getattr(settings, "ask_read_max_docs", 3)
        if enumeration is not None:
            max_candidates = getattr(settings, "ask_enumeration_max_candidates", max_candidates)
            top_n = getattr(settings, "ask_enumeration_top_n", top_n)
        expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
        expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
        expand_candidates = getattr(settings, "ask_rerank_expand_candidates", 10)
        expand_top_n = getattr(settings, "ask_rerank_expand_top_n", 2)
        if candidates and rrf_margin_ratio(candidates, probe=expand_probe) < expand_ratio:
            max_candidates = min(len(candidates), max_candidates + expand_candidates)
            top_n = min(len(candidates), top_n + expand_top_n)
        doc_id_list = [int(item["document_id"]) for item in candidates]
        doc_scores = _doc_similarity_scores(state.get("query_embedding"), doc_id_list)
        # Score by max(raw-query, HyDE) similarity so HyDE-recovered docs aren't
        # demoted here by the raw-query similarity that missed them in the first place.
        hyde_scores = _doc_similarity_scores(state.get("hyde_embedding"), doc_id_list)
        if doc_scores or hyde_scores:
            def _sim(doc_id: int) -> float:
                return max(doc_scores.get(doc_id, -1.0), hyde_scores.get(doc_id, -1.0))
            candidates = sorted(
                candidates,
                key=lambda item: (
                    _sim(int(item["document_id"])),
                    float(item.get("rrf_score") or 0.0),
                ),
                reverse=True,
            )
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        top_chunks = state.get("top_chunks") or {}
        chunk_candidates = state.get("chunk_candidates") or []
        chunk_candidates_by_doc: dict[int, list[dict[str, Any]]] = {}
        if chunk_candidates:
            for item in chunk_candidates:
                doc_id = item.get("document_id")
                if doc_id is None:
                    continue
                chunk_candidates_by_doc.setdefault(int(doc_id), []).append(item)
            for items in chunk_candidates_by_doc.values():
                items.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        fallback_doc_ids = [int(item["document_id"]) for item in candidates]
        fallback_summaries: dict[int, str] = {}
        fallback_chunks: dict[int, list[dict[str, Any]]] = {}
        embeddings = state.get("query_embeddings") or []
        if not embeddings and state.get("query_embedding"):
            embeddings = [state["query_embedding"]]
        if fallback_doc_ids:
            with SessionLocal() as db:
                rows = db.execute(
                    sa_text(
                        """
                        SELECT document_id, summary
                        FROM rag_doc
                        WHERE document_id = ANY(:doc_ids)
                        AND summary IS NOT NULL
                        """
                    ),
                    {"doc_ids": fallback_doc_ids},
                ).mappings().all()
                fallback_summaries = {
                    int(row["document_id"]): (row["summary"] or "").strip() for row in rows
                }
                if embeddings:
                    best_chunks: dict[int, dict[str, Any]] = {}
                    for embedding in embeddings:
                        chunk_map = top_chunks_for_docs(
                            db,
                            embedding,
                            fallback_doc_ids,
                            per_doc=1,
                        )
                        for doc_id, items in chunk_map.items():
                            if not items:
                                continue
                            candidate = items[0]
                            score = float(candidate.get("score") or 0.0)
                            current = best_chunks.get(doc_id)
                            if current is None or score > float(current.get("score") or 0.0):
                                best_chunks[doc_id] = candidate
                    fallback_chunks = {doc_id: [chunk] for doc_id, chunk in best_chunks.items()}
        rerank_candidates = []
        for item in candidates:
            doc_id = int(item["document_id"])
            snippet = ""
            chunk_list = top_chunks.get(doc_id) or []
            if chunk_list:
                snippet = best_snippet(question, chunk_list)
            if not snippet:
                snippet = (item.get("summary") or item.get("text") or "").strip()
            if not snippet:
                snippet = fallback_summaries.get(doc_id, "")
            if not snippet and doc_id in fallback_chunks:
                snippet = best_snippet(question, fallback_chunks[doc_id])
            if snippet and keywords and coverage_score(snippet, keywords) == 0:
                candidate_chunks = chunk_candidates_by_doc.get(doc_id) or []
                if candidate_chunks:
                    improved = best_snippet(question, candidate_chunks)
                    if improved and coverage_score(improved, keywords) > 0:
                        snippet = improved
            if not snippet:
                snippet = (item.get("title") or "").strip()
            rerank_candidates.append(
                {
                    "document_id": doc_id,
                    "issue_date": item.get("issue_date"),
                    "title": item.get("title"),
                    "doc_kind": item.get("doc_kind"),
                    "doc_subkind": item.get("doc_subkind"),
                    "ref": item.get("ref"),
                    "snippet": snippet,
                }
            )

        llm_top_n = min(len(rerank_candidates), max(top_n, read_max_docs))
        if len(rerank_candidates) <= llm_top_n:
            doc_ids = [item["document_id"] for item in rerank_candidates]
            doc_ids, all_candidates, dropped = _apply_edition_recency(
                state, doc_ids, all_candidates, enumeration, request_id
            )
            elapsed = time.monotonic() - start
            logger.info(
                "rerank.skip req=%s candidates=%s elapsed=%.2fs",
                request_id,
                len(rerank_candidates),
                elapsed,
            )
            return return_with_profile(
                state,
                "rerank",
                _rerank_updates(doc_ids, enumeration, all_candidates, persist_candidates=bool(dropped)),
                elapsed_seconds=round(elapsed, 3),
                skipped=True,
                selected_docs=len(doc_ids),
                candidate_docs=len(rerank_candidates),
            )
        doc_ids = rerank_titles(
            question,
            rerank_candidates,
            top_n=llm_top_n,
            return_all=True,
        )
        coverage_keep = []
        coverage_keep_n = getattr(settings, "ask_rerank_coverage_keep", 2)
        if coverage_keep_n > 0 and keywords:
            threshold = 1 if len(keywords) <= 3 else 2
            scored = []
            for item in rerank_candidates:
                doc_id = int(item["document_id"])
                text = f"{item.get('title') or ''} {item.get('snippet') or ''}"
                score = coverage_score(text, keywords)
                if score >= threshold:
                    scored.append((score, doc_id))
            scored.sort(key=lambda item: item[0], reverse=True)
            coverage_keep = [doc_id for _, doc_id in scored[:coverage_keep_n]]
        if coverage_keep:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in coverage_keep + doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged
        if not doc_ids:
            doc_ids = [item["document_id"] for item in rerank_candidates]
        doc_ids = prepend_recent_relevant_docs(question, keywords, rerank_candidates, doc_ids)
        doc_ids, all_candidates, dropped = _apply_edition_recency(
            state, doc_ids, all_candidates, enumeration, request_id
        )
        elapsed = time.monotonic() - start
        logger.info(
            "rerank.done req=%s selected=%s elapsed=%.2fs",
            request_id,
            len(doc_ids),
            elapsed,
        )
        return return_with_profile(
            state,
            "rerank",
            _rerank_updates(doc_ids, enumeration, all_candidates, persist_candidates=bool(dropped)),
            elapsed_seconds=round(elapsed, 3),
            skipped=False,
            selected_docs=len(doc_ids),
            candidate_docs=len(rerank_candidates),
        )
    except Exception:
        logger.exception("rerank.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
