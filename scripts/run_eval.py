from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import json
import os
import re
import time
from typing import Any

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from api.config import enabled_lanes, get_settings
from api.db import SessionLocal
from api.intent import analyze_intent
from api.ollama import OllamaClient
from api.query_expansion import (
    build_bm25_queries,
    build_prf_query,
    decompose_question,
    guess_language,
    is_relative_time_query,
    llm_expand_query,
)
from api.rerank import rerank_titles
from api.retrieval import (
    RetrievalFilters,
    bm25_search,
    rrf_fuse,
    top_chunks_for_docs,
    title_bm25_search,
    title_vector_search,
    vector_search,
)
from api.taxonomy import canonical_doc_kind

settings = get_settings()
LANES = enabled_lanes(settings)
BM25_LIMIT = max(10, settings.ask_bm25_max_docs)


@dataclass
class StageTimings:
    embed: float = 0.0
    vector: float = 0.0
    bm25: float = 0.0
    title: float = 0.0
    hybrid: float = 0.0
    hybrid_nofilter: float = 0.0
    rerank: float = 0.0


def _parse_k_values(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return sorted(set(items))


def _format_candidates(rows: list[dict[str, Any]], score_key: str, limit: int) -> list[dict[str, Any]]:
    formatted = []
    for rank, row in enumerate(rows[:limit], start=1):
        formatted.append(
            {
                "rank": rank,
                "document_id": int(row["document_id"]),
                "score": float(row.get(score_key, 0.0) or 0.0),
            }
        )
    return formatted


def _dedupe_docs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(item)
    return deduped


def _merge_docs(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in primary + secondary:
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(item)
    return merged


def _merge_with_budget(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    limit: int,
    secondary_ratio: float = 0.3,
) -> list[dict[str, Any]]:
    if not primary:
        return secondary[:limit]
    if not secondary:
        return primary[:limit]
    budget = max(5, int(limit * secondary_ratio))
    keep_primary = max(1, limit - budget)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _add(items: list[dict[str, Any]]) -> None:
        for item in items:
            if len(result) >= limit:
                return
            doc_id = item.get("document_id")
            if doc_id is None:
                continue
            doc_id = int(doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result.append(item)

    _add(primary[:keep_primary])
    added_secondary = 0
    for item in secondary:
        if len(result) >= limit or added_secondary >= budget:
            break
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        result.append(item)
        added_secondary += 1

    if len(result) < limit:
        _add(primary[keep_primary:])
    return result


def _extract_ref_tokens(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"\b\d{4}/\d+\b", text)
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:6]


def _find_ref_hits(
    db: Session,
    question: str,
    filters: RetrievalFilters,
    per_token_limit: int = 5,
) -> list[dict[str, Any]]:
    tokens = _extract_ref_tokens(question)
    if not tokens:
        return []

    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _run_for_token(token: str, use_language: bool) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "exact_ref": token,
            "like_ref": f"%{token}%",
            "limit": per_token_limit,
        }
        clauses = ["(dd.ref = :exact_ref OR dd.ref ILIKE :like_ref OR dd.title ILIKE :like_ref)"]
        if use_language and filters.language:
            clauses.append("di.language = :language")
            params["language"] = filters.language
        if filters.since_date:
            clauses.append("di.date >= :since_date")
            params["since_date"] = filters.since_date
        if filters.until_date:
            clauses.append("di.date <= :until_date")
            params["until_date"] = filters.until_date

        where_sql = " AND ".join(clauses)
        rows = db.execute(
            sa_text(
                f"""
                SELECT
                    dd.id AS document_id,
                    dd.title,
                    dd.ref,
                    dd.type,
                    dd.doc_kind,
                    dd.doc_subkind,
                    dd.pdf_url,
                    dd.html_url,
                    di.date AS issue_date
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                WHERE {where_sql}
                ORDER BY
                    CASE
                        WHEN dd.ref = :exact_ref THEN 0
                        WHEN dd.ref ILIKE :like_ref THEN 1
                        ELSE 2
                    END,
                    di.date DESC,
                    dd.id DESC
                LIMIT :limit
                """
            ),
            params,
        ).mappings().all()
        return [dict(row) for row in rows]

    for token in tokens:
        token_hits = _run_for_token(token, use_language=True)
        if not token_hits and filters.language:
            token_hits = _run_for_token(token, use_language=False)
        for rank, row in enumerate(token_hits, start=1):
            doc_id = int(row["document_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            enriched = dict(row)
            # Strong lexical anchor for ref/CVE-style lookups.
            enriched["score"] = max(1.0, 5.0 - (rank * 0.1))
            found.append(enriched)

    return found


def _collect_facet_specs(
    question: str,
    intent: dict[str, Any],
    client: OllamaClient,
    embed_cache: dict[str, list[float]],
) -> tuple[list[list[float]], list[tuple[str, str | None]]]:
    max_facets = max(1, getattr(settings, "ask_max_facets", 3))
    facet_questions = [question]
    facets = decompose_question(question, max_facets=max_facets)
    if facets:
        seen = {question.strip().lower()}
        for facet in facets:
            key = facet.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            facet_questions.append(facet)
            if len(facet_questions) >= max_facets:
                break

    if question in embed_cache:
        embeddings = [embed_cache[question]]
    else:
        embedding = client.embed(question)
        embed_cache[question] = embedding
        embeddings = [embedding]
    expansion = llm_expand_query(question, intent) if settings.ask_llm_expand else {}
    specs: list[tuple[str, str | None]] = []
    for idx, facet_question in enumerate(facet_questions):
        if idx == 0:
            bm25_query, bm25_strict_query = build_bm25_queries(
                facet_question,
                intent,
                expansion=expansion,
            )
        else:
            bm25_query, bm25_strict_query = build_bm25_queries(facet_question, intent)
        specs.append((bm25_query, bm25_strict_query))
    return embeddings, specs


def _attach_snippets(
    db: Session,
    candidates: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    if not candidates or not embeddings:
        return candidates
    doc_ids = [int(item["document_id"]) for item in candidates]
    best_chunks: dict[int, dict[str, Any]] = {}
    for embedding in embeddings:
        chunk_map = top_chunks_for_docs(db, embedding, doc_ids, per_doc=1)
        for doc_id, items in chunk_map.items():
            if not items:
                continue
            candidate = items[0]
            score = float(candidate.get("score") or 0.0)
            current = best_chunks.get(doc_id)
            if current is None or score > float(current.get("score") or 0.0):
                best_chunks[doc_id] = candidate
    enriched: list[dict[str, Any]] = []
    for item in candidates:
        doc_id = int(item["document_id"])
        snippet = best_chunks.get(doc_id, {}).get("text")
        if snippet:
            enriched_item = dict(item)
            enriched_item["snippet"] = snippet
            enriched.append(enriched_item)
        else:
            enriched.append(item)
    return enriched


def _format_rerank(doc_ids: list[int], limit: int) -> list[dict[str, Any]]:
    formatted = []
    for rank, doc_id in enumerate(doc_ids[:limit], start=1):
        formatted.append({"rank": rank, "document_id": int(doc_id)})
    return formatted


def _normalize_gold_sets(entry: dict[str, Any]) -> list[list[int]]:
    raw_sets = entry.get("gold_sets")
    sets: list[list[int]] = []

    def _normalize(items: Any) -> list[int]:
        if not isinstance(items, list):
            return []
        seen: set[int] = set()
        out: list[int] = []
        for value in items:
            try:
                doc_id = int(value)
            except (TypeError, ValueError):
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc_id)
        return sorted(out)

    if isinstance(raw_sets, list):
        seen_sets: set[tuple[int, ...]] = set()
        for item in raw_sets:
            normalized = _normalize(item)
            if not normalized:
                continue
            key = tuple(normalized)
            if key in seen_sets:
                continue
            seen_sets.add(key)
            sets.append(normalized)
        if sets:
            return sets

    raw_doc_ids = entry.get("doc_ids")
    doc_ids = _normalize(raw_doc_ids)
    if doc_ids:
        return [doc_ids]

    raw_doc_id = entry.get("doc_id")
    if raw_doc_id is not None:
        try:
            return [[int(raw_doc_id)]]
        except (TypeError, ValueError):
            return []
    return []


def _compute_hits(doc_ids: list[int], gold_sets: list[list[int]], k_values: list[int]) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    if not gold_sets:
        for k in k_values:
            hits[str(k)] = False
        return hits
    gold = [set(group) for group in gold_sets if group]
    if not gold:
        for k in k_values:
            hits[str(k)] = False
        return hits
    for k in k_values:
        topk = set(doc_ids[:k])
        hits[str(k)] = any(group.issubset(topk) for group in gold)
    return hits


def _rrf_margin_ratio(candidates: list[dict[str, Any]], probe: int = 5) -> float:
    if not candidates:
        return 0.0
    top_score = float(candidates[0].get("rrf_score") or 0.0)
    if top_score <= 0:
        return 0.0
    idx = min(len(candidates) - 1, max(1, probe - 1))
    pivot = float(candidates[idx].get("rrf_score") or 0.0)
    return (top_score - pivot) / top_score


def _filters_to_dict(filters: RetrievalFilters) -> dict[str, Any]:
    return {
        "language": filters.language,
        "doc_kind": filters.doc_kind,
        "doc_subkind": filters.doc_subkind,
        "since_date": filters.since_date.isoformat() if filters.since_date else None,
        "until_date": filters.until_date.isoformat() if filters.until_date else None,
    }


def _normalize_intent_filters(
    question: str,
    intent: dict[str, Any],
    has_kind: bool,
) -> RetrievalFilters:
    lang = intent.get("language")
    if lang == "ca":
        language = "va_va"
    elif lang == "es":
        language = "es_es"
    else:
        language = guess_language(question)
    since_date = intent.get("since_date")
    until_date = intent.get("until_date")
    if is_relative_time_query(question) and (settings.ask_temporal_policy or "").lower() == "filter":
        today = date.today()
        window_start = today - timedelta(days=settings.feed_recent_days)
        if since_date is None or since_date > window_start:
            since_date = window_start
        if until_date is None:
            until_date = today
    return RetrievalFilters(
        language=language,
        doc_kind=None,
        doc_subkind=None,
        since_date=since_date,
        until_date=until_date,
    )


def _run_sources(
    db: Session,
    query_embedding: list[float],
    bm25_query: str,
    bm25_strict_query: str | None,
    filters: RetrievalFilters,
    limit: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    vector_hits = vector_search(db, query_embedding, filters, limit=limit) if "vector" in LANES else []
    bm25_limit = BM25_LIMIT
    bm25_hits = bm25_search(db, bm25_query, filters, limit=bm25_limit) if "bm25" in LANES else []
    if bm25_strict_query and "bm25" in LANES:
        bm25_strict_hits = bm25_search(db, bm25_strict_query, filters, limit=bm25_limit)
    else:
        bm25_strict_hits = []
    title_hits_raw = title_vector_search(db, query_embedding, filters, limit=limit) if "title" in LANES else []
    title_lexical_hits = (
        title_bm25_search(db, bm25_query, filters, limit=bm25_limit) if "title" in LANES else []
    )
    bm25_hits = _dedupe_docs(bm25_hits)
    bm25_strict_hits = _dedupe_docs(bm25_strict_hits)
    title_lexical_hits = _dedupe_docs(title_lexical_hits)
    strict_min = getattr(settings, "bm25_strict_primary_min", 10)
    if bm25_strict_hits and len(bm25_strict_hits) >= strict_min:
        primary_hits = bm25_strict_hits
        secondary_sources = []
        secondary_weights = []
        if bm25_hits:
            secondary_sources.append(bm25_hits)
            secondary_weights.append(getattr(settings, "bm25_fuse_weight_chunk", 1.0))
        if title_lexical_hits:
            secondary_sources.append(title_lexical_hits)
            secondary_weights.append(getattr(settings, "bm25_fuse_weight_title", 0.9))
    else:
        primary_hits = bm25_hits
        secondary_sources = []
        secondary_weights = []
        if bm25_strict_hits:
            secondary_sources.append(bm25_strict_hits)
            secondary_weights.append(getattr(settings, "bm25_fuse_weight_strict", 1.2))
        if title_lexical_hits:
            secondary_sources.append(title_lexical_hits)
            secondary_weights.append(getattr(settings, "bm25_fuse_weight_title", 0.9))
    secondary_hits = _combine_sources(
        secondary_sources,
        max_docs=bm25_limit,
        weights=secondary_weights,
    ) if secondary_sources else []
    bm25_hits = _merge_with_budget(primary_hits, secondary_hits, bm25_limit)
    return _dedupe_docs(vector_hits), _dedupe_docs(bm25_hits), _dedupe_docs(title_hits_raw), title_lexical_hits


def _combine_sources(
    sources: list[list[dict[str, Any]]],
    max_docs: int,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not sources:
        return []
    if len(sources) == 1:
        return _dedupe_docs(sources[0])
    use_weights = weights if weights and len(weights) == len(sources) else [1.0] * len(sources)
    return rrf_fuse(sources, max_docs=max_docs, weights=use_weights)


def _run_sources_all_facets(
    db: Session,
    question: str,
    query_embedding: list[float],
    bm25_specs: list[tuple[str, str | None]],
    filters: RetrievalFilters,
    limit: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    vector_hits, _, title_hits, title_lexical_hits = _run_sources(
        db, query_embedding, bm25_specs[0][0], bm25_specs[0][1], filters, limit
    )
    bm25_runs = [
        _run_sources(db, query_embedding, bm25_query, bm25_strict_query, filters, limit)
        for bm25_query, bm25_strict_query in bm25_specs
    ]
    prf_used = False
    if bm25_runs:
        prf_query = build_prf_query(
            bm25_specs[0][0],
            bm25_runs[0][1],
            max_docs=getattr(settings, "bm25_prf_docs", 5),
            max_terms=getattr(settings, "bm25_prf_terms", 6),
        )
        if prf_query:
            bm25_runs.append(_run_sources(db, query_embedding, prf_query, None, filters, limit))
            prf_used = True
    bm25_primary = bm25_runs[0][1] if bm25_runs else []
    secondary_sources = [run[1] for run in bm25_runs[1:] if run[1]]
    secondary_hits = _combine_sources(secondary_sources, max_docs=BM25_LIMIT) if secondary_sources else []
    bm25_hits = _merge_with_budget(bm25_primary, secondary_hits, BM25_LIMIT)
    ref_hits = _find_ref_hits(db, question, filters, per_token_limit=5)
    if ref_hits:
        # Inject deterministic ref matches as high-confidence anchors.
        bm25_hits = _merge_docs(ref_hits, bm25_hits)
        title_hits = _merge_docs(ref_hits, title_hits)
        title_lexical_hits = _merge_docs(ref_hits, title_lexical_hits)
    counts = {
        "vector": len(vector_hits),
        "bm25": len(bm25_hits),
        "title": len(title_hits),
        "title_lexical": len(title_lexical_hits),
        "ref_hits": len(ref_hits),
        "facets": len(bm25_specs),
        "bm25_prf": 1 if prf_used else 0,
    }
    return vector_hits, bm25_hits, title_hits, title_lexical_hits, counts


def _compute_hybrid(
    db: Session,
    question: str,
    query_embedding: list[float],
    bm25_specs: list[tuple[str, str | None]],
    filters: RetrievalFilters,
    limit: int,
    precomputed: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], bool, bool]:
    if precomputed:
        vector_hits, bm25_hits, title_hits, title_lexical_hits, counts = precomputed
    else:
        vector_hits, bm25_hits, title_hits, title_lexical_hits, counts = _run_sources_all_facets(
            db, question, query_embedding, bm25_specs, filters, limit
        )
    min_docs = max(1, getattr(settings, "ask_min_docs", 3))
    sources: list[list[dict[str, Any]]] = []
    weights: list[float] = []
    if "vector" in LANES:
        sources.append(vector_hits)
        weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
    if "bm25" in LANES:
        sources.append(bm25_hits)
        weights.append(getattr(settings, "ask_rrf_weight_bm25", 0.5))
    if "title" in LANES:
        sources.append(title_hits)
        weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
        if title_lexical_hits and len(title_hits) < min_docs:
            sources.append(_dedupe_docs(title_lexical_hits))
            weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))
    fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
    soft_language = False
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_docs = getattr(settings, "ask_max_docs_expand", 20)
    if filters.language:
        margin = _rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
        if len(fused) < min_docs or margin < expand_ratio:
            relaxed_filters = RetrievalFilters(
                language=None,
                doc_kind=filters.doc_kind,
                doc_subkind=filters.doc_subkind,
                since_date=filters.since_date,
                until_date=filters.until_date,
            )
            (
                relaxed_vector,
                relaxed_bm25,
                relaxed_title,
                relaxed_title_lexical,
                relaxed_counts,
            ) = _run_sources_all_facets(db, question, query_embedding, bm25_specs, relaxed_filters, limit)
            if "vector" in LANES:
                sources.append(relaxed_vector)
                weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
            if "bm25" in LANES:
                sources.append(relaxed_bm25)
                weights.append(getattr(settings, "ask_rrf_weight_bm25", 0.5))
            if "title" in LANES:
                sources.append(relaxed_title)
                weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
                if relaxed_title_lexical and len(relaxed_title) < min_docs:
                    sources.append(_dedupe_docs(relaxed_title_lexical))
                    weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))
            counts.update(
                {
                    "vector_lang_relaxed": len(relaxed_vector),
                    "bm25_lang_relaxed": len(relaxed_bm25),
                    "title_lang_relaxed": len(relaxed_title),
                    "title_lexical_lang_relaxed": len(relaxed_title_lexical),
                    "facets_lang_relaxed": relaxed_counts.get("facets"),
                }
            )
            fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
            soft_language = True
    rrf_expanded = False
    if fused and _rrf_margin_ratio(fused, probe=expand_probe) < expand_ratio:
        fused = rrf_fuse(
            sources,
            max_docs=settings.ask_max_docs + expand_docs,
            weights=weights,
        )
        rrf_expanded = True
    return fused, counts, rrf_expanded, soft_language


def _compute_hybrid_with_fallbacks(
    db: Session,
    question: str,
    query_embedding: list[float],
    bm25_specs: list[tuple[str, str | None]],
    filters: RetrievalFilters,
    limit: int,
    precomputed: tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, int],
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], RetrievalFilters, list[str], dict[str, int], bool]:
    fused, counts, rrf_expanded, soft_language = _compute_hybrid(
        db, question, query_embedding, bm25_specs, filters, limit, precomputed=precomputed
    )
    fallbacks: list[str] = []
    if soft_language:
        fallbacks.append("soft_language")

    min_docs = max(1, getattr(settings, "ask_min_docs", 3))
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    current_filters = filters

    def _relax(new_filters: RetrievalFilters, reason: str, allow_margin: bool = False) -> None:
        nonlocal fused, counts, current_filters, rrf_expanded
        if len(fused) >= min_docs:
            if not allow_margin:
                return
            margin = _rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
            if margin >= expand_ratio:
                return
        current_filters = new_filters
        fused, counts, rrf_expanded_inner, soft_language_inner = _compute_hybrid(
            db, question, query_embedding, bm25_specs, current_filters, limit
        )
        if soft_language_inner:
            fallbacks.append("soft_language")
        fallbacks.append(reason)
        if rrf_expanded_inner:
            rrf_expanded = True

    if filters.doc_kind:
        _relax(
            RetrievalFilters(
                language=filters.language,
                doc_kind=None,
                doc_subkind=None,
                since_date=filters.since_date,
                until_date=filters.until_date,
            ),
            "drop_doc_kind",
            allow_margin=True,
        )
    if filters.language:
        _relax(
            RetrievalFilters(
                language=None,
                doc_kind=current_filters.doc_kind,
                doc_subkind=current_filters.doc_subkind,
                since_date=current_filters.since_date,
                until_date=current_filters.until_date,
            ),
            "drop_language",
            allow_margin=True,
        )
    if current_filters.since_date or current_filters.until_date:
        _relax(
            RetrievalFilters(
                language=current_filters.language,
                doc_kind=current_filters.doc_kind,
                doc_subkind=current_filters.doc_subkind,
                since_date=None,
                until_date=None,
            ),
            "drop_dates",
            allow_margin=True,
        )
    _relax(RetrievalFilters(), "no_filters")

    return fused, current_filters, fallbacks, counts, rrf_expanded


def _rerank(
    db: Session,
    embeddings: list[list[float]],
    question: str,
    candidates: list[dict[str, Any]],
    top_n: int,
    max_candidates: int,
    skip_llm: bool = False,
) -> list[int]:
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_candidates = getattr(settings, "ask_rerank_expand_candidates", 10)
    expand_top_n = getattr(settings, "ask_rerank_expand_top_n", 2)
    if candidates and _rrf_margin_ratio(candidates, probe=expand_probe) < expand_ratio:
        max_candidates = min(len(candidates), max_candidates + expand_candidates)
        top_n = min(len(candidates), top_n + expand_top_n)
    trimmed = candidates[:max_candidates] if len(candidates) > max_candidates else candidates
    base_ids = [int(item["document_id"]) for item in trimmed]
    if skip_llm:
        return base_ids
    trimmed = _attach_snippets(db, trimmed, embeddings)
    if len(trimmed) <= 5:
        return base_ids
    coverage_keep = max(0, int(getattr(settings, "ask_rerank_coverage_keep", 0)))
    doc_ids = rerank_titles(
        question,
        trimmed,
        top_n=max(top_n, coverage_keep),
        return_all=True,
    )
    if not doc_ids:
        doc_ids = base_ids

    merged: list[int] = []
    seen: set[int] = set()
    for doc_id in doc_ids + base_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)

    # Guardrail: keep a small hybrid head visible in the early rerank window
    # without overriding the model's top-1 decision.
    if coverage_keep > 0 and merged:
        window = max(top_n, coverage_keep * 2)
        for doc_id in reversed(base_ids[:coverage_keep]):
            if doc_id in merged[:window]:
                continue
            if doc_id in merged:
                merged.remove(doc_id)
            insert_at = min(len(merged), window - 1 if window > 0 else 0)
            merged.insert(insert_at, doc_id)

    return merged


def _safe_analyze_intent(question: str) -> dict[str, Any]:
    try:
        return analyze_intent(question)
    except Exception as exc:
        print(f"[warn] intent failed: {exc} question={question!r}")
        return {
            "language": None,
            "doc_kind": None,
            "doc_subkind": None,
            "keywords": [],
            "since_date": None,
            "until_date": None,
            "needs_online": False,
            "entities": {},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_set.json")
    parser.add_argument("--output-dir", default="data/eval_reports")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--k-values", default="1,3,5,10,20,50")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--include-nofilter", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--fast-intent", action="store_true")
    parser.add_argument("--skip-rerank-llm", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        eval_set = json.load(fh)

    if not eval_set:
        raise SystemExit("Eval set is empty")

    k_values = _parse_k_values(args.k_values)
    if not k_values:
        raise SystemExit("No k values provided")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.output_dir, exist_ok=True)

    client = OllamaClient()
    embed_cache: dict[str, list[float]] = {}

    summary_counts: dict[str, dict[str, int]] = {}
    summary_totals: dict[str, int] = {}
    by_kind_counts: dict[str, dict[str, dict[str, int]]] = {}
    by_kind_totals: dict[str, int] = {}
    miss_samples: dict[str, list[dict[str, Any]]] = {}
    latency_totals: dict[str, float] = {
        "embed": 0.0,
        "vector": 0.0,
        "bm25": 0.0,
        "title": 0.0,
        "hybrid": 0.0,
        "hybrid_nofilter": 0.0,
        "rerank": 0.0,
    }

    rows_for_csv: list[dict[str, Any]] = []
    query_results: list[dict[str, Any]] = []

    with SessionLocal() as db:
        has_kind = bool(
            db.execute(sa_text("SELECT 1 FROM dogv_documents WHERE doc_kind IS NOT NULL LIMIT 1")).scalar()
        )
        total = len(eval_set)
        for idx, entry in enumerate(eval_set, start=1):
            question = entry["question"]
            gold_sets = _normalize_gold_sets(entry)
            if not gold_sets:
                raise SystemExit(f"Entry id={entry.get('id')} has no valid gold_sets/doc_ids/doc_id")
            gold_doc_ids = sorted({doc_id for group in gold_sets for doc_id in group})
            doc_kind = entry.get("doc_kind")
            doc_subkind = entry.get("doc_subkind")
            language = entry.get("language")

            timings = StageTimings()

            if args.fast_intent:
                intent = {
                    "language": None,
                    "doc_kind": None,
                    "doc_subkind": None,
                    "keywords": [],
                    "since_date": None,
                    "until_date": None,
                    "needs_online": False,
                    "entities": {},
                }
            else:
                intent = _safe_analyze_intent(question)
            filters = _normalize_intent_filters(question, intent, has_kind)
            embed_start = time.perf_counter()
            embeddings, bm25_specs = _collect_facet_specs(question, intent, client, embed_cache)
            timings.embed = time.perf_counter() - embed_start
            bm25_query_main = bm25_specs[0][0] if bm25_specs else question

            sources_start = time.perf_counter()
            (
                vector_hits,
                bm25_hits,
                title_hits,
                title_lexical_hits,
                counts_sources,
            ) = _run_sources_all_facets(
                db,
                question,
                embeddings[0],
                bm25_specs,
                filters,
                args.max_candidates,
            )
            sources_elapsed = time.perf_counter() - sources_start
            timings.vector = sources_elapsed
            timings.bm25 = sources_elapsed
            timings.title = sources_elapsed

            precomputed = (vector_hits, bm25_hits, title_hits, title_lexical_hits, counts_sources)

            hybrid_start = time.perf_counter()
            hybrid_final, final_filters, fallbacks, counts_final, rrf_expanded = _compute_hybrid_with_fallbacks(
                db,
                question,
                embeddings[0],
                bm25_specs,
                filters,
                args.max_candidates,
                precomputed=precomputed,
            )
            timings.hybrid = time.perf_counter() - hybrid_start

            hybrid_filtered, counts_filtered, rrf_expanded_initial, _ = _compute_hybrid(
                db,
                question,
                embeddings[0],
                bm25_specs,
                filters,
                args.max_candidates,
                precomputed=precomputed,
            )

            hybrid_nofilter = []
            if args.include_nofilter:
                nofilter_start = time.perf_counter()
                hybrid_nofilter, _, _, _ = _compute_hybrid(
                    db,
                    question,
                    embeddings[0],
                    bm25_specs,
                    RetrievalFilters(),
                    args.max_candidates,
                )
                timings.hybrid_nofilter = time.perf_counter() - nofilter_start

            rerank_start = time.perf_counter()
            rerank_base_candidates = getattr(settings, "ask_rerank_max_candidates", 10)
            reranked_ids = _rerank(
                db,
                embeddings,
                question,
                hybrid_final,
                top_n=args.rerank_top_n,
                max_candidates=rerank_base_candidates,
                skip_llm=args.skip_rerank_llm,
            )
            timings.rerank = time.perf_counter() - rerank_start

            candidates = {
                "vector": _format_candidates(vector_hits, "score", args.max_candidates),
                "bm25": _format_candidates(bm25_hits, "score", args.max_candidates),
                "title": _format_candidates(title_hits, "score", args.max_candidates),
                "hybrid": _format_candidates(hybrid_filtered, "rrf_score", args.max_candidates),
                "hybrid_final": _format_candidates(hybrid_final, "rrf_score", args.max_candidates),
            }
            if args.include_nofilter:
                candidates["hybrid_nofilter"] = _format_candidates(
                    hybrid_nofilter, "rrf_score", args.max_candidates
                )
            candidates["rerank"] = _format_rerank(reranked_ids, args.max_candidates)

            stage_docs = {
                "vector": [row["document_id"] for row in candidates["vector"]],
                "bm25": [row["document_id"] for row in candidates["bm25"]],
                "title": [row["document_id"] for row in candidates["title"]],
                "hybrid": [row["document_id"] for row in candidates["hybrid_final"]],
                "rerank": [row["document_id"] for row in candidates["rerank"]],
            }
            if args.include_nofilter:
                stage_docs["hybrid_nofilter"] = [
                    row["document_id"] for row in candidates["hybrid_nofilter"]
                ]

            hits = {stage: _compute_hits(doc_ids, gold_sets, k_values) for stage, doc_ids in stage_docs.items()}

            for stage, stage_hits in hits.items():
                summary_counts.setdefault(stage, {str(k): 0 for k in k_values})
                summary_totals.setdefault(stage, 0)
                summary_totals[stage] += 1
                for k, hit in stage_hits.items():
                    if hit:
                        summary_counts[stage][k] += 1

                if not stage_hits.get(str(k_values[-1]), False):
                    miss_samples.setdefault(stage, [])
                    if len(miss_samples[stage]) < 20:
                        miss_samples[stage].append(
                            {
                                "id": entry.get("id"),
                                "gold_sets": gold_sets,
                                "question": question,
                                "doc_kind": doc_kind,
                                "doc_subkind": doc_subkind,
                            }
                        )

            if doc_kind:
                kind_key = canonical_doc_kind(doc_kind) or doc_kind
                by_kind_totals[kind_key] = by_kind_totals.get(kind_key, 0) + 1
                by_kind_counts.setdefault(kind_key, {})
                for stage, stage_hits in hits.items():
                    by_kind_counts[kind_key].setdefault(stage, {str(k): 0 for k in k_values})
                    for k, hit in stage_hits.items():
                        if hit:
                            by_kind_counts[kind_key][stage][k] += 1

            latency_totals["embed"] += timings.embed
            latency_totals["vector"] += timings.vector
            latency_totals["bm25"] += timings.bm25
            latency_totals["title"] += timings.title
            latency_totals["hybrid"] += timings.hybrid
            latency_totals["hybrid_nofilter"] += timings.hybrid_nofilter
            latency_totals["rerank"] += timings.rerank

            query_results.append(
                {
                    "id": entry.get("id"),
                    "question": question,
                    "gold_sets": gold_sets,
                    "gold_doc_ids": gold_doc_ids,
                    "doc_kind": doc_kind,
                    "doc_subkind": doc_subkind,
                    "language": language,
                    "bm25_query": bm25_query_main,
                    "filters": _filters_to_dict(filters),
                    "filters_final": _filters_to_dict(final_filters),
                    "fallbacks": fallbacks,
                    "counts": counts_final,
                    "counts_initial": counts_filtered,
                    "rrf_expanded": rrf_expanded,
                    "rrf_expanded_initial": rrf_expanded_initial,
                    "candidates": candidates,
                    "hits": hits,
                    "timings": asdict(timings),
                }
            )

            for stage, rows in candidates.items():
                for row in rows:
                    rows_for_csv.append(
                        {
                            "query_id": entry.get("id"),
                            "stage": stage,
                            "rank": row.get("rank"),
                            "document_id": row.get("document_id"),
                            "score": row.get("score"),
                            "in_gold": row.get("document_id") in gold_doc_ids,
                        }
                    )

            if idx == total or idx % 10 == 0:
                print(f"[progress] {idx}/{total} queries processed", flush=True)

    summary = {}
    for stage, counts in summary_counts.items():
        total = summary_totals.get(stage, 0) or 1
        summary[stage] = {f"recall@{k}": counts[str(k)] / total for k in k_values}

    by_kind_summary: dict[str, dict[str, dict[str, float]]] = {}
    for kind, stage_counts in by_kind_counts.items():
        total = by_kind_totals.get(kind, 0) or 1
        by_kind_summary[kind] = {}
        for stage, counts in stage_counts.items():
            by_kind_summary[kind][stage] = {f"recall@{k}": counts[str(k)] / total for k in k_values}

    avg_latency = {key: value / len(eval_set) for key, value in latency_totals.items()}

    report = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "eval_set": args.input,
        "size": len(eval_set),
        "k_values": k_values,
        "settings": {
            "ask_max_docs": settings.ask_max_docs,
            "ask_min_docs": settings.ask_min_docs,
            "ask_read_max_docs": settings.ask_read_max_docs,
            "ask_chunks_per_doc": settings.ask_chunks_per_doc,
            "ask_rerank_top_n": settings.ask_rerank_top_n,
            "ask_rerank_max_candidates": settings.ask_rerank_max_candidates,
            "ask_doc_confidence_min": settings.ask_doc_confidence_min,
            "ask_bm25_max_docs": settings.ask_bm25_max_docs,
            "feed_recent_days": settings.feed_recent_days,
            "bm25_fuse_weight_chunk": settings.bm25_fuse_weight_chunk,
            "bm25_fuse_weight_strict": settings.bm25_fuse_weight_strict,
            "bm25_fuse_weight_title": settings.bm25_fuse_weight_title,
            "bm25_strict_primary_min": settings.bm25_strict_primary_min,
            "ask_rrf_weight_vector": settings.ask_rrf_weight_vector,
            "ask_rrf_weight_bm25": settings.ask_rrf_weight_bm25,
            "ask_rrf_weight_title": settings.ask_rrf_weight_title,
            "ask_rrf_weight_title_lexical": settings.ask_rrf_weight_title_lexical,
            "chunk_min_tokens": settings.chunk_min_tokens,
            "chunk_max_tokens": settings.chunk_max_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "ollama_model": settings.ollama_model,
            "ollama_embed_model": settings.ollama_embed_model,
            "fast_intent": args.fast_intent,
            "skip_rerank_llm": args.skip_rerank_llm,
        },
        "summary": {
            "recall": summary,
            "recall_by_doc_kind": by_kind_summary,
            "misses": miss_samples,
            "avg_latency": avg_latency,
        },
        "queries": query_results,
    }

    report_path = os.path.join(args.output_dir, f"{run_id}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote report to {report_path}")
    if args.write_csv:
        csv_path = os.path.join(args.output_dir, f"{run_id}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["query_id", "stage", "rank", "document_id", "score", "in_gold"],
            )
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"Wrote candidates to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
