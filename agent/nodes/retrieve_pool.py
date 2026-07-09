"""Multi-lane hybrid pool computation for the retrieve node.

Extracted from ``retrieve_candidates_node``: a :class:`PoolQuery` carries the
per-request retrieval context, :func:`compute_pool` runs every lane and fuses
one candidate pool for a given filter set, and :func:`apply_relaxation_ladder`
re-computes the pool down the filter-relaxation ladder until it is populated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent.shared import rrf_margin_ratio
from api.config import get_settings
from api.db import SessionLocal
from api.query_expansion import build_prf_query
from api.retrieval import (
    RetrievalFilters,
    bm25_search,
    dedupe_docs,
    fuse_sources,
    group_top_chunks,
    merge_with_budget,
    rrf_fuse,
    title_bm25_search,
    title_vector_search,
    vector_search,
)

settings = get_settings()


@dataclass(frozen=True)
class PoolQuery:
    """Immutable per-request context shared by every pool computation pass."""

    lanes: set[str]
    bm25_query: str
    bm25_specs: list[tuple[str, str | None]]
    query_embedding: list[float]
    hyde_embedding: list[float] | None
    bm25_limit: int
    min_docs: int
    expand_ratio: float
    expand_probe: int
    expand_docs: int
    anchors_enabled: bool

    def with_hyde(self, hyde_embedding: list[float]) -> PoolQuery:
        return replace(self, hyde_embedding=hyde_embedding)


@dataclass
class PoolResult:
    """One computed candidate pool (fused docs + supporting chunk evidence)."""

    fused: list[dict[str, Any]]
    top_chunks: dict[int, list[dict[str, Any]]]
    chunk_candidates: list[dict[str, Any]]
    counts: dict[str, Any]
    rrf_expanded: bool
    soft_language: bool


def inject_semantic_anchors(
    fused: list[dict[str, Any]],
    sources: list[list[dict[str, Any]]],
    weights: list[float],
    anchor_lanes: list[list[dict[str, Any]]],
    anchor_top: int,
) -> tuple[list[dict[str, Any]], int]:
    """Guarantee a fused-pool slot for docs at the top of a raw-query semantic lane.

    A paraphrase/annex question gives the answer-bearing doc zero lexical-lane
    votes, while the correlated BM25 lanes (broad+strict+title, x facets, +PRF)
    each vote for the same generic near-matches — so a doc that is top-3 by
    title/chunk embedding similarity can still fall past the capped RRF cutoff.
    Any doc ranked within `anchor_top` of an anchor lane that fusion evicted is
    re-fused over the full source set and appended with its true rrf_score, so
    downstream rerank (which re-sorts by document similarity) gets to judge it.
    Returns (fused, appended_count); no-op when the anchors already made the pool.
    """
    if anchor_top <= 0 or not fused:
        return fused, 0
    anchor_ids: list[int] = []
    for lane in anchor_lanes:
        for row in lane[:anchor_top]:
            doc_id = row.get("document_id")
            if doc_id is not None and int(doc_id) not in anchor_ids:
                anchor_ids.append(int(doc_id))
    if not anchor_ids:
        return fused, 0
    present = {int(item["document_id"]) for item in fused}
    missing = [doc_id for doc_id in anchor_ids if doc_id not in present]
    if not missing:
        return fused, 0
    full = rrf_fuse(
        sources,
        max_docs=sum(len(source) for source in sources),
        weights=weights,
    )
    by_id = {int(row["document_id"]): row for row in full}
    appended = 0
    for doc_id in missing:
        row = by_id.get(doc_id)
        if row is not None:
            fused.append(row)
            appended += 1
    return fused, appended


def run_bm25_lane(
    query: PoolQuery,
    filters: RetrievalFilters,
    bm25_query_value: str,
    bm25_strict_value: str | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run the chunk/strict/title BM25 searches for one query string.

    Returns ``(bm25_hits, bm25_strict_hits, bm25_title_hits, chunk_candidates)``
    where ``bm25_hits`` is the budget-merged doc lane and ``chunk_candidates``
    keeps the raw chunk rows for downstream per-doc chunk grouping.
    """
    with SessionLocal() as db:
        bm25_hits_raw: list[dict[str, Any]] = []
        bm25_hits: list[dict[str, Any]] = []
        bm25_strict_hits: list[dict[str, Any]] = []
        bm25_title_hits: list[dict[str, Any]] = []
        if "bm25" in query.lanes:
            bm25_hits_raw = bm25_search(db, bm25_query_value, filters, limit=query.bm25_limit)
            bm25_hits = dedupe_docs(bm25_hits_raw)
            if bm25_strict_value:
                bm25_strict_hits = bm25_search(
                    db,
                    bm25_strict_value,
                    filters,
                    limit=query.bm25_limit,
                )
            bm25_title_hits = title_bm25_search(
                db,
                bm25_query_value,
                filters,
                limit=query.bm25_limit,
            )
            bm25_hits = dedupe_docs(bm25_hits)
            bm25_strict_hits = dedupe_docs(bm25_strict_hits)
            bm25_title_hits = dedupe_docs(bm25_title_hits)
            strict_min = getattr(settings, "bm25_strict_primary_min", 10)
            if bm25_strict_hits and len(bm25_strict_hits) >= strict_min:
                primary_hits = bm25_strict_hits
                secondary_sources = []
                secondary_weights = []
                if bm25_hits:
                    secondary_sources.append(bm25_hits)
                    secondary_weights.append(getattr(settings, "bm25_fuse_weight_chunk", 1.0))
                if bm25_title_hits:
                    secondary_sources.append(bm25_title_hits)
                    secondary_weights.append(getattr(settings, "bm25_fuse_weight_title", 0.9))
            else:
                primary_hits = bm25_hits
                secondary_sources = []
                secondary_weights = []
                if bm25_strict_hits:
                    secondary_sources.append(bm25_strict_hits)
                    secondary_weights.append(getattr(settings, "bm25_fuse_weight_strict", 1.2))
                if bm25_title_hits:
                    secondary_sources.append(bm25_title_hits)
                    secondary_weights.append(getattr(settings, "bm25_fuse_weight_title", 0.9))
            secondary_hits = (
                fuse_sources(
                    secondary_sources,
                    max_docs=query.bm25_limit,
                    weights=secondary_weights,
                )
                if secondary_sources
                else []
            )
            bm25_hits = merge_with_budget(primary_hits, secondary_hits, query.bm25_limit)
    chunk_candidates: list[dict[str, Any]] = []
    if "bm25" in query.lanes:
        chunk_candidates += bm25_hits_raw
        chunk_candidates += bm25_strict_hits
    return bm25_hits, bm25_strict_hits, bm25_title_hits, chunk_candidates


def run_all_facets(
    query: PoolQuery, filters: RetrievalFilters
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Run every lane (vector, title, per-facet BM25 + PRF) for one filter set."""
    with SessionLocal() as db:
        vector_hits_raw = (
            vector_search(db, query.query_embedding, filters, limit=100)
            if "vector" in query.lanes
            else []
        )
        vector_hits = dedupe_docs(vector_hits_raw)
        title_hits_raw: list[dict[str, Any]] = []
        title_lexical_hits: list[dict[str, Any]] = []
        title_hits: list[dict[str, Any]] = []
        if "title" in query.lanes:
            title_hits_raw = title_vector_search(db, query.query_embedding, filters, limit=50)
            title_lexical_hits = title_bm25_search(
                db, query.bm25_query, filters, limit=query.bm25_limit
            )
            title_hits = dedupe_docs(title_hits_raw)

    bm25_runs = [
        run_bm25_lane(query, filters, facet_bm25, facet_bm25_strict)
        for facet_bm25, facet_bm25_strict in query.bm25_specs
    ]
    prf_used = False
    if bm25_runs:
        prf_query = build_prf_query(
            query.bm25_query,
            bm25_runs[0][3],
            max_docs=getattr(settings, "bm25_prf_docs", 5),
            max_terms=getattr(settings, "bm25_prf_terms", 6),
        )
        if prf_query:
            bm25_runs.append(run_bm25_lane(query, filters, prf_query, None))
            prf_used = True
    bm25_primary = bm25_runs[0][0] if bm25_runs else []
    secondary_sources = [run[0] for run in bm25_runs[1:] if run[0]]
    secondary_hits = (
        fuse_sources(secondary_sources, max_docs=query.bm25_limit) if secondary_sources else []
    )
    bm25_hits = merge_with_budget(bm25_primary, secondary_hits, query.bm25_limit)

    bm25_strict_hits = []
    bm25_title_hits = []
    chunk_candidates: list[dict[str, Any]] = []
    for run in bm25_runs:
        bm25_strict_hits += run[1]
        bm25_title_hits += run[2]
        chunk_candidates += run[3]

    if "vector" in query.lanes:
        chunk_candidates += vector_hits_raw

    counts = {
        "vector": len(vector_hits),
        "bm25": len(bm25_hits),
        "bm25_strict": len(bm25_strict_hits),
        "bm25_title": len(bm25_title_hits),
        "title": len(title_hits),
        "title_lexical": len(title_lexical_hits),
        "facets": len(query.bm25_specs),
        "bm25_prf": 1 if prf_used else 0,
    }
    return (
        vector_hits,
        bm25_hits,
        bm25_strict_hits,
        bm25_title_hits,
        title_hits,
        title_lexical_hits,
        chunk_candidates,
        counts,
    )


def compute_pool(query: PoolQuery, filters: RetrievalFilters) -> PoolResult:
    """Compute one fused candidate pool: lanes -> weighted RRF -> guarantees.

    Applies, in order: in-pass soft language relaxation when the language-filtered
    pool is thin or flat, RRF pool expansion on a flat margin, and the semantic-
    anchor slot guarantee (RC4).
    """
    (
        vector_hits,
        bm25_hits,
        bm25_strict_hits,
        bm25_title_hits,
        title_hits,
        title_lexical_hits,
        chunk_candidates,
        counts,
    ) = run_all_facets(query, filters)
    sources: list[list[dict[str, Any]]] = []
    weights: list[float] = []
    # Raw-query semantic lanes whose top ranks are pool-guaranteed (RC4).
    # HyDE is excluded: its lane is synthetic-query-driven and separately
    # gated, and anchoring it would re-open the recall-dilution failure.
    anchor_lanes: list[list[dict[str, Any]]] = []
    if "vector" in query.lanes:
        sources.append(vector_hits)
        weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
        anchor_lanes.append(vector_hits)
        if query.hyde_embedding is not None:
            with SessionLocal() as db:
                hyde_hits = dedupe_docs(vector_search(db, query.hyde_embedding, filters, limit=100))
            sources.append(hyde_hits)
            weights.append(getattr(settings, "ask_rrf_weight_hyde", 1.0))
    if "bm25" in query.lanes:
        sources.append(bm25_hits)
        weights.append(getattr(settings, "ask_rrf_weight_bm25", 1.0))
    if "title" in query.lanes:
        sources.append(title_hits)
        weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
        anchor_lanes.append(title_hits)
        if title_lexical_hits and len(title_hits) < query.min_docs:
            sources.append(dedupe_docs(title_lexical_hits))
            weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))

    fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
    soft_language = False
    if filters.language:
        margin = rrf_margin_ratio(fused, probe=query.expand_probe) if fused else 0.0
        if len(fused) < query.min_docs or margin < query.expand_ratio:
            relaxed_filters = replace(filters, language=None)
            (
                relaxed_vector,
                relaxed_bm25,
                relaxed_bm25_strict,
                relaxed_bm25_title,
                relaxed_title,
                relaxed_title_lexical,
                relaxed_chunks,
                relaxed_counts,
            ) = run_all_facets(query, relaxed_filters)
            if "vector" in query.lanes:
                sources.append(relaxed_vector)
                weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
                anchor_lanes.append(relaxed_vector)
            if "bm25" in query.lanes:
                sources.append(relaxed_bm25)
                weights.append(getattr(settings, "ask_rrf_weight_bm25", 1.0))
            if "title" in query.lanes:
                sources.append(relaxed_title)
                weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
                anchor_lanes.append(relaxed_title)
                if relaxed_title_lexical and len(relaxed_title) < query.min_docs:
                    sources.append(dedupe_docs(relaxed_title_lexical))
                    weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))
            chunk_candidates += relaxed_chunks
            counts.update(
                {
                    "vector_lang_relaxed": len(relaxed_vector),
                    "bm25_lang_relaxed": len(relaxed_bm25),
                    "bm25_strict_lang_relaxed": len(relaxed_bm25_strict),
                    "bm25_title_lang_relaxed": len(relaxed_bm25_title),
                    "title_lang_relaxed": len(relaxed_title),
                    "title_lexical_lang_relaxed": len(relaxed_title_lexical),
                    "facets_lang_relaxed": relaxed_counts.get("facets"),
                }
            )
            fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
            soft_language = True

    rrf_expanded = False
    if fused and rrf_margin_ratio(fused, probe=query.expand_probe) < query.expand_ratio:
        fused = rrf_fuse(
            sources,
            max_docs=settings.ask_max_docs + query.expand_docs,
            weights=weights,
        )
        rrf_expanded = True
    if query.anchors_enabled:
        fused, anchors_added = inject_semantic_anchors(
            fused,
            sources,
            weights,
            anchor_lanes,
            anchor_top=getattr(settings, "ask_semantic_anchor_top", 3),
        )
        if anchors_added:
            counts["semantic_anchors"] = anchors_added
    top_chunks = group_top_chunks(
        chunk_candidates,
        per_doc=getattr(settings, "ask_chunks_per_doc", 4),
    )
    return PoolResult(fused, top_chunks, chunk_candidates, counts, rrf_expanded, soft_language)


def apply_relaxation_ladder(
    query: PoolQuery,
    pool: PoolResult,
    filters: RetrievalFilters,
    preserve_temporal_window: bool,
    allow_margin_fallback: bool,
) -> tuple[PoolResult, RetrievalFilters, list[str]]:
    """Progressively drop filters until the pool has ``min_docs`` candidates.

    Ladder: doc_kind -> language -> dates -> everything (keeping an explicit
    temporal window when the question asked for one). Each step recomputes the
    whole pool; ``rrf_expanded`` accumulates across steps and every applied
    relaxation is recorded in the returned fallback reasons.
    """
    fallbacks: list[str] = []
    if pool.soft_language:
        fallbacks.append("soft_language")

    def _relax(new_filters: RetrievalFilters, reason: str, allow_margin: bool = False) -> None:
        nonlocal pool, filters
        if len(pool.fused) >= query.min_docs:
            if not allow_margin:
                return
            margin = rrf_margin_ratio(pool.fused, probe=query.expand_probe) if pool.fused else 0.0
            if margin >= query.expand_ratio:
                return
        filters = new_filters
        new_pool = compute_pool(query, filters)
        if new_pool.soft_language:
            fallbacks.append("soft_language")
        fallbacks.append(reason)
        new_pool.rrf_expanded = new_pool.rrf_expanded or pool.rrf_expanded
        pool = new_pool

    if filters.doc_kind:
        _relax(
            replace(filters, doc_kind=None, doc_subkind=None),
            "drop_doc_kind",
            allow_margin=allow_margin_fallback,
        )
    if filters.language:
        _relax(
            replace(filters, language=None),
            "drop_language",
            allow_margin=allow_margin_fallback,
        )
    if not preserve_temporal_window and (filters.since_date or filters.until_date):
        _relax(
            replace(filters, since_date=None, until_date=None),
            "drop_dates",
            allow_margin=allow_margin_fallback,
        )
    if preserve_temporal_window:
        _relax(
            RetrievalFilters(since_date=filters.since_date, until_date=filters.until_date),
            "no_filters_keep_dates",
        )
    else:
        _relax(RetrievalFilters(), "no_filters")
    return pool, filters, fallbacks
