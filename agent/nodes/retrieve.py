from __future__ import annotations

import logging
import time
from typing import Any

from agent.shared import QAState, return_with_profile, rrf_margin_ratio
from api.config import enabled_lanes, get_settings
from api.db import SessionLocal
from api.embed import EmbedClient
from api.query_expansion import build_bm25_queries, build_prf_query, decompose_question
from api.retrieval import (
    RetrievalFilters,
    bm25_search,
    group_top_chunks,
    rrf_fuse,
    title_bm25_search,
    title_vector_search,
    vector_search,
)

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def retrieve_candidates_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    question = state["question"]
    bm25_query = state.get("bm25_query") or question
    bm25_strict_query = state.get("bm25_strict_query")
    intent = state.get("intent") or {}
    filters = state["filters"]
    lanes = enabled_lanes(settings)
    bm25_limit = max(10, settings.ask_bm25_max_docs)
    min_docs = max(1, getattr(settings, "ask_min_docs", 3))
    try:
        client = EmbedClient()
        max_facets = max(1, getattr(settings, "ask_max_facets", 3))
        embeddings = client.embed_batch([question])
        if not embeddings or len(embeddings) != 1:
            embeddings = [client.embed(question)]
        query_embedding = embeddings[0]

        bm25_facet_questions = [question]
        facets = decompose_question(question, max_facets=max_facets)
        if facets:
            seen = {question.strip().lower()}
            for facet in facets:
                key = facet.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                bm25_facet_questions.append(facet)
                if len(bm25_facet_questions) >= max_facets:
                    break

        def _dedupe_docs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen: set[int] = set()
            deduped: list[dict[str, Any]] = []
            for item in items:
                doc_id = item.get("document_id")
                if doc_id is None or doc_id in seen:
                    continue
                seen.add(int(doc_id))
                deduped.append(item)
            return deduped

        def _merge_docs(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
            merged: list[dict[str, Any]] = []
            seen: set[int] = set()
            for item in primary + secondary:
                doc_id = item.get("document_id")
                if doc_id is None or doc_id in seen:
                    continue
                seen.add(int(doc_id))
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
                    if doc_id is None or doc_id in seen:
                        continue
                    seen.add(int(doc_id))
                    result.append(item)

            _add(primary[:keep_primary])
            added_secondary = 0
            for item in secondary:
                if len(result) >= limit or added_secondary >= budget:
                    break
                doc_id = item.get("document_id")
                if doc_id is None or doc_id in seen:
                    continue
                seen.add(int(doc_id))
                result.append(item)
                added_secondary += 1

            if len(result) < limit:
                _add(primary[keep_primary:])
            return result

        def _run_for_bm25(
            filters_to_use: RetrievalFilters,
            bm25_query_value: str,
            bm25_strict_value: str | None,
        ):
            with SessionLocal() as db:
                bm25_hits_raw: list[dict[str, Any]] = []
                bm25_hits: list[dict[str, Any]] = []
                bm25_strict_hits: list[dict[str, Any]] = []
                bm25_title_hits: list[dict[str, Any]] = []
                if "bm25" in lanes:
                    bm25_hits_raw = bm25_search(db, bm25_query_value, filters_to_use, limit=bm25_limit)
                    bm25_hits = _dedupe_docs(bm25_hits_raw)
                    if bm25_strict_value:
                        bm25_strict_hits = bm25_search(
                            db,
                            bm25_strict_value,
                            filters_to_use,
                            limit=bm25_limit,
                        )
                    bm25_title_hits = title_bm25_search(
                        db,
                        bm25_query_value,
                        filters_to_use,
                        limit=bm25_limit,
                    )
                    bm25_hits = _dedupe_docs(bm25_hits)
                    bm25_strict_hits = _dedupe_docs(bm25_strict_hits)
                    bm25_title_hits = _dedupe_docs(bm25_title_hits)
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
                    secondary_hits = _combine_sources(
                        secondary_sources,
                        max_docs=bm25_limit,
                        weights=secondary_weights,
                    ) if secondary_sources else []
                    bm25_hits = _merge_with_budget(primary_hits, secondary_hits, bm25_limit)
            chunk_candidates: list[dict[str, Any]] = []
            if "bm25" in lanes:
                chunk_candidates += bm25_hits_raw
                chunk_candidates += bm25_strict_hits
            return (
                bm25_hits,
                bm25_strict_hits,
                bm25_title_hits,
                chunk_candidates,
            )

        expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
        expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
        expand_docs = getattr(settings, "ask_max_docs_expand", 20)

        bm25_specs: list[tuple[str, str | None]] = []
        for idx, facet_question in enumerate(bm25_facet_questions):
            if idx == 0:
                facet_bm25 = bm25_query
                facet_bm25_strict = bm25_strict_query
            else:
                facet_bm25, facet_bm25_strict = build_bm25_queries(facet_question, intent)
            bm25_specs.append((facet_bm25, facet_bm25_strict))

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

        def _run_all_facets(filters_to_use: RetrievalFilters):
            with SessionLocal() as db:
                vector_hits_raw = (
                    vector_search(db, query_embedding, filters_to_use, limit=100)
                    if "vector" in lanes
                    else []
                )
                vector_hits = _dedupe_docs(vector_hits_raw)
                title_hits_raw: list[dict[str, Any]] = []
                title_lexical_hits: list[dict[str, Any]] = []
                title_hits: list[dict[str, Any]] = []
                if "title" in lanes:
                    title_hits_raw = title_vector_search(db, query_embedding, filters_to_use, limit=50)
                    title_lexical_hits = title_bm25_search(db, bm25_query, filters_to_use, limit=bm25_limit)
                    title_hits = _dedupe_docs(title_hits_raw)

            bm25_runs = [
                _run_for_bm25(filters_to_use, facet_bm25, facet_bm25_strict)
                for facet_bm25, facet_bm25_strict in bm25_specs
            ]
            prf_used = False
            if bm25_runs:
                prf_query = build_prf_query(
                    bm25_query,
                    bm25_runs[0][3],
                    max_docs=getattr(settings, "bm25_prf_docs", 5),
                    max_terms=getattr(settings, "bm25_prf_terms", 6),
                )
                if prf_query:
                    bm25_runs.append(_run_for_bm25(filters_to_use, prf_query, None))
                    prf_used = True
            bm25_primary = bm25_runs[0][0] if bm25_runs else []
            secondary_sources = [run[0] for run in bm25_runs[1:] if run[0]]
            secondary_hits = _combine_sources(secondary_sources, max_docs=bm25_limit) if secondary_sources else []
            bm25_hits = _merge_with_budget(bm25_primary, secondary_hits, bm25_limit)

            bm25_strict_hits = []
            bm25_title_hits = []
            chunk_candidates: list[dict[str, Any]] = []
            for run in bm25_runs:
                bm25_strict_hits += run[1]
                bm25_title_hits += run[2]
                chunk_candidates += run[3]

            if "vector" in lanes:
                chunk_candidates += vector_hits_raw

            counts = {
                "vector": len(vector_hits),
                "bm25": len(bm25_hits),
                "bm25_strict": len(bm25_strict_hits),
                "bm25_title": len(bm25_title_hits),
                "title": len(title_hits),
                "title_lexical": len(title_lexical_hits),
                "facets": len(bm25_specs),
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

        def _compute(filters_to_use: RetrievalFilters):
            (
                vector_hits,
                bm25_hits,
                bm25_strict_hits,
                bm25_title_hits,
                title_hits,
                title_lexical_hits,
                chunk_candidates,
                counts,
            ) = _run_all_facets(filters_to_use)
            sources: list[list[dict[str, Any]]] = []
            weights: list[float] = []
            if "vector" in lanes:
                sources.append(vector_hits)
                weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
            if "bm25" in lanes:
                sources.append(bm25_hits)
                weights.append(getattr(settings, "ask_rrf_weight_bm25", 1.0))
            if "title" in lanes:
                sources.append(title_hits)
                weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
                if title_lexical_hits and len(title_hits) < min_docs:
                    sources.append(_dedupe_docs(title_lexical_hits))
                    weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))

            fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
            soft_language = False
            if filters_to_use.language:
                margin = rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
                if len(fused) < min_docs or margin < expand_ratio:
                    relaxed_filters = RetrievalFilters(
                        language=None,
                        doc_kind=filters_to_use.doc_kind,
                        doc_subkind=filters_to_use.doc_subkind,
                        since_date=filters_to_use.since_date,
                        until_date=filters_to_use.until_date,
                    )
                    (
                        relaxed_vector,
                        relaxed_bm25,
                        relaxed_bm25_strict,
                        relaxed_bm25_title,
                        relaxed_title,
                        relaxed_title_lexical,
                        relaxed_chunks,
                        relaxed_counts,
                    ) = _run_all_facets(relaxed_filters)
                    if "vector" in lanes:
                        sources.append(relaxed_vector)
                        weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
                    if "bm25" in lanes:
                        sources.append(relaxed_bm25)
                        weights.append(getattr(settings, "ask_rrf_weight_bm25", 1.0))
                    if "title" in lanes:
                        sources.append(relaxed_title)
                        weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
                        if relaxed_title_lexical and len(relaxed_title) < min_docs:
                            sources.append(_dedupe_docs(relaxed_title_lexical))
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
            if fused and rrf_margin_ratio(fused, probe=expand_probe) < expand_ratio:
                fused = rrf_fuse(
                    sources,
                    max_docs=settings.ask_max_docs + expand_docs,
                    weights=weights,
                )
                rrf_expanded = True
            top_chunks = group_top_chunks(
                chunk_candidates,
                per_doc=getattr(settings, "ask_chunks_per_doc", 4),
            )
            return fused, top_chunks, chunk_candidates, counts, rrf_expanded, soft_language

        fused, top_chunks, chunk_candidates, counts, rrf_expanded, soft_language = _compute(filters)
        fallbacks: list[str] = []
        if soft_language:
            fallbacks.append("soft_language")
        preserve_temporal_window = bool(state.get("feed_query")) and bool(
            filters.since_date or filters.until_date
        )
        allow_margin_fallback = bool(getattr(settings, "ask_fallback_allow_margin", False))

        def _relax(new_filters: RetrievalFilters, reason: str, allow_margin: bool = False) -> None:
            nonlocal filters, fused, top_chunks, chunk_candidates, counts, rrf_expanded
            if len(fused) >= min_docs:
                if not allow_margin:
                    return
                margin = rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
                if margin >= expand_ratio:
                    return
            filters = new_filters
            fused, top_chunks, chunk_candidates, counts, rrf_expanded_inner, soft_language_inner = _compute(filters)
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
                allow_margin=allow_margin_fallback,
            )
        if filters.language:
            _relax(
                RetrievalFilters(
                    language=None,
                    doc_kind=filters.doc_kind,
                    doc_subkind=filters.doc_subkind,
                    since_date=filters.since_date,
                    until_date=filters.until_date,
                ),
                "drop_language",
                allow_margin=allow_margin_fallback,
            )
        if not preserve_temporal_window and (filters.since_date or filters.until_date):
            _relax(
                RetrievalFilters(
                    language=filters.language,
                    doc_kind=filters.doc_kind,
                    doc_subkind=filters.doc_subkind,
                    since_date=None,
                    until_date=None,
                ),
                "drop_dates",
                allow_margin=allow_margin_fallback,
            )
        if preserve_temporal_window:
            _relax(
                RetrievalFilters(
                    language=None,
                    doc_kind=None,
                    doc_subkind=None,
                    since_date=filters.since_date,
                    until_date=filters.until_date,
                ),
                "no_filters_keep_dates",
            )
        else:
            _relax(RetrievalFilters(), "no_filters")
        used_fallback = "+".join(fallbacks) if fallbacks else None

        elapsed = time.monotonic() - start
        logger.info(
            "retrieve.done req=%s candidates=%s chunks=%s sources=%s fallback=%s rrf_expand=%s elapsed=%.2fs",
            request_id,
            len(fused),
            len(chunk_candidates),
            counts,
            used_fallback,
            rrf_expanded,
            elapsed,
        )
        return return_with_profile(
            state,
            "retrieve",
            {
            "candidate_docs": fused,
            "top_chunks": top_chunks,
            "chunk_candidates": chunk_candidates,
            "query_embedding": query_embedding,
            "query_embeddings": embeddings,
            "filters": filters,
            },
            elapsed_seconds=round(elapsed, 3),
            candidate_docs=len(fused),
            chunk_candidates=len(chunk_candidates),
            fallback=used_fallback,
            rrf_expanded=bool(rrf_expanded),
            source_counts=counts,
        )
    except Exception:
        logger.exception("retrieve.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
