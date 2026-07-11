from __future__ import annotations

import logging
import time

from agent.nodes.retrieve_pool import (
    PoolQuery,
    apply_relaxation_ladder,
    compute_pool,
)
from agent.shared import QAState, return_with_profile, rrf_margin_ratio
from api.config import enabled_lanes, get_settings
from api.embed import EmbedClient
from api.enumeration import parse_enumeration
from api.query_classifiers import is_reference_query
from agent.nodes.second_hop import apply_second_hop
from api.query_expansion import (
    build_bm25_queries,
    build_hyde_document,
    decompose_question,
)

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _build_facet_specs(
    question: str,
    bm25_query: str,
    bm25_strict_query: str | None,
    intent: dict,
    max_facets: int,
) -> list[tuple[str, str | None]]:
    """BM25 (query, strict_query) specs: the main question plus decomposed facets."""
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
    specs: list[tuple[str, str | None]] = []
    for idx, facet_question in enumerate(facet_questions):
        if idx == 0:
            specs.append((bm25_query, bm25_strict_query))
        else:
            specs.append(build_bm25_queries(facet_question, intent))
    return specs


def _hyde_statically_enabled(question: str) -> bool:
    """HyDE exclusions decidable before any retrieval pass.

    HyDE bridges the vague/colloquial/Valencian query -> formal-document gap that
    leaves the gold doc deep in the raw-query ranking. Static exclusions: the
    master switch, and reference-queries — the hypothetical drifts off the cited
    norm and the heavy HyDE RRF lane evicts the correctly-retrieved gold (v2-092).
    The build itself (an LLM hop) is further confidence-gated on the baseline pool.
    """
    if not getattr(settings, "ask_hyde_enabled", False):
        return False
    if getattr(settings, "ask_hyde_conditional", True) and is_reference_query(question):
        return False
    return True


def _build_hyde_embedding(client: EmbedClient, question: str) -> list[float] | None:
    """Generate + embed the hypothetical document; None on any failure."""
    try:
        hyde_text = build_hyde_document(question)
        if not hyde_text:
            return None
        hyde_embs = client.embed_batch([hyde_text])
        return hyde_embs[0] if hyde_embs else None
    except Exception:
        return None


def retrieve_candidates_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    question = state["question"]
    filters = state["filters"]
    try:
        client = EmbedClient()
        embeddings = client.embed_batch([question])
        if not embeddings or len(embeddings) != 1:
            embeddings = [client.embed(question)]

        bm25_query = state.get("bm25_query") or question
        query = PoolQuery(
            lanes=enabled_lanes(settings),
            bm25_query=bm25_query,
            bm25_specs=_build_facet_specs(
                question,
                bm25_query,
                state.get("bm25_strict_query"),
                state.get("intent") or {},
                max_facets=max(1, getattr(settings, "ask_max_facets", 3)),
            ),
            query_embedding=embeddings[0],
            hyde_embedding=None,
            bm25_limit=max(10, settings.ask_bm25_max_docs),
            min_docs=max(1, getattr(settings, "ask_min_docs", 3)),
            expand_ratio=getattr(settings, "ask_rrf_expand_margin_ratio", 0.12),
            expand_probe=getattr(settings, "ask_rrf_margin_probe", 5),
            expand_docs=getattr(settings, "ask_max_docs_expand", 20),
            # Semantic anchors are skipped for enumeration queries: those want the
            # exhaustive month+category series (rerank widens its caps for them), and
            # anchoring a semantically-top doc displaces series members from the read
            # set (measured on tester #30: anchors=2 dropped 3 refs from the listing).
            anchors_enabled=(
                getattr(settings, "ask_semantic_anchor_enabled", False)
                and parse_enumeration(question) is None
            ),
        )

        pool = compute_pool(query, filters)

        # Confidence-gate HyDE: the baseline pool (no HyDE) is now computed. Only pay
        # the HyDE generation+embedding hop when that baseline ranking is low-confidence
        # — a shallow RRF margin means no document clearly dominates, the anchor-poor
        # vague/colloquial case HyDE is for. A confident baseline already has the gold
        # near the top; firing HyDE there only adds latency and risks the reader citing
        # a drifted sibling. Calibrated on eval_v2: margin<0.22 fires on every query
        # HyDE is known to recover (v2-020/032/034/035/099) and skips the confident-
        # baseline regressions (v2-023/078). When it fires, recompute with the HyDE lane.
        hyde_enabled = _hyde_statically_enabled(question)
        hyde_gate_margin = None
        if hyde_enabled and getattr(settings, "ask_hyde_confidence_gated", True):
            hyde_gate_margin = (
                rrf_margin_ratio(pool.fused, probe=query.expand_probe) if pool.fused else 0.0
            )
            if hyde_gate_margin >= getattr(settings, "ask_hyde_margin_threshold", 0.22):
                hyde_enabled = False
        if hyde_enabled:
            hyde_embedding = _build_hyde_embedding(client, question)
            if hyde_embedding is not None:
                query = query.with_hyde(hyde_embedding)
                pool = compute_pool(query, filters)

        preserve_temporal_window = bool(state.get("feed_query")) and bool(
            filters.since_date or filters.until_date
        )
        pool, filters, fallbacks = apply_relaxation_ladder(
            query,
            pool,
            filters,
            preserve_temporal_window,
            allow_margin_fallback=bool(getattr(settings, "ask_fallback_allow_margin", False)),
        )
        used_fallback = "+".join(fallbacks) if fallbacks else None

        # Second hop: gated, only pays for an extra retrieval pass when the
        # question is multi-entity AND an entity's document is missing from the
        # pool the ladder above already settled on. See agent/nodes/second_hop.py.
        pool, hop_doc_ids, hop_profile = apply_second_hop(
            question,
            pool,
            filters,
            state.get("intent") or {},
            client,
            request_id,
        )
        if hop_doc_ids:
            # Pin the added docs into the read set: rerank's caps are non-
            # deterministic and would otherwise be free to drop a hop doc that
            # only entered the pool at the very end of the fused list. Merge with
            # any pin already set upstream instead of overwriting it.
            existing_pins = [int(d) for d in (state.get("norm_pin_doc_ids") or []) if d is not None]
            for doc_id in hop_doc_ids:
                if doc_id not in existing_pins:
                    existing_pins.append(doc_id)
            norm_pin_doc_ids = existing_pins
        else:
            norm_pin_doc_ids = state.get("norm_pin_doc_ids") or []

        hyde_fired = query.hyde_embedding is not None
        elapsed = time.monotonic() - start
        logger.info(
            "retrieve.done req=%s candidates=%s chunks=%s sources=%s fallback=%s rrf_expand=%s hyde=%s gate_margin=%s second_hop=%s elapsed=%.2fs",
            request_id,
            len(pool.fused),
            len(pool.chunk_candidates),
            pool.counts,
            used_fallback,
            pool.rrf_expanded,
            hyde_fired,
            None if hyde_gate_margin is None else round(hyde_gate_margin, 3),
            bool(hop_profile),
            elapsed,
        )
        return return_with_profile(
            state,
            "retrieve",
            {
                "candidate_docs": pool.fused,
                "top_chunks": pool.top_chunks,
                "chunk_candidates": pool.chunk_candidates,
                "query_embedding": query.query_embedding,
                "query_embeddings": embeddings,
                "hyde_embedding": query.hyde_embedding,
                "filters": filters,
                "norm_pin_doc_ids": norm_pin_doc_ids,
            },
            elapsed_seconds=round(elapsed, 3),
            candidate_docs=len(pool.fused),
            chunk_candidates=len(pool.chunk_candidates),
            fallback=used_fallback,
            rrf_expanded=bool(pool.rrf_expanded),
            source_counts=pool.counts,
            hyde_fired=hyde_fired,
            hyde_gate_margin=(None if hyde_gate_margin is None else round(hyde_gate_margin, 3)),
            second_hop=hop_profile,
        )
    except Exception:
        logger.exception(
            "retrieve.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start
        )
        raise
