from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agent.nodes import (
    analyze_intent_node,
    answer_node,
    backfill_node,
    contextualize_query_node,
    online_ingest_node,
    read_docs_node,
    rerank_titles_node,
    retrieve_candidates_node,
    temporal_guard_node,
)
from agent.shared import QAState
from api.config import get_settings
from api.query_classifiers import is_reference_query

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _should_continue_after_temporal(state: QAState) -> str:
    if state.get("temporal_reject"):
        return "reject"
    return "continue"


def _should_backfill(state: QAState) -> str:
    if not settings.backfill_enabled:
        return "rerank_titles"
    if state.get("backfill_attempted"):
        return "rerank_titles"
    # No candidates at all -> try the targeted on-demand fetch.
    if not state.get("candidate_docs"):
        return "backfill"
    # Reference query (names a specific disposition): retrieval may have returned
    # only tangential same-number/topic docs while the cited norm itself is missing
    # from the window. Route to backfill; the node cheaply skips if it's in corpus.
    if is_reference_query(state.get("question") or ""):
        return "backfill"
    return "rerank_titles"


def build_graph():
    graph = StateGraph(QAState)
    graph.add_node("contextualize_query", contextualize_query_node)
    graph.add_node("analyze_intent", analyze_intent_node)
    graph.add_node("temporal_guard", temporal_guard_node)
    graph.add_node("online_ingest", online_ingest_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("backfill", backfill_node)
    graph.add_node("rerank_titles", rerank_titles_node)
    graph.add_node("read_docs", read_docs_node)
    graph.add_node("answer_node", answer_node)

    graph.set_entry_point("contextualize_query")
    graph.add_edge("contextualize_query", "analyze_intent")
    graph.add_edge("analyze_intent", "temporal_guard")
    graph.add_conditional_edges(
        "temporal_guard",
        _should_continue_after_temporal,
        {"reject": END, "continue": "online_ingest"},
    )
    graph.add_edge("online_ingest", "retrieve_candidates")
    graph.add_conditional_edges(
        "retrieve_candidates",
        _should_backfill,
        {"backfill": "backfill", "rerank_titles": "rerank_titles"},
    )
    graph.add_edge("backfill", "retrieve_candidates")
    graph.add_edge("rerank_titles", "read_docs")
    graph.add_edge("read_docs", "answer_node")
    graph.add_edge("answer_node", END)

    return graph.compile()
