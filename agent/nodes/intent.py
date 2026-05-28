from __future__ import annotations

from datetime import timedelta
import logging
import time

from agent.shared import QAState, return_with_profile
from api.config import get_settings
from api.intent import analyze_intent, analyze_intent_and_expand
from api.query_expansion import build_bm25_queries, guess_language, is_relative_time_query
from api.retrieval import RetrievalFilters
from api.temporal import local_today, resolve_relative_date_range

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _map_language(intent_lang: str | None) -> str:
    if intent_lang == "ca":
        return "va_va"
    return "es_es"


def analyze_intent_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    debug = bool(state.get("debug"))
    try:
        if settings.ask_llm_expand:
            intent, expansion = analyze_intent_and_expand(state["question"])
        else:
            intent = analyze_intent(state["question"])
            expansion = {}
        intent_lang = intent.get("language")
        if intent_lang in ("es", "ca"):
            lang = _map_language(intent_lang)
        else:
            lang = guess_language(state["question"])
        lang_filter = lang
        doc_kind = intent.get("doc_kind")
        doc_subkind = intent.get("doc_subkind")
        bm25_query, bm25_strict_query = build_bm25_queries(
            state["question"],
            intent,
            expansion=expansion,
        )
        since_date = intent.get("since_date")
        until_date = intent.get("until_date")
        relative_range = resolve_relative_date_range(
            state["question"],
            timezone_name=settings.temporal_timezone,
            week_start=settings.temporal_week_start,
        )
        feed_query = bool(relative_range) or is_relative_time_query(state["question"])
        if feed_query and (settings.ask_temporal_policy or "").lower() == "filter":
            if relative_range:
                since_date, until_date = relative_range
            else:
                today = local_today(settings.temporal_timezone)
                window_start = today - timedelta(days=settings.feed_recent_days)
                if since_date is None or since_date > window_start:
                    since_date = window_start
                if until_date is None:
                    until_date = today
        filters = RetrievalFilters(
            language=lang_filter,
            doc_kind=None,
            doc_subkind=None,
            since_date=since_date,
            until_date=until_date,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "intent.done req=%s lang=%s doc_kind=%s doc_subkind=%s elapsed=%.2fs",
            request_id,
            lang,
            doc_kind,
            doc_subkind,
            elapsed,
        )
        return return_with_profile(
            state,
            "intent",
            {
            "intent": intent,
            "language": lang,
            "filters": filters,
            "bm25_query": bm25_query,
            "bm25_strict_query": bm25_strict_query,
            "feed_query": feed_query,
            },
            elapsed_seconds=round(elapsed, 3),
            fallback=False,
            language=lang,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        if debug:
            logger.warning(
                "intent.fallback req=%s reason=%s question=%r elapsed=%.2fs",
                request_id,
                exc,
                state.get("question"),
                elapsed,
            )
        else:
            logger.warning(
                "intent.fallback req=%s reason=%s elapsed=%.2fs",
                request_id,
                exc,
                elapsed,
            )
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
        lang = guess_language(state["question"])
        bm25_query, bm25_strict_query = build_bm25_queries(
            state["question"],
            intent,
            expansion={},
        )
        since_date = None
        until_date = None
        relative_range = resolve_relative_date_range(
            state["question"],
            timezone_name=settings.temporal_timezone,
            week_start=settings.temporal_week_start,
        )
        feed_query = bool(relative_range) or is_relative_time_query(state["question"])
        if feed_query and (settings.ask_temporal_policy or "").lower() == "filter":
            if relative_range:
                since_date, until_date = relative_range
            else:
                today = local_today(settings.temporal_timezone)
                window_start = today - timedelta(days=settings.feed_recent_days)
                since_date = window_start
                until_date = today
        filters = RetrievalFilters(
            language=lang,
            doc_kind=None,
            doc_subkind=None,
            since_date=since_date,
            until_date=until_date,
        )
        return return_with_profile(
            state,
            "intent",
            {
            "intent": intent,
            "language": lang,
            "filters": filters,
            "bm25_query": bm25_query,
            "bm25_strict_query": bm25_strict_query,
            "feed_query": feed_query,
            },
            elapsed_seconds=round(elapsed, 3),
            fallback=True,
            language=lang,
            error=type(exc).__name__,
        )
