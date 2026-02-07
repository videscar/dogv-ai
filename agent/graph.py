from __future__ import annotations

from datetime import date, timedelta
import logging
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import text as sa_text

from api.answer import build_answer, no_evidence_answer
from api.auto_ingest import ensure_month_ingested, ensure_range_ingested, ensure_recent_ingested
from api.config import enabled_lanes, get_settings
from api.dogv_urls import build_html_url, build_pdf_url
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
from api.reader import extract_evidence
from api.rerank import rerank_titles
from api.retrieval import (
    RetrievalFilters,
    bm25_search,
    group_top_chunks,
    rrf_fuse,
    top_chunks_for_docs,
    title_bm25_search,
    title_vector_search,
    vector_search,
)
from api.db import SessionLocal
from api.models import DogvDocument, DogvIssue

settings = get_settings()
DEFAULT_LANGS = ["es_es", "va_va"]
logger = logging.getLogger("dogv.graph")


class QAState(TypedDict, total=False):
    question: str
    debug: bool
    intent: dict[str, Any]
    language: str
    request_id: str
    temporal_reject: bool
    query_embedding: list[float]
    query_embeddings: list[list[float]]
    bm25_query: str
    bm25_strict_query: str | None
    filters: RetrievalFilters
    candidate_docs: list[dict[str, Any]]
    chunk_candidates: list[dict[str, Any]]
    top_chunks: dict[int, list[dict[str, Any]]]
    selected_doc_ids: list[int]
    evidence: list[dict[str, Any]]
    full_docs: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    backfill_attempted: bool
    feed_query: bool
    online_ingest_done: bool


def _map_language(intent_lang: str | None) -> str:
    if intent_lang == "ca":
        return "va_va"
    return "es_es"


def _rrf_margin_ratio(candidates: list[dict[str, Any]], probe: int = 5) -> float:
    if not candidates:
        return 0.0
    top_score = float(candidates[0].get("rrf_score") or 0.0)
    if top_score <= 0:
        return 0.0
    idx = min(len(candidates) - 1, max(1, probe - 1))
    pivot = float(candidates[idx].get("rrf_score") or 0.0)
    return (top_score - pivot) / top_score


_RERANK_STOPWORDS = {
    "que",
    "para",
    "por",
    "como",
    "con",
    "una",
    "unos",
    "unas",
    "pero",
    "sobre",
    "este",
    "esta",
    "estos",
    "estas",
    "del",
    "las",
    "los",
    "un",
    "al",
    "el",
    "la",
    "es",
    "en",
    "de",
    "y",
    "o",
    "a",
    "i",
    "qui",
    "quin",
    "quina",
    "com",
    "uns",
    "unes",
    "els",
    "les",
    "dels",
    "per",
    "amb",
    "mes",
    "subvencio",
    "subvencions",
    "subvencion",
    "subvenciones",
    "ajuda",
    "ajudes",
    "ayuda",
    "ayudas",
    "convocatoria",
    "convocatòria",
    "convocatories",
    "bases",
    "resolucio",
    "resolució",
    "resolucion",
    "decret",
    "decreto",
    "extracte",
    "extracto",
}


def _extract_keywords_simple(text: str) -> list[str]:
    tokens = re.findall(r"[\w·'-]+", (text or "").lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 4 or token in _RERANK_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _coverage_score(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    lower = text.lower()
    return sum(lower.count(k) for k in keywords)


def _needs_eligibility(question: str) -> bool:
    return bool(
        re.search(
            r"\b(qui|qu[ií]en|beneficiar|beneficiari|beneficiario|sol·licit|solicitar|requisit|requisito|destinatari|destinatario|pot|puede|poden|pueden)\b",
            question,
            re.IGNORECASE,
        )
    )


def _needs_amounts(question: str) -> bool:
    if not question:
        return False
    return bool(re.search(r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", question, re.IGNORECASE))


def _is_base_like(item: dict[str, Any]) -> bool:
    doc_kind = (item.get("doc_kind") or "").lower()
    doc_subkind = (item.get("doc_subkind") or "").lower()
    title = (item.get("title") or "").lower()
    if doc_subkind in {"bases", "convocatoria", "convocatòria"}:
        return True
    if "bases reguladoras" in title or "bases reguladores" in title or "convocatoria" in title:
        return True
    if doc_kind in {"subvenciones", "ayudas"} and ("decret" in title or "decreto" in title):
        return True
    return False


def _best_snippet(question: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    keywords = _extract_keywords_simple(question)
    if not keywords:
        return (chunks[0].get("text") or "").strip()
    best_text = ""
    best_score = -1
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        score = _coverage_score(text, keywords)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text or (chunks[0].get("text") or "").strip()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

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


def analyze_intent_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    debug = bool(state.get("debug"))
    try:
        intent = analyze_intent(state["question"])
        intent_lang = intent.get("language")
        if intent_lang in ("es", "ca"):
            lang = _map_language(intent_lang)
        else:
            lang = guess_language(state["question"])
        lang_filter = lang
        doc_kind = intent.get("doc_kind")
        doc_subkind = intent.get("doc_subkind")
        expansion = llm_expand_query(state["question"], intent) if settings.ask_llm_expand else {}
        bm25_query, bm25_strict_query = build_bm25_queries(
            state["question"],
            intent,
            expansion=expansion,
        )
        since_date = intent.get("since_date")
        until_date = intent.get("until_date")
        feed_query = is_relative_time_query(state["question"])
        if feed_query and (settings.ask_temporal_policy or "").lower() == "filter":
            today = date.today()
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
        logger.info(
            "intent.done req=%s lang=%s doc_kind=%s doc_subkind=%s elapsed=%.2fs",
            request_id,
            lang,
            doc_kind,
            doc_subkind,
            time.monotonic() - start,
        )
        return {
            "intent": intent,
            "language": lang,
            "filters": filters,
            "bm25_query": bm25_query,
            "bm25_strict_query": bm25_strict_query,
            "feed_query": feed_query,
        }
    except Exception as exc:
        if debug:
            logger.warning(
                "intent.fallback req=%s reason=%s question=%r elapsed=%.2fs",
                request_id,
                exc,
                state.get("question"),
                time.monotonic() - start,
            )
        else:
            logger.warning(
                "intent.fallback req=%s reason=%s elapsed=%.2fs",
                request_id,
                exc,
                time.monotonic() - start,
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
        expansion = llm_expand_query(state["question"], intent) if settings.ask_llm_expand else {}
        bm25_query, bm25_strict_query = build_bm25_queries(
            state["question"],
            intent,
            expansion=expansion,
        )
        since_date = None
        until_date = None
        feed_query = is_relative_time_query(state["question"])
        if feed_query and (settings.ask_temporal_policy or "").lower() == "filter":
            today = date.today()
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
        return {
            "intent": intent,
            "language": lang,
            "filters": filters,
            "bm25_query": bm25_query,
            "bm25_strict_query": bm25_strict_query,
            "feed_query": feed_query,
        }


def _temporal_reject_message(language: str | None) -> str:
    if (language or "").startswith(("va", "ca")):
        return (
            "La consulta fa referencia a una data relativa (hui, esta setmana, etc.). "
            "Indica un rang de dates concret (YYYY-MM-DD a YYYY-MM-DD) per a poder buscar al DOGV."
        )
    return (
        "La consulta hace referencia a una fecha relativa (hoy, esta semana, etc.). "
        "Indica un rango de fechas concreto (YYYY-MM-DD a YYYY-MM-DD) para poder buscar en el DOGV."
    )


def temporal_guard_node(state: QAState) -> QAState:
    policy = (settings.ask_temporal_policy or "reject").lower()
    question = state.get("question") or ""
    if policy == "reject" and is_relative_time_query(question):
        language = state.get("language") or guess_language(question)
        return {
            "answer": _temporal_reject_message(language),
            "citations": [],
            "temporal_reject": True,
        }
    return {"temporal_reject": False}


def _should_continue_after_temporal(state: QAState) -> str:
    if state.get("temporal_reject"):
        return "reject"
    return "continue"


def online_ingest_node(state: QAState) -> QAState:
    started_at = time.monotonic()
    request_id = state.get("request_id")
    try:
        if not settings.auto_ingest_enabled:
            logger.info("ingest.skip req=%s reason=disabled elapsed=%.2fs", request_id, time.monotonic() - started_at)
            return {"online_ingest_done": True}

        intent = state.get("intent") or {}
        since_date = intent.get("since_date")
        until_date = intent.get("until_date")

        if since_date or until_date:
            ingest_start = since_date or until_date
            ingest_end = until_date or since_date
            if ingest_start and ingest_end:
                ensure_range_ingested(ingest_start, ingest_end, DEFAULT_LANGS)
        else:
            today = date.today()
            with SessionLocal() as db:
                row = db.execute(sa_text("SELECT MAX(date) AS max_date FROM dogv_issues")).mappings().one()
                max_date = row["max_date"]
            if not max_date or max_date < today:
                ensure_recent_ingested(settings.auto_ingest_max_days, DEFAULT_LANGS)
        logger.info("ingest.done req=%s elapsed=%.2fs", request_id, time.monotonic() - started_at)
        return {"online_ingest_done": True}
    except Exception:
        logger.exception("ingest.error req=%s elapsed=%.2fs", request_id, time.monotonic() - started_at)
        raise


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
        client = OllamaClient()
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
                weights.append(getattr(settings, "ask_rrf_weight_bm25", 0.5))
            if "title" in lanes:
                sources.append(title_hits)
                weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
                if title_lexical_hits and len(title_hits) < min_docs:
                    sources.append(_dedupe_docs(title_lexical_hits))
                    weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))

            fused = rrf_fuse(sources, max_docs=settings.ask_max_docs, weights=weights)
            soft_language = False
            if filters_to_use.language:
                margin = _rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
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
                        weights.append(getattr(settings, "ask_rrf_weight_bm25", 0.5))
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
            if fused and _rrf_margin_ratio(fused, probe=expand_probe) < expand_ratio:
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
        allow_margin_fallback = bool(getattr(settings, "ask_fallback_allow_margin", False))

        def _relax(new_filters: RetrievalFilters, reason: str, allow_margin: bool = False) -> None:
            nonlocal filters, fused, top_chunks, chunk_candidates, counts, rrf_expanded
            if len(fused) >= min_docs:
                if not allow_margin:
                    return
                margin = _rrf_margin_ratio(fused, probe=expand_probe) if fused else 0.0
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
        if filters.since_date or filters.until_date:
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
        _relax(RetrievalFilters(), "no_filters")
        used_fallback = "+".join(fallbacks) if fallbacks else None

        logger.info(
            "retrieve.done req=%s candidates=%s chunks=%s sources=%s fallback=%s rrf_expand=%s elapsed=%.2fs",
            request_id,
            len(fused),
            len(chunk_candidates),
            counts,
            used_fallback,
            rrf_expanded,
            time.monotonic() - start,
        )
        return {
            "candidate_docs": fused,
            "top_chunks": top_chunks,
            "chunk_candidates": chunk_candidates,
            "query_embedding": query_embedding,
            "query_embeddings": embeddings,
            "filters": filters,
        }
    except Exception:
        logger.exception("retrieve.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise


def backfill_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    try:
        if not settings.backfill_enabled:
            logger.info("backfill.skip req=%s reason=disabled elapsed=%.2fs", request_id, time.monotonic() - start)
            return {"backfill_attempted": True}
        if state.get("candidate_docs"):
            logger.info("backfill.skip req=%s reason=candidates elapsed=%.2fs", request_id, time.monotonic() - start)
            return {"backfill_attempted": True}
        if state.get("backfill_attempted"):
            logger.info("backfill.skip req=%s reason=already_attempted elapsed=%.2fs", request_id, time.monotonic() - start)
            return {"backfill_attempted": True}

        with SessionLocal() as db:
            row = db.execute(
                sa_text("SELECT MIN(date) AS min_date FROM dogv_issues")
            ).mappings().one()
            min_date = row["min_date"]

        if min_date:
            target = min_date - timedelta(days=1)
            ensure_month_ingested(target, DEFAULT_LANGS)

        logger.info("backfill.done req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        return {"backfill_attempted": True}
    except Exception:
        logger.exception("backfill.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise


def _should_backfill(state: QAState) -> str:
    if not settings.backfill_enabled:
        return "rerank_titles"
    if state.get("candidate_docs"):
        return "rerank_titles"
    if state.get("backfill_attempted"):
        return "rerank_titles"
    return "backfill"


def rerank_titles_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    candidates = state.get("candidate_docs") or []
    all_candidates = candidates
    try:
        question = state["question"]
        keywords = _extract_keywords_simple(question)
        max_candidates = getattr(settings, "ask_rerank_max_candidates", 10)
        top_n = getattr(settings, "ask_rerank_top_n", 5)
        read_max_docs = getattr(settings, "ask_read_max_docs", 3)
        expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
        expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
        expand_candidates = getattr(settings, "ask_rerank_expand_candidates", 10)
        expand_top_n = getattr(settings, "ask_rerank_expand_top_n", 2)
        if candidates and _rrf_margin_ratio(candidates, probe=expand_probe) < expand_ratio:
            max_candidates = min(len(candidates), max_candidates + expand_candidates)
            top_n = min(len(candidates), top_n + expand_top_n)
        doc_scores = _doc_similarity_scores(
            state.get("query_embedding"),
            [int(item["document_id"]) for item in candidates],
        )
        if doc_scores:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    doc_scores.get(int(item["document_id"]), -1.0),
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
            for doc_id, items in chunk_candidates_by_doc.items():
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
                snippet = _best_snippet(question, chunk_list)
            if not snippet:
                snippet = (item.get("summary") or item.get("text") or "").strip()
            if not snippet:
                snippet = fallback_summaries.get(doc_id, "")
            if not snippet and doc_id in fallback_chunks:
                snippet = _best_snippet(question, fallback_chunks[doc_id])
            if snippet and keywords and _coverage_score(snippet, keywords) == 0:
                candidate_chunks = chunk_candidates_by_doc.get(doc_id) or []
                if candidate_chunks:
                    improved = _best_snippet(question, candidate_chunks)
                    if improved and _coverage_score(improved, keywords) > 0:
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
            logger.info(
                "rerank.skip req=%s candidates=%s elapsed=%.2fs",
                request_id,
                len(rerank_candidates),
                time.monotonic() - start,
            )
            return {"selected_doc_ids": doc_ids}
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
                score = _coverage_score(text, keywords)
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
        logger.info(
            "rerank.done req=%s selected=%s elapsed=%.2fs",
            request_id,
            len(doc_ids),
            time.monotonic() - start,
        )
        return {"selected_doc_ids": doc_ids}
    except Exception:
        logger.exception("rerank.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise




def read_docs_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    doc_ids = state.get("selected_doc_ids") or []
    if not doc_ids:
        logger.info("read.skip req=%s reason=no_docs elapsed=%.2fs", request_id, time.monotonic() - start)
        return {"evidence": []}
    question = state["question"]
    keywords = _extract_keywords_simple(question)
    needs_eligibility = _needs_eligibility(question)
    amount_extra = getattr(settings, "ask_read_amount_docs", 1)
    read_max_docs = getattr(settings, "ask_read_max_docs", 3)
    rerank_cap = getattr(settings, "ask_rerank_max_candidates", read_max_docs)
    read_max_docs = max(read_max_docs, min(len(doc_ids), rerank_cap))
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_docs = getattr(settings, "ask_read_expand_docs", 2)
    candidate_docs = state.get("candidate_docs") or []
    if candidate_docs and _rrf_margin_ratio(candidate_docs, probe=expand_probe) < expand_ratio:
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
                text = f"{text} {_best_snippet(question, chunk_list)}"
            score = _coverage_score(text, keywords)
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
    if needs_eligibility and _needs_amounts(question):
        eligibility_extra = max(eligibility_extra, 3)
    if needs_eligibility and eligibility_extra > 0 and candidate_docs:
        base_scored: list[tuple[int, int]] = []
        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            if _is_base_like(item):
                text = (item.get("title") or "")
                chunk_list = top_chunks.get(doc_id) or []
                if chunk_list:
                    text = f"{text} {_best_snippet(question, chunk_list)}"
                score = _coverage_score(text, keywords) if keywords else 0
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

    if amount_extra > 0 and candidate_docs and _needs_amounts(question):
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
        chunk_candidates = state.get("chunk_candidates") or []
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

        docs_by_id = {doc.id: (doc, issue) for doc, issue in docs}

        def _build_payload(
            target_ids: list[int],
            chunks_map: dict[int, list[dict[str, Any]]],
            extra_chunks: dict[int, list[dict[str, Any]]],
            ordered_chunks: dict[int, list[dict[str, Any]]],
        ):
            payload: list[dict[str, Any]] = []
            target_chunks = getattr(settings, "ask_chunks_per_doc", 4)
            chunk_max_chars = getattr(settings, "ask_chunk_max_chars", 1200)
            fallback_chars = getattr(settings, "ask_doc_fallback_chars", 12000)
            for doc_id in target_ids:
                row = docs_by_id.get(doc_id)
                if not row:
                    continue
                doc, issue = row
                primary = chunks_map.get(doc.id, [])
                fallback = extra_chunks.get(doc.id, [])
                merged = primary if primary else fallback
                if primary and fallback and len(primary) < target_chunks:
                    seen = {c.get("chunk_id") for c in primary}
                    merged = primary + [c for c in fallback if c.get("chunk_id") not in seen]
                if len(merged) < target_chunks:
                    extra = ordered_chunks.get(doc.id, [])
                    existing_text = {c.get("text") for c in merged if c.get("text")}
                    merged = merged + [c for c in extra if c.get("text") not in existing_text]
                chunks = [c["text"][:chunk_max_chars] for c in merged[:target_chunks]]
                if not chunks and doc.text:
                    chunks = [doc.text[:fallback_chars]]
                payload.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "chunks": chunks,
                    }
                )
            return payload

        def _build_full_docs(target_ids: list[int]) -> list[dict[str, Any]]:
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
                if len(full_docs_local) >= settings.full_doc_max_docs:
                    break
            return full_docs_local

        confidence_min = getattr(settings, "ask_doc_confidence_min", 0.06)
        top_score = float(candidate_docs[0].get("rrf_score", 0.0)) if candidate_docs else 0.0
        high_confidence = top_score >= confidence_min

        docs_payload = _build_payload(doc_ids, top_chunks, fallback_chunks, extra_chunks)
        evidence = extract_evidence(state["question"], docs_payload, full_docs=None)
        full_docs = []
        if evidence or high_confidence:
            full_docs = _build_full_docs(doc_ids)
            if not evidence and full_docs:
                evidence = extract_evidence(state["question"], docs_payload, full_docs=full_docs)
                if not evidence:
                    full_docs = []
        chunk_count = sum(len(doc.get("chunks") or []) for doc in docs_payload)
        char_count = sum(len(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
        token_est = sum(_estimate_tokens(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
        full_token_est = sum(_estimate_tokens(doc.get("text") or "") for doc in full_docs)
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
            time.monotonic() - start,
        )
        return {"evidence": evidence, "full_docs": full_docs}
    except Exception:
        logger.exception("read.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise


def answer_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    intent = state.get("intent") or {}
    language = state.get("language") or _map_language(intent.get("language"))
    evidence = state.get("evidence") or []
    full_docs = state.get("full_docs") or []
    def _parse_citation_id(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            for key in ("doc_id", "document_id", "id"):
                if key in value:
                    return _parse_citation_id(value.get(key))
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text.startswith("doc_"):
                text = text[4:]
            if text.isdigit():
                return int(text)
        return None
    try:
        if not evidence and not full_docs:
            logger.info("answer.skip req=%s reason=no_evidence elapsed=%.2fs", request_id, time.monotonic() - start)
            return no_evidence_answer(language)
        result = build_answer(state["question"], language, evidence, full_docs=full_docs)
        answer = result.get("answer") or ""
        cited_ids: set[int] = set()
        for raw in result.get("citations") or []:
            parsed = _parse_citation_id(raw)
            if parsed is not None:
                cited_ids.add(parsed)

        citations = []
        if not cited_ids and evidence:
            for item in evidence:
                parsed = _parse_citation_id(item)
                if parsed is not None:
                    cited_ids.add(parsed)
        if cited_ids:
            with SessionLocal() as db:
                docs = (
                    db.query(DogvDocument, DogvIssue)
                    .join(DogvIssue)
                    .filter(DogvDocument.id.in_(cited_ids))
                    .all()
                )
            for doc, issue in docs:
                citations.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "pdf_url": build_pdf_url(doc.pdf_url),
                        "html_url": build_html_url(doc.html_url),
                    }
                )

        logger.info(
            "answer.done req=%s citations=%s evidence=%s full_docs=%s elapsed=%.2fs",
            request_id,
            len(citations),
            len(evidence),
            len(full_docs),
            time.monotonic() - start,
        )
        return {"answer": answer, "citations": citations}
    except Exception:
        logger.exception("answer.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise


def build_graph():
    graph = StateGraph(QAState)
    graph.add_node("analyze_intent", analyze_intent_node)
    graph.add_node("temporal_guard", temporal_guard_node)
    graph.add_node("online_ingest", online_ingest_node)
    graph.add_node("retrieve_candidates", retrieve_candidates_node)
    graph.add_node("backfill", backfill_node)
    graph.add_node("rerank_titles", rerank_titles_node)
    graph.add_node("read_docs", read_docs_node)
    graph.add_node("answer_node", answer_node)

    graph.set_entry_point("analyze_intent")
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
