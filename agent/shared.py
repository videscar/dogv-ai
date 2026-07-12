from __future__ import annotations

import re
from typing import Any, TypedDict

from api.retrieval import RetrievalFilters


class QAState(TypedDict, total=False):
    question: str
    debug: bool
    # Prior conversation turns ({"role": "user"|"assistant", "content": str}),
    # most-recent last. Empty for single-turn requests.
    history: list[dict[str, Any]]
    # The user's verbatim latest message, preserved when contextualize_query
    # rewrites `question` into a standalone form for retrieval.
    raw_question: str
    contextualized: bool
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
    # Reader payload (docs + the exact chunks shown to the evidence extractor),
    # kept in state for /ask debug introspection only.
    read_payload: list[dict[str, Any]]
    full_docs: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    backfill_attempted: bool
    ondemand_doc_id: int | None
    ondemand_ref: str | None
    norm_pin_doc_ids: list[int]
    # In-pool docs a second hop identified as a distinct entity's best evidence:
    # protected from edition-recency suppression, but not force-read like pins.
    second_hop_protect_ids: list[int]
    norm_target_ref: dict[str, str] | None
    feed_query: bool
    online_ingest_done: bool
    profile: dict[str, Any]


def merge_profile(
    state: QAState,
    stage: str,
    **metrics: Any,
) -> dict[str, Any]:
    profile = dict(state.get("profile") or {})
    stage_metrics = dict(profile.get(stage) or {})
    stage_metrics.update(metrics)
    profile[stage] = stage_metrics
    return profile


def return_with_profile(
    state: QAState,
    stage: str,
    payload: dict[str, Any],
    **metrics: Any,
) -> QAState:
    result = dict(payload)
    result["profile"] = merge_profile(state, stage, **metrics)
    return result


def rrf_margin_ratio(candidates: list[dict[str, Any]], probe: int = 5) -> float:
    if not candidates:
        return 0.0
    top_score = float(candidates[0].get("rrf_score") or 0.0)
    if top_score <= 0:
        return 0.0
    idx = min(len(candidates) - 1, max(1, probe - 1))
    pivot = float(candidates[idx].get("rrf_score") or 0.0)
    return (top_score - pivot) / top_score


RERANK_STOPWORDS = {
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


def extract_keywords_simple(text: str) -> list[str]:
    tokens = re.findall(r"[\w·'-]+", (text or "").lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 4 or token in RERANK_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def coverage_score(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    lower = text.lower()
    return sum(lower.count(k) for k in keywords)


def best_snippet(question: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    keywords = extract_keywords_simple(question)
    if not keywords:
        return (chunks[0].get("text") or "").strip()
    best_text = ""
    best_score = -1
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        score = coverage_score(text, keywords)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text or (chunks[0].get("text") or "").strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)
