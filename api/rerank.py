from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from agent.shared import coverage_score

from .config import get_settings
from .llm import LlmClient

RERANK_SYSTEM = (
    "Eres un reranker para documentos del DOGV. "
    "Prioriza recall: incluye todos los documentos plausibles, "
    "pero selecciona los mas utiles para responder. "
    "Asegura que el tema central de la pregunta aparece en el titulo o extracto. "
    "aunque no estes seguro del orden exacto. "
    "Devuelve SOLO JSON con campo doc_ids (lista ordenada). "
    "No inventes ids fuera de la lista."
)

settings = get_settings()
RERANK_TIMEOUT = 120

RERANK_USER = """Pregunta:
{question}

Documentos candidatos (id | fecha | doc_kind | doc_subkind | ref | titulo | extracto):
{candidates}

Devuelve hasta {top_n} ids en orden de prioridad.
"""


def _shorten(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def _format_candidate(item: dict[str, Any]) -> str:
    snippet = item.get("snippet") or item.get("summary") or item.get("text") or ""
    return (
        f"{item['document_id']} | {item.get('issue_date') or ''} | "
        f"{item.get('doc_kind') or ''} | {item.get('doc_subkind') or ''} | "
        f"{item.get('ref') or ''} | {_shorten(item.get('title'))} | {_shorten(snippet)}"
    )


def asks_most_recent(question: str) -> bool:
    if not question:
        return False
    return bool(
        re.search(
            r"(m[aá]s reciente|m[ée]s recent|mes recent|[uú]ltim[oa]?|ultim[oa]?|latest)",
            question.lower(),
        )
    )


def parse_issue_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            if len(raw) >= 10:
                return date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return None


def prepend_recent_relevant_docs(
    question: str,
    keywords: list[str],
    rerank_candidates: list[dict[str, Any]],
    doc_ids: list[int],
) -> list[int]:
    if not asks_most_recent(question) or not rerank_candidates:
        return doc_ids

    keep_n = max(1, int(getattr(settings, "ask_rerank_recent_keep", 2)))
    scored: list[tuple[int, int, int]] = []
    for item in rerank_candidates:
        doc_id = int(item["document_id"])
        issue_date = parse_issue_date(item.get("issue_date"))
        if issue_date is None:
            continue
        text = f"{item.get('title') or ''} {item.get('snippet') or ''}"
        coverage = coverage_score(text, keywords) if keywords else 1
        if keywords and coverage <= 0:
            continue
        scored.append((coverage, issue_date.toordinal(), doc_id))

    if not scored:
        return doc_ids

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    recent_doc_ids = [doc_id for _, _, doc_id in scored[:keep_n]]
    merged: list[int] = []
    seen: set[int] = set()
    for doc_id in recent_doc_ids + doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)
    return merged


def rerank_titles(
    question: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    return_all: bool = False,
) -> list[int]:
    if not candidates:
        return []

    lines = [_format_candidate(item) for item in candidates]

    client = LlmClient(timeout=min(settings.llm_timeout, RERANK_TIMEOUT))
    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {
            "role": "user",
            "content": RERANK_USER.format(
                question=question,
                candidates="\n".join(lines),
                top_n=top_n,
            ),
        },
    ]

    doc_ids: list[int] = []
    try:
        result = client.chat_json(messages, temperature=0.0, enable_thinking=False)
        doc_ids = [int(x) for x in (result.get("doc_ids") or [])]
    except Exception:
        pass

    candidate_ids = {int(item["document_id"]) for item in candidates}
    seen: set[int] = set()
    merged: list[int] = []

    for doc_id in doc_ids:
        if doc_id in seen or doc_id not in candidate_ids:
            continue
        seen.add(doc_id)
        merged.append(doc_id)
    for item in candidates:
        doc_id = int(item["document_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)

    if return_all:
        return merged
    return merged[:top_n]
