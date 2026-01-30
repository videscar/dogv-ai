from __future__ import annotations

from typing import Any

from .config import get_settings
from .ollama import OllamaClient


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
RERANK_TIMEOUT = 45

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


def rerank_titles(
    question: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    return_all: bool = False,
) -> list[int]:
    if not candidates:
        return []

    lines = [_format_candidate(item) for item in candidates]

    client = OllamaClient(timeout=min(settings.ollama_timeout, RERANK_TIMEOUT))
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
        result = client.chat_json(messages, temperature=0.0)
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
