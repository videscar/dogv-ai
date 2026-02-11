from __future__ import annotations

from typing import Any

from .ollama import OllamaClient


CLASSIFY_SYSTEM = (
    "Eres un clasificador de documentos del DOGV. "
    "Devuelve SOLO JSON con campos: doc_kind, doc_subkind, confidence (0-1), tags (lista)."
)

CLASSIFY_USER = """Titulo:
{title}

Extracto:
{excerpt}

Categorias doc_kind permitidas:
- Empleo Publico
- Ayudas
- Subvenciones
- Premios
- Becas
- Otros

Subcategorias doc_subkind permitidas (si aplica):
- OPE
- Convocatoria
- Listas
- Tribunales
- Resultados
- Correcciones
- Bases
- Adjudicaciones
"""


def classify_document(
    title: str | None,
    text: str | None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    excerpt = (text or "")[:2000]
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": CLASSIFY_USER.format(title=title or "", excerpt=excerpt)},
    ]
    client = OllamaClient(base_url=base_url, model=model, timeout=timeout)
    return client.chat_json(messages, temperature=0.0)
