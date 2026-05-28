from __future__ import annotations

import unicodedata
from typing import Any

from .llm import LlmClient


# Canonical labels the rest of the pipeline (retrieval filters, metadata) relies
# on. The LLM occasionally returns accented, singular, or invented variants, so
# we normalize its output back onto these instead of storing it verbatim.
DOC_KINDS = ("Empleo Publico", "Ayudas", "Subvenciones", "Premios", "Becas", "Otros")
DOC_SUBKINDS = (
    "OPE",
    "Convocatoria",
    "Listas",
    "Tribunales",
    "Resultados",
    "Correcciones",
    "Bases",
    "Adjudicaciones",
)


def _normalize_label(value: str | None) -> str:
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.strip().lower()


# Common singular / variant forms the model emits, mapped to canonical labels.
# (Spanish plurals aren't a simple trailing-"s", e.g. subvención -> subvenciones.)
_DOC_KIND_ALIASES = {
    "empleo": "Empleo Publico",
    "ayuda": "Ayudas",
    "subvencion": "Subvenciones",
    "premio": "Premios",
    "beca": "Becas",
    "otro": "Otros",
}
_DOC_SUBKIND_ALIASES = {
    "lista": "Listas",
    "tribunal": "Tribunales",
    "resultado": "Resultados",
    "correccion": "Correcciones",
    "base": "Bases",
    "adjudicacion": "Adjudicaciones",
}


def _build_lookup(canonical: tuple[str, ...], aliases: dict[str, str]) -> dict[str, str]:
    lookup = {_normalize_label(label): label for label in canonical}
    for alias, label in aliases.items():
        lookup.setdefault(_normalize_label(alias), label)
    return lookup


_DOC_KIND_LOOKUP = _build_lookup(DOC_KINDS, _DOC_KIND_ALIASES)
_DOC_SUBKIND_LOOKUP = _build_lookup(DOC_SUBKINDS, _DOC_SUBKIND_ALIASES)


def normalize_doc_kind(value: str | None) -> str:
    """Map a raw model label onto a canonical doc_kind, defaulting to 'Otros'."""
    return _DOC_KIND_LOOKUP.get(_normalize_label(value), "Otros")


def normalize_doc_subkind(value: str | None) -> str | None:
    """Map a raw model label onto a canonical doc_subkind, or None if unknown."""
    return _DOC_SUBKIND_LOOKUP.get(_normalize_label(value))


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
    client = LlmClient(base_url=base_url, model=model, timeout=timeout)
    result = client.chat_json(messages, temperature=0.0, enable_thinking=False)
    result["doc_kind"] = normalize_doc_kind(result.get("doc_kind"))
    result["doc_subkind"] = normalize_doc_subkind(result.get("doc_subkind"))
    return result
