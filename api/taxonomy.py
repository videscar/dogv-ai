from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class TaxonomyEntry:
    label: str
    aliases: tuple[str, ...]
    hints: tuple[str, ...]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", lowered) if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped).strip()


DOC_KIND_ENTRIES = [
    TaxonomyEntry(
        label="Empleo Publico",
        aliases=(
            "empleo publico",
            "ocupacion publica",
            "ocupacio publica",
            "empleo",
            "ocupacio",
            "ocupacion",
        ),
        hints=(
            "empleo publico",
            "ocupacion publica",
            "ocupacio publica",
            "oferta de empleo",
            "oferta publica de empleo",
            "oferta d'ocupacio",
            "ocupacio publica",
        ),
    ),
    TaxonomyEntry(
        label="Ayudas",
        aliases=("ayudas", "ajudas", "ajudes", "ajuda"),
        hints=("ayudas", "ajudas", "ajudes", "ajuda"),
    ),
    TaxonomyEntry(
        label="Subvenciones",
        aliases=("subvenciones", "subvencions", "subvencion", "subvencio"),
        hints=("subvenciones", "subvencions", "subvencion", "subvencio"),
    ),
    TaxonomyEntry(
        label="Premios",
        aliases=("premios", "premis", "premio", "premi"),
        hints=("premios", "premis", "premio", "premi"),
    ),
    TaxonomyEntry(
        label="Becas",
        aliases=("becas", "beca", "beques"),
        hints=("becas", "beques", "beca"),
    ),
    TaxonomyEntry(
        label="Otros",
        aliases=("otros", "altres"),
        hints=(),
    ),
]

DOC_SUBKIND_ENTRIES = [
    TaxonomyEntry(
        label="OPE",
        aliases=(
            "ope",
            "oferta de empleo publico",
            "oferta publica de empleo",
            "oferta publica d'ocupacio",
            "oferta d'ocupacio publica",
            "oferta d'ocupacio",
        ),
        hints=(
            "ope",
            "oferta de empleo publico",
            "oferta publica de empleo",
            "oferta d'ocupacio",
        ),
    ),
    TaxonomyEntry(
        label="Convocatoria",
        aliases=("convocatoria", "convocatories"),
        hints=("convocatoria", "convocatories", "convoca", "convocan"),
    ),
    TaxonomyEntry(
        label="Listas",
        aliases=(
            "listas",
            "llistes",
            "lista",
            "llista",
            "relacion",
            "relacio",
            "relaciones",
        ),
        hints=("listas", "llistes", "lista", "llista", "relacion", "relacio"),
    ),
    TaxonomyEntry(
        label="Tribunales",
        aliases=("tribunales", "tribunals", "tribunal"),
        hints=("tribunal", "tribunales", "tribunals"),
    ),
    TaxonomyEntry(
        label="Resultados",
        aliases=("resultados", "resultats", "resultado", "resultat", "aprobados", "aprovats"),
        hints=("resultados", "resultats", "aprobados", "aprovats"),
    ),
    TaxonomyEntry(
        label="Correcciones",
        aliases=("correcciones", "correccion", "correccio", "errata", "errades"),
        hints=("correcciones", "correccion", "correccio", "errata", "errades"),
    ),
    TaxonomyEntry(
        label="Bases",
        aliases=("bases", "bases reguladoras", "bases reguladores", "base reguladora"),
        hints=("bases", "bases reguladoras", "bases reguladores", "base reguladora"),
    ),
    TaxonomyEntry(
        label="Adjudicaciones",
        aliases=(
            "adjudicaciones",
            "adjudicacion",
            "adjudicacions",
            "adjudicacio",
            "concesion",
            "concessio",
            "concedeix",
            "concedido",
            "concedit",
        ),
        hints=("adjudicaciones", "adjudicacion", "adjudicacions", "concesion", "concessio"),
    ),
]


_DOC_KIND_MAP = {
    normalize_text(alias): entry.label
    for entry in DOC_KIND_ENTRIES
    for alias in (entry.label,) + entry.aliases
}
_DOC_SUBKIND_MAP = {
    normalize_text(alias): entry.label
    for entry in DOC_SUBKIND_ENTRIES
    for alias in (entry.label,) + entry.aliases
}

_DOC_KIND_HINTS = {entry.label: entry.hints for entry in DOC_KIND_ENTRIES}
_DOC_SUBKIND_HINTS = {entry.label: entry.hints for entry in DOC_SUBKIND_ENTRIES}


def normalize_doc_kind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    return _DOC_KIND_MAP.get(normalized)


def normalize_doc_subkind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    return _DOC_SUBKIND_MAP.get(normalized)


def _contains_hint(question: str, hint: str) -> bool:
    if not hint:
        return False
    normalized_question = normalize_text(question)
    normalized_hint = normalize_text(hint)
    if " " in normalized_hint:
        return normalized_hint in normalized_question
    return bool(re.search(rf"\b{re.escape(normalized_hint)}\b", normalized_question))


def should_filter_doc_kind(question: str, doc_kind: str | None) -> bool:
    if not doc_kind:
        return False
    hints = _DOC_KIND_HINTS.get(doc_kind, ())
    if not hints:
        return False
    return any(_contains_hint(question, hint) for hint in hints)


def should_filter_doc_subkind(question: str, doc_subkind: str | None) -> bool:
    if not doc_subkind:
        return False
    hints = _DOC_SUBKIND_HINTS.get(doc_subkind, ())
    if not hints:
        return False
    return any(_contains_hint(question, hint) for hint in hints)


def canonical_doc_kind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    return _DOC_KIND_MAP.get(normalized) or value


def canonical_doc_subkind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_text(value)
    return _DOC_SUBKIND_MAP.get(normalized) or value
