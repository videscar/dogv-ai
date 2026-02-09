from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .ollama import OllamaClient
from .query_expansion import normalize_expansion_terms
from .taxonomy import canonical_doc_kind, canonical_doc_subkind, normalize_text


INTENT_SYSTEM = (
    "Eres un analista de intencion para consultas del DOGV. "
    "Devuelve SOLO JSON con estos campos: "
    "language (es|ca), doc_kind, doc_subkind, keywords (lista), "
    "since_date (YYYY-MM-DD o null), until_date (YYYY-MM-DD o null), "
    "needs_online (true|false), entities (objeto con organismo, municipio, cuerpo). "
    "Si no puedes determinar doc_kind o doc_subkind, usa Otros o Desconocido."
)

INTENT_USER = """Pregunta del usuario:
{question}

Categorias principales permitidas para doc_kind (si dudas, usa Otros/Desconocido):
- Empleo Publico
- Ayudas
- Subvenciones
- Premios
- Becas
- Otros

Subcategorias frecuentes para doc_subkind (si aplica; si dudas, usa Otros/Desconocido):
- OPE
- Convocatoria
- Listas
- Tribunales
- Resultados
- Correcciones
- Bases
- Adjudicaciones
- Otros
"""

INTENT_EXPAND_SYSTEM = (
    "Eres un analista de intencion para consultas del DOGV. "
    "Devuelve SOLO JSON con estos campos: "
    "language (es|ca), doc_kind, doc_subkind, keywords (lista), "
    "since_date (YYYY-MM-DD o null), until_date (YYYY-MM-DD o null), "
    "needs_online (true|false), entities (objeto con organismo, municipio, cuerpo), "
    "expansion_keywords (lista), expansion_phrases (lista). "
    "expansion_keywords: solo palabras sueltas, evitando terminos genericos. "
    "expansion_phrases: frases de 2 a 5 palabras relevantes para recuperar documentos. "
    "Si no puedes determinar doc_kind o doc_subkind, usa Otros o Desconocido."
)

INTENT_EXPAND_USER = """Pregunta del usuario:
{question}

Categorias principales permitidas para doc_kind (si dudas, usa Otros/Desconocido):
- Empleo Publico
- Ayudas
- Subvenciones
- Premios
- Becas
- Otros

Subcategorias frecuentes para doc_subkind (si aplica; si dudas, usa Otros/Desconocido):
- OPE
- Convocatoria
- Listas
- Tribunales
- Resultados
- Correcciones
- Bases
- Adjudicaciones
- Otros
"""

_UNKNOWN_LABELS = {
    "unknown",
    "desconocido",
    "desconocida",
    "desconocidos",
    "desconocidas",
    "desconegut",
    "desconeguda",
    "desconeguts",
    "desconegudes",
    "otros",
    "otro",
    "other",
    "n/a",
    "na",
    "none",
    "null",
}


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _clean_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _normalize_language(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lang = value.strip().lower()
    return lang if lang in ("es", "ca") else None


def normalize_intent(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}

    language = _normalize_language(data.get("language"))

    doc_kind_raw = _clean_label(data.get("doc_kind"))
    doc_kind = canonical_doc_kind(doc_kind_raw) if doc_kind_raw else None
    if doc_kind_raw and not doc_kind:
        normalized = normalize_text(doc_kind_raw)
        if normalized in _UNKNOWN_LABELS:
            doc_kind = "Otros"
        else:
            doc_kind = "Otros"

    doc_subkind_raw = _clean_label(data.get("doc_subkind"))
    doc_subkind = canonical_doc_subkind(doc_subkind_raw) if doc_subkind_raw else None
    if doc_subkind_raw and not doc_subkind:
        normalized = normalize_text(doc_subkind_raw)
        if normalized in _UNKNOWN_LABELS:
            doc_subkind = "Otros"

    keywords: list[str] = []
    raw_keywords = data.get("keywords")
    if isinstance(raw_keywords, list):
        for item in raw_keywords:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    keywords.append(token)
    elif isinstance(raw_keywords, str):
        token = raw_keywords.strip()
        if token:
            keywords.append(token)

    entities: dict[str, str] = {}
    raw_entities = data.get("entities")
    if isinstance(raw_entities, dict):
        for key in ("organismo", "municipio", "cuerpo"):
            value = raw_entities.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    entities[key] = text

    since_date = _parse_date(data.get("since_date"))
    until_date = _parse_date(data.get("until_date"))
    needs_online = bool(data.get("needs_online")) if isinstance(data.get("needs_online"), bool) else False

    return {
        "language": language,
        "doc_kind": doc_kind,
        "doc_subkind": doc_subkind,
        "keywords": keywords,
        "since_date": since_date,
        "until_date": until_date,
        "needs_online": needs_online,
        "entities": entities,
    }


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def analyze_intent(question: str) -> dict[str, Any]:
    client = OllamaClient()
    messages = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": INTENT_USER.format(question=question)},
    ]
    result = client.chat_json(messages, temperature=0.0)
    return normalize_intent(result)


def analyze_intent_and_expand(
    question: str,
    max_keywords: int = 6,
    max_phrases: int = 4,
    max_tokens: int = 8,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    client = OllamaClient()
    messages = [
        {"role": "system", "content": INTENT_EXPAND_SYSTEM},
        {"role": "user", "content": INTENT_EXPAND_USER.format(question=question)},
    ]
    try:
        result = client.chat_json(messages, temperature=0.0)
        intent = normalize_intent(result)
        expansion_obj = result.get("expansion") if isinstance(result, dict) else None
        raw_keywords = _string_list(result.get("expansion_keywords") if isinstance(result, dict) else None)
        raw_phrases = _string_list(result.get("expansion_phrases") if isinstance(result, dict) else None)
        if isinstance(expansion_obj, dict):
            if not raw_keywords:
                raw_keywords = _string_list(expansion_obj.get("keywords"))
            if not raw_phrases:
                raw_phrases = _string_list(expansion_obj.get("phrases"))
        expansion = normalize_expansion_terms(
            question,
            raw_keywords,
            raw_phrases,
            max_keywords=max_keywords,
            max_phrases=max_phrases,
            max_tokens=max_tokens,
        )
        return intent, expansion
    except Exception:
        intent = analyze_intent(question)
        from .query_expansion import llm_expand_query

        expansion = llm_expand_query(
            question,
            intent,
            max_keywords=max_keywords,
            max_phrases=max_phrases,
            max_tokens=max_tokens,
        )
        return intent, expansion
