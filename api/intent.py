from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Any

from .llm import LlmClient
from .query_expansion import normalize_expansion_terms
from .taxonomy import canonical_doc_kind, canonical_doc_subkind, normalize_text


def needs_amounts(question: str) -> bool:
    if not question:
        return False
    return bool(
        re.search(r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", question, re.IGNORECASE)
    )


def needs_eligibility(question: str) -> bool:
    return bool(
        re.search(
            r"\b(qui|qui[eé]n|beneficiar|beneficiari|beneficiario|sol·licit[a-z]*|solicit[a-z]*|requisit|requisito|destinatari|destinatario|pot|puede|poden|pueden)\b",
            question,
            re.IGNORECASE,
        )
    )


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


_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")

_MONTH_NUMBERS = {
    "enero": 1, "gener": 1,
    "febrero": 2, "febrer": 2,
    "marzo": 3, "març": 3,
    "abril": 4,
    "mayo": 5, "maig": 5,
    "junio": 6, "juny": 6,
    "julio": 7, "juliol": 7,
    "agosto": 8, "agost": 8,
    "septiembre": 9, "setiembre": 9, "setembre": 9,
    "octubre": 10,
    "noviembre": 11, "novembre": 11,
    "diciembre": 12, "desembre": 12,
}  # fmt: skip

_MONTH_YEAR_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTH_NUMBERS) + r")\s+(?:de|del)\s+(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)


def _infer_month_range(question: str | None) -> tuple[date | None, date | None]:
    """Month window from explicit 'agost de 2024'/'julio de 2025' phrases (es/va).

    The intent LLM regularly misses these, especially in Valencian; without a
    window the December re-editions of an August act outrank it and the
    edition-recency pruning treats the question as recency-ambiguous. Multiple
    hits span min..max month."""
    if not question:
        return None, None
    stamps = []
    for match in _MONTH_YEAR_PATTERN.finditer(question):
        month = _MONTH_NUMBERS[match.group(1).lower()]
        stamps.append((int(match.group(2)), month))
    if not stamps:
        return None, None
    start_year, start_month = min(stamps)
    end_year, end_month = max(stamps)
    last_day = calendar.monthrange(end_year, end_month)[1]
    return date(start_year, start_month, 1), date(end_year, end_month, last_day)


def _infer_year_range(question: str | None) -> tuple[date | None, date | None]:
    if not question:
        return None, None
    years = [int(match.group(1)) for match in _YEAR_PATTERN.finditer(question)]
    if not years:
        return None, None
    start_year = min(years)
    end_year = max(years)
    return date(start_year, 1, 1), date(end_year, 12, 31)


def normalize_intent(raw: Any, question: str | None = None) -> dict[str, Any]:
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
    if since_date is None and until_date is None:
        inferred_since, inferred_until = _infer_month_range(question)
        if inferred_since is None:
            inferred_since, inferred_until = _infer_year_range(question)
        since_date = inferred_since
        until_date = inferred_until
    needs_online = (
        bool(data.get("needs_online")) if isinstance(data.get("needs_online"), bool) else False
    )

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
    client = LlmClient()
    messages = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": INTENT_USER.format(question=question)},
    ]
    result = client.chat_json(messages, temperature=0.0, enable_thinking=False)
    return normalize_intent(result, question=question)


def analyze_intent_and_expand(
    question: str,
    max_keywords: int = 6,
    max_phrases: int = 4,
    max_tokens: int = 8,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    client = LlmClient()
    messages = [
        {"role": "system", "content": INTENT_EXPAND_SYSTEM},
        {"role": "user", "content": INTENT_EXPAND_USER.format(question=question)},
    ]
    try:
        result = client.chat_json(messages, temperature=0.0, enable_thinking=False)
        intent = normalize_intent(result, question=question)
        expansion_obj = result.get("expansion") if isinstance(result, dict) else None
        raw_keywords = _string_list(
            result.get("expansion_keywords") if isinstance(result, dict) else None
        )
        raw_phrases = _string_list(
            result.get("expansion_phrases") if isinstance(result, dict) else None
        )
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
