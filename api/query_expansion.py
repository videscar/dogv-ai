from __future__ import annotations

from typing import Any
import re

from .ollama import OllamaClient

_STOPWORDS = {
    "que",
    "qui",
    "quin",
    "quina",
    "quins",
    "quines",
    "com",
    "quan",
    "on",
    "per",
    "amb",
    "des",
    "para",
    "por",
    "sobre",
    "entre",
    "desde",
    "hasta",
    "sin",
    "segun",
    "según",
    "del",
    "de",
    "la",
    "el",
    "les",
    "els",
    "los",
    "las",
    "una",
    "uno",
    "unos",
    "unas",
    "nou",
    "nova",
    "nous",
    "noves",
    "data",
    "generalitat",
    "consell",
    "conselleria",
    "presidencia",
    "presidència",
    "departament",
    "govern",
    "ha",
    "han",
    "s'ha",
    "es",
    "se",
    "si",
    "sí",
    "ja",
    "ya",
    "esta",
    "este",
    "aquesta",
    "aquest",
    "hoy",
    "ahir",
    "ayer",
    "donde",
    "dónde",
    "cual",
    "cuál",
    "cuales",
    "cuáles",
    "como",
    "cómo",
    "quiero",
    "ver",
    "buscar",
    "encontrar",
    "consultar",
    "participar",
    "poder",
    "puedo",
    "podria",
    "podría",
    "algun",
    "algún",
    "alguna",
    "algunos",
    "algunas",
    "cualquier",
    "cualquiera",
    "cualesquiera",
    "sera",
    "serà",
    "esta",
    "aquesta",
    "nou",
    "nova",
    "nous",
    "noves",
}

_SHORT_TOKEN_ALLOWLIST = {
    "ope",
    "umh",
    "ivass",
    "gva",
    "c1",
    "c2",
    "a1",
    "a2",
    "b1",
    "b2",
}

_GENERIC_TERMS = {
    "anuncio",
    "resolucion",
    "resolución",
    "decreto",
    "orden",
    "acuerdo",
    "convocatoria",
    "convocar",
    "bases",
    "base",
    "lista",
    "listado",
    "listes",
    "listes",
    "admitidos",
    "admitidas",
    "excluidos",
    "excluidas",
    "tribunal",
    "tribunales",
    "concurso",
    "oposicion",
    "oposición",
    "publicado",
    "publicada",
    "publicacion",
    "publicación",
    "modificacion",
    "modificación",
    "correccion",
    "corrección",
    "nombramiento",
    "cese",
    "plaza",
    "plazas",
    "puesto",
    "puestos",
    "oferta",
    "empleo",
    "publico",
    "público",
    "provision",
    "provisión",
    "resolucio",
    "resolució",
    "ayuntamiento",
    "universidad",
    "universitat",
    "consorcio",
    "generalitat",
    "personal",
    "convocado",
    "convocada",
    "convocó",
    "salido",
    "publicaron",
    "aprobó",
    "aprobado",
}

_LOW_SIGNAL_TERMS = {
    "publicado",
    "publicada",
    "publicacion",
    "publicación",
    "publicar",
    "salido",
    "salida",
    "salir",
    "aprobado",
    "aprobada",
    "aprobó",
    "aprobacion",
    "aprobación",
    "aprobar",
    "convocado",
    "convocada",
    "convocar",
    "modificado",
    "modificada",
    "modificar",
}

_FEED_PATTERNS = [
    r"\besta\s+semana\b",
    r"\baquesta\s+setmana\b",
    r"\besta\s+setmana\b",
    r"\bultim[oa]s?\s+semana[s]?\b",
    r"\búltim[oa]s?\s+semana[s]?\b",
    r"\bultim[oa]s?\s+d[ií]as\b",
    r"\búltim[oa]s?\s+d[ií]as\b",
    r"\bdarrers?\s+dies\b",
    r"\bnovedades?\b",
    r"\bnuevas?\s+convocatorias?\b",
    r"\búltimes?\s+convocatòries\b",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w·'/-]+", text.lower())


def _keep_token(token: str) -> bool:
    if len(token) >= 4:
        return True
    if token in _SHORT_TOKEN_ALLOWLIST:
        return True
    if any(ch.isdigit() for ch in token) and len(token) >= 2:
        return True
    if re.fullmatch(r"[a-z]\d", token):
        return True
    return False


_CONNECTORS = {
    "de",
    "del",
    "d'",
    "l'",
    "la",
    "el",
    "i",
    "y",
    "da",
    "do",
    "dos",
    "des",
    "en",
    "para",
    "per",
    "a",
    "al",
}


def _extract_phrases(question: str, keywords: list[str] | None = None) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def _add_phrase(value: str) -> None:
        phrase = _clean_phrase(value)
        if not phrase:
            return
        if phrase in seen:
            return
        if not _phrase_is_useful(phrase):
            return
        seen.add(phrase)
        phrases.append(phrase)

    if keywords:
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            if " " in keyword:
                _add_phrase(keyword)

    for match in re.findall(r"\"([^\"]+)\"", question or ""):
        _add_phrase(match)

    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9'·/-]+", question or "")
    seq: list[str] = []

    def is_entity_token(tok: str) -> bool:
        return tok[:1].isupper() or tok.isupper() or any(ch.isdigit() for ch in tok)

    def flush() -> None:
        if len(seq) >= 2 and any(is_entity_token(t) for t in seq):
            _add_phrase(" ".join(seq))
        seq.clear()

    for tok in tokens:
        low = tok.lower()
        if is_entity_token(tok):
            seq.append(tok)
            continue
        if seq and low in _CONNECTORS:
            seq.append(low)
            continue
        flush()
    flush()

    for phrase in _extract_content_phrases(question):
        _add_phrase(phrase)

    return phrases


def _phrase_is_useful(phrase: str) -> bool:
    if any(ch.isdigit() for ch in phrase):
        return True
    tokens = [token for token in phrase.split() if token not in _CONNECTORS]
    if not tokens:
        return False
    return any(token.lower() not in _LOW_SIGNAL_TERMS for token in tokens)


def _extract_content_phrases(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9'·/-]+", question or "")
    phrases: list[str] = []
    seq: list[str] = []
    content_positions: list[int] = []

    def _emit() -> None:
        if len(content_positions) < 2:
            return
        max_content = 3
        for i in range(len(content_positions)):
            for j in range(i + 1, min(len(content_positions), i + max_content)):
                start = content_positions[i]
                end = content_positions[j]
                phrase = " ".join(seq[start : end + 1])
                phrases.append(phrase)
                if len(phrases) >= 8:
                    return

    def _flush() -> None:
        nonlocal seq, content_positions
        if seq:
            _emit()
        seq = []
        content_positions = []

    for tok in tokens:
        low = tok.lower()
        if low in _CONNECTORS:
            if seq:
                seq.append(low)
            continue
        if low in _STOPWORDS:
            _flush()
            if len(phrases) >= 8:
                break
            continue
        if not _keep_token(low):
            _flush()
            if len(phrases) >= 8:
                break
            continue
        seq.append(low)
        content_positions.append(len(seq) - 1)
        if len(content_positions) >= 5:
            _flush()
            if len(phrases) >= 8:
                break
    _flush()
    return phrases


def _prioritize_generic_terms(terms: list[str]) -> list[str]:
    primary = [t for t in terms if t not in _LOW_SIGNAL_TERMS]
    secondary = [t for t in terms if t in _LOW_SIGNAL_TERMS]
    primary.sort(key=len, reverse=True)
    secondary.sort(key=len, reverse=True)
    return primary + secondary


def _select_anchor_terms(
    question: str,
    keywords: list[str] | None,
    max_terms: int = 3,
    phrase_tokens: set[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    for token in _tokenize(question):
        if not _keep_token(token):
            continue
        if token in _STOPWORDS:
            continue
        if token in _LOW_SIGNAL_TERMS:
            continue
        candidates.append(token)
    if isinstance(keywords, list):
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            token = keyword.strip().lower()
            if " " in token:
                continue
            if not _keep_token(token):
                continue
            if token in _STOPWORDS or token in _LOW_SIGNAL_TERMS:
                continue
            candidates.append(token)

    def score(token: str) -> int:
        value = len(token)
        if token in _GENERIC_TERMS:
            value -= 1
        else:
            value += 2
        if phrase_tokens and token in phrase_tokens:
            value += 2
        if any(ch.isdigit() for ch in token):
            value += 2
        if token in _SHORT_TOKEN_ALLOWLIST:
            value += 1
        return value

    unique = sorted({tok for tok in candidates}, key=lambda tok: (-score(tok), tok))
    return unique[:max_terms]


def _rank_phrase_tokens(phrases: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for phrase in phrases:
        for token in phrase.split():
            if token in _CONNECTORS:
                continue
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return []
    return sorted(counts, key=lambda tok: (-counts[tok], -len(tok), tok))


def is_feed_query(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in _FEED_PATTERNS)


_VALENCIAN_MARKERS = {
    "s'ha",
    "ha estat",
    "per a",
    "llocs",
    "lloc",
    "nomenament",
    "cessament",
    "errades",
    "correccio",
    "correcció",
    "publicat",
    "convocat",
    "convocada",
    "resolucio",
    "resolució",
    "decret",
    "generalitat",
    "consell",
    "conselleria",
}


def guess_language(text: str) -> str:
    if not text:
        return "es_es"
    lower = text.lower()
    if "·" in lower or "ç" in lower:
        return "va_va"
    if "l'" in lower:
        return "va_va"
    hits = sum(1 for marker in _VALENCIAN_MARKERS if marker in lower)
    if hits >= 2:
        return "va_va"
    return "es_es"


def _clean_phrase(text: str) -> str | None:
    phrase = text.strip().strip("\"'")
    if not phrase:
        return None
    phrase = re.sub(r"\s+", " ", phrase.lower()).strip()
    tokens = phrase.split()
    if len(tokens) < 2:
        return None
    if len(phrase) < 6:
        return None
    for token in tokens:
        if len(token) == 1 and not token.isdigit():
            return None
    return phrase


def build_bm25_query(question: str, intent: dict[str, Any]) -> str:
    keywords = intent.get("keywords") if isinstance(intent, dict) else None
    phrases = _extract_phrases(question, keywords if isinstance(keywords, list) else None)
    tokens = _tokenize(question)
    specific_terms: list[str] = []
    generic_terms: list[str] = []
    seen: set[str] = set()

    for phrase in phrases:
        quoted = f"\"{phrase}\""
        if quoted in seen:
            continue
        seen.add(quoted)
        specific_terms.append(quoted)

    for token in tokens:
        if not _keep_token(token):
            continue
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        if token in _GENERIC_TERMS:
            generic_terms.append(token)
        else:
            specific_terms.append(token)

    if isinstance(keywords, list):
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            token = keyword.lower().strip()
            if " " in token:
                continue
            if not _keep_token(token):
                continue
            if token in _STOPWORDS:
                continue
            if token in seen:
                continue
            seen.add(token)
            if token in _GENERIC_TERMS:
                generic_terms.append(token)
            else:
                specific_terms.append(token)

    terms: list[str] = []
    terms.extend(specific_terms)
    generic_terms = _prioritize_generic_terms(generic_terms)
    max_terms = 8
    max_generic = 2 if specific_terms else 4
    added_generic = 0
    for token in generic_terms:
        if token in terms:
            continue
        terms.append(token)
        added_generic += 1
        if added_generic >= max_generic or len(terms) >= max_terms:
            break
    if not terms:
        return question
    return " OR ".join(terms[:max_terms])


def build_bm25_queries(question: str, intent: dict[str, Any]) -> tuple[str, str | None]:
    broad = build_bm25_query(question, intent)
    keywords = intent.get("keywords") if isinstance(intent, dict) else None

    phrases = _extract_phrases(question, keywords if isinstance(keywords, list) else None)
    phrase_tokens = {
        tok
        for phrase in phrases
        for tok in phrase.split()
        if tok not in _CONNECTORS
    }
    anchor_terms = _select_anchor_terms(
        question,
        keywords if isinstance(keywords, list) else None,
        phrase_tokens=phrase_tokens if phrase_tokens else None,
    )
    phrase_token_ranked = _rank_phrase_tokens(phrases)

    terms: list[str] = []
    if isinstance(keywords, list):
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            token = keyword.strip().lower()
            if " " in token:
                continue
            if not _keep_token(token):
                continue
            if token in _STOPWORDS or token in _GENERIC_TERMS:
                continue
            if token not in terms:
                terms.append(token)

    if not terms:
        for token in _tokenize(question):
            if not _keep_token(token):
                continue
            if token in _STOPWORDS or token in _GENERIC_TERMS:
                continue
            if token not in terms:
                terms.append(token)

    if not phrases:
        if len(anchor_terms) >= 2:
            strict_terms = anchor_terms[: min(3, len(anchor_terms))]
            strict = " ".join(strict_terms)
            if strict != broad:
                return broad, strict
        return broad, None

    strict = None
    anchor_and_terms = []
    if phrase_token_ranked:
        anchor_and_terms = phrase_token_ranked[:4]
    elif anchor_terms:
        anchor_and_terms = anchor_terms[:3]
    if len(anchor_and_terms) >= 2:
        strict = " ".join(anchor_and_terms)
    elif phrases:
        phrase_part = " OR ".join(f"\"{phrase}\"" for phrase in phrases[:2])
        if anchor_terms:
            term_part = " OR ".join(anchor_terms[:2])
            strict = f"({phrase_part}) ({term_part})"
        else:
            strict = f"({phrase_part})"

    if strict == broad:
        return broad, None
    return broad, strict


def build_prf_query(
    base_query: str,
    docs: list[dict[str, Any]],
    max_docs: int = 5,
    max_terms: int = 6,
    min_df: int = 2,
) -> str | None:
    if not base_query or not docs:
        return None
    base_tokens = {tok for tok in _tokenize(base_query) if tok}
    term_freq: dict[str, int] = {}
    doc_freq: dict[str, int] = {}
    for doc in docs[:max_docs]:
        text = f"{doc.get('title') or ''} {doc.get('text') or ''}"
        tokens = [tok for tok in _tokenize(text) if _keep_token(tok)]
        tokens = [tok for tok in tokens if tok not in _STOPWORDS and tok not in _GENERIC_TERMS]
        if not tokens:
            continue
        unique = set(tokens)
        for tok in unique:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
        for tok in tokens:
            term_freq[tok] = term_freq.get(tok, 0) + 1
    candidates = [
        tok
        for tok in term_freq
        if tok not in base_tokens and doc_freq.get(tok, 0) >= min_df
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda t: (doc_freq.get(t, 0), term_freq.get(t, 0)), reverse=True)
    extra = candidates[:max_terms]
    if not extra:
        return None
    return base_query + " OR " + " OR ".join(extra)


FACET_SYSTEM = (
    "Divide la pregunta en subpreguntas atomicas y generales, "
    "sin asumir informacion del dominio. "
    "Devuelve SOLO JSON con campo facets (lista de strings). "
    "No devuelvas mas de {max_facets}."
)

FACET_USER = """Pregunta:
{question}

Devuelve facets como lista de subpreguntas, o lista vacia si no procede.
"""


def _fallback_facets(question: str, max_facets: int) -> list[str]:
    cleaned = " ".join((question or "").replace("¿", "").split())
    if not cleaned:
        return []
    parts = re.split(r"\?\s*|;\s+|\.\s+", cleaned)
    facets = []
    for part in parts:
        part = part.strip(" .")
        if not part:
            continue
        if len(part) < 6:
            continue
        if len(part.split()) < 2:
            continue
        facets.append(part)
        if len(facets) >= max_facets:
            break
    if len(facets) >= 2:
        return facets[:max_facets]
    return []


def decompose_question(question: str, max_facets: int = 3) -> list[str]:
    if not question or max_facets <= 0:
        return []
    lower = question.lower()
    has_split = (
        ("?" in question and question.count("?") > 1)
        or (";" in question)
        or (" y " in lower)
        or (" i " in lower)
    )
    if not has_split:
        return []
    return _fallback_facets(question, max_facets)
