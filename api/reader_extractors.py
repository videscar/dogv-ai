from __future__ import annotations

import re
from typing import Any

_STOPWORDS = {
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
    "convocatories",
    "convocatòria",
    "bases",
    "resolucio",
    "resolució",
    "resolucion",
    "decret",
    "decreto",
    "extracte",
    "extracto",
}

_ENTITY_CONNECTORS = {
    "de",
    "del",
    "d",
    "d'",
    "l",
    "l'",
    "la",
    "les",
    "el",
    "els",
    "i",
    "o",
    "e",
}


def _looks_like_list(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if "***" in text:
        return True
    if re.search(r"\b(nif|cve|dni)\b", lower):
        return True
    if re.search(r"\b[A-Z]{2,}\w*/\d{4}/\d{3,}\b", text):
        return True
    digits = sum(ch.isdigit() for ch in text)
    if digits >= 14 and digits / max(len(text), 1) > 0.12:
        return True
    if re.search(r"\b(annex|anexo|llistat|listado|relaci[oó]n)\b", lower) and digits >= 6:
        return True
    if len(re.findall(r"[A-ZÁÉÍÓÚÜÑ]{2,}", text)) >= 8 and digits >= 4:
        return True
    return False


def _extract_phrases(question: str, max_phrases: int = 8) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\"([^\"]+)\"|'([^']+)'", question):
        phrase = (match[0] or match[1]).strip().lower()
        phrase = re.sub(r"\s+", " ", phrase)
        if len(phrase) < 6:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= max_phrases:
            return phrases

    tokens = [t.lower() for t in re.findall(r"[\w·'-]+", question) if len(t) >= 2]
    for size in (4, 3, 2):
        for i in range(len(tokens) - size + 1):
            phrase_tokens = tokens[i : i + size]
            if not any(len(t) >= 4 and t not in _STOPWORDS for t in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            if len(phrase) < 8:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def _is_capitalized_token(token: str) -> bool:
    if not token:
        return False
    if re.match(r"^[dDlL]'[A-ZÀ-Ý]", token):
        return True
    return token[0].isupper()


def _extract_entities(question: str, max_entities: int = 6) -> list[str]:
    tokens = re.findall(r"[\w·'-]+", question)
    entities: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        lower = token.lower()
        if lower in _STOPWORDS or not _is_capitalized_token(token):
            i += 1
            continue
        phrase_tokens = [token]
        cap_count = 1
        i += 1
        while i < len(tokens):
            next_token = tokens[i]
            next_lower = next_token.lower()
            if _is_capitalized_token(next_token):
                phrase_tokens.append(next_token)
                cap_count += 1
                i += 1
                continue
            if next_lower in _ENTITY_CONNECTORS:
                phrase_tokens.append(next_token)
                i += 1
                continue
            break
        if cap_count < 2:
            continue
        entity = " ".join(phrase_tokens).strip().lower()
        entity = re.sub(r"\s+", " ", entity)
        if entity in seen:
            continue
        seen.add(entity)
        entities.append(entity)
        if len(entities) >= max_entities:
            break
    return entities


def _extract_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[\w·'-]+", question.lower())
    keywords = [t for t in tokens if len(t) >= 4 and t not in _STOPWORDS]
    seen = set()
    result = []
    for token in keywords:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _matched_keywords(text: str, keywords: list[str]) -> set[str]:
    if not text or not keywords:
        return set()
    lower = text.lower()
    return {k for k in keywords if k in lower}


def _score_text(text: str, keywords: list[str]) -> int:
    if not text or not keywords:
        return 0
    lower = text.lower()
    return sum(lower.count(k) for k in keywords)


def _fallback_evidence(
    question: str,
    docs: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    min_hits = 2 if len(keywords) >= 3 else 1
    candidates: list[tuple[int, int, str]] = []
    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        for chunk in doc.get("chunks") or []:
            matches = _matched_keywords(chunk, keywords)
            if len(matches) < min_hits:
                continue
            score = _score_text(chunk, list(matches))
            candidates.append((score, int(doc_id), chunk))

    if not candidates and full_docs:
        for doc in full_docs:
            doc_id = doc.get("document_id")
            if doc_id is None:
                continue
            text = doc.get("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                matches = _matched_keywords(line, keywords)
                if len(matches) < min_hits:
                    continue
                score = _score_text(line, list(matches))
                candidates.append((score, int(doc_id), line))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    best: dict[int, str] = {}
    for _, doc_id, text in candidates:
        if doc_id in best:
            continue
        best[doc_id] = text.strip()
        if len(best) >= 3:
            break

    return [
        {
            "doc_id": doc_id,
            "quote": quote[:800],
            "detail": "Extracto con coincidencias lexicas.",
        }
        for doc_id, quote in best.items()
    ]


def _numeric_evidence(
    question: str,
    docs: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not re.search(
        r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", question, re.IGNORECASE
    ):
        return []

    candidates: list[tuple[int, int, str]] = []
    # NB: was r"\b\\d[\\d\\.,]*\\b" — in a raw string \\d is a literal backslash+d, so
    # it matched only text containing backslashes (never, in DOGV text). The numeric
    # extractor silently returned nothing, leaving amount questions with no
    # deterministic figure backstop. This matches numbers like 207.000 / 1.366,74.
    amount_re = re.compile(r"\b\d[\d.,]*\b")

    def _score_amount(text: str) -> int:
        return len(amount_re.findall(text))

    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        for chunk in doc.get("chunks") or []:
            score = _score_amount(chunk)
            if score:
                candidates.append((score, int(doc_id), chunk))

    if not candidates and full_docs:
        for doc in full_docs:
            doc_id = doc.get("document_id")
            if doc_id is None:
                continue
            text = doc.get("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                score = _score_amount(line)
                if score:
                    candidates.append((score, int(doc_id), line))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    best: dict[int, str] = {}
    for _, doc_id, text in candidates:
        if doc_id in best:
            continue
        best[doc_id] = text.strip()
        if len(best) >= 2:
            break

    return [
        {
            "doc_id": doc_id,
            "quote": quote[:800],
            "detail": "Extracto con cantidades.",
        }
        for doc_id, quote in best.items()
    ]


# A program's *convocatoria/bases* declares its total budget/dotación; resoluciones
# de concesión/adjudicación declare amounts awarded to specific beneficiaries. When
# the question asks for a program-level total, the figure must come from the
# convocatoria/bases doc, not a concesión — a general DOGV distinction, not a
# per-question rule. These cues fire on intent ("import total/global/màxim",
# "dotació", "pressupost"...) and on the chunk that actually states such a total.
_PROGRAM_TOTAL_INTENT = re.compile(
    r"(import\w*\s+(?:total|global|m[aà]xim\w*)|importe\s+global|quantia\s+total|"
    r"dotaci[oó]\w*|dotad\w*|dotat\w*|pressupost\w*|presupuest\w*|"
    r"(?:total|global)\s+m[aà]xim\w*|m[aà]xim\w*\s+(?:total|global))",
    re.IGNORECASE,
)
_TOTAL_CUE = re.compile(
    r"(import\w*\s+(?:total|global|m[aà]xim\w*)|importe\s+global|quantia\s+total|"
    r"dotaci[oó]\w*|pressupost\w*|presupuest\w*|(?:total|global)\s+m[aà]xim\w*)",
    re.IGNORECASE,
)
_CONVOCATORIA_SUBKINDS = {"convocatoria", "convocatòria", "convocatories", "bases"}
_CONCESION_SUBKINDS = {"resultados", "resultats", "adjudicaciones", "adjudicacions"}
_CONCEDID_INTENT = re.compile(r"\b(concedid\w*|atorgad\w*|concedit\w*)\b", re.IGNORECASE)


def _program_total_evidence(
    question: str,
    docs: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pin the total/global/dotación figure from convocatoria/bases docs.

    Fires only when (a) the question asks for a program-level total AND (b) a
    Convocatoria/Bases-subkind doc is present in the window. Otherwise a no-op, so
    questions about individually-awarded amounts keep using the concesión doc.
    """
    if not _PROGRAM_TOTAL_INTENT.search(question or ""):
        return []
    amount_re = re.compile(r"\b\d[\d.,]*\b")
    # The awarded total ("subvencions concedides / atorgades") is declared in the
    # concesión (Resultados/Adjudicaciones), not the convocatoria. For that intent
    # scan ONLY those subkinds — a convocatoria sibling's own "import total" line
    # must not out-score it. Convocatoria-total questions are left untouched.
    if _CONCEDID_INTENT.search(question or ""):
        allowed_subkinds = _CONCESION_SUBKINDS
    else:
        allowed_subkinds = _CONVOCATORIA_SUBKINDS
    candidates: list[tuple[int, int, str]] = []

    def _scan(items: list[dict[str, Any]], get_text):
        n = len(items or [])
        for pos, d in enumerate(items or []):
            subkind = (d.get("doc_subkind") or "").strip().lower()
            if subkind not in allowed_subkinds:
                continue
            doc_id = d.get("document_id")
            if doc_id is None:
                continue
            for chunk in get_text(d):
                if not chunk or not _TOTAL_CUE.search(chunk):
                    continue
                if not amount_re.search(chunk):
                    continue
                cue = len(_TOTAL_CUE.findall(chunk)) + (
                    1 if "€" in chunk or "euro" in chunk.lower() else 0
                )
                # Prefer the total from the best-RANKED doc — the reranker already
                # decided which doc the question is about; cue density only breaks
                # ties within a doc. Without this, a figure-dense sibling convocatoria
                # out-scores the gold (v2-007: política-lingüística doc 83683 cue=10
                # beat the rank-1 GV-Talent doc 86084 cue=5, pinning 600k not 1.6M).
                score = (n - pos) * 100 + cue
                candidates.append((score, int(doc_id), chunk))

    _scan(docs, lambda d: d.get("chunks") or [])
    if not candidates and full_docs:
        _scan(full_docs, lambda d: (d.get("text") or "").splitlines())

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    best: dict[int, str] = {}
    for _, doc_id, text in candidates:
        if doc_id in best:
            continue
        best[doc_id] = text.strip()
        if len(best) >= 2:
            break
    return [
        {
            "doc_id": doc_id,
            "quote": quote[:800],
            "detail": "Importe total/global de la convocatoria o bases.",
        }
        for doc_id, quote in best.items()
    ]


def _eligibility_evidence(
    question: str,
    docs: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    if not re.search(
        r"\b(qui|qu[ií]en|beneficiar|beneficiari|beneficiario|sol·licit|solicitar|requisit|requisito|destinatari|destinatario|pot|puede|poden|pueden)\b",
        question,
        re.IGNORECASE,
    ):
        return []

    eligibility_terms = re.compile(
        r"\b(beneficiar|beneficiari|beneficiario|sol·licit|solicitud|requisit|requisito|destinatari|destinatario|titular|persona fisica|persona jurídica|podran|podrán|pueden|podran)\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str]] = []

    def _score(text: str) -> int:
        return len(eligibility_terms.findall(text))

    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        for chunk in doc.get("chunks") or []:
            score = _score(chunk)
            if score > 0:
                candidates.append((score, int(doc_id), chunk))

    if not candidates and full_docs:
        for doc in full_docs:
            doc_id = doc.get("document_id")
            if doc_id is None:
                continue
            text = doc.get("text") or ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                score = _score(line)
                if score > 0:
                    candidates.append((score, int(doc_id), line))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    best: dict[int, str] = {}
    for _, doc_id, text in candidates:
        if doc_id in best:
            continue
        best[doc_id] = text.strip()
        if len(best) >= max_items:
            break

    return [
        {
            "doc_id": doc_id,
            "quote": quote[:800],
            "detail": "Extracto de requisitos/beneficiarios.",
        }
        for doc_id, quote in best.items()
    ]


def _clean_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for item in items:
        doc_id = item.get("doc_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        detail = (item.get("detail") or "").strip()
        cleaned.append({"doc_id": int(doc_id), "quote": quote, "detail": detail})
    return cleaned
