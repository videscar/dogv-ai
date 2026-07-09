"""Resolve a natural-language disposition reference to a DOGV document.

Used by the on-demand historical-fetch path: when a query explicitly names a
disposition (e.g. "Decreto 185/2018") that is not in the corpus, we resolve it
against the DOGV portal's full-text search, recover the disposition's internal
id + publication date, and hand that date to the normal date-range ingest path.

The portal search (verified 2026-06-22):
  POST {base}/dogv-portal/dogv/search?lang=&page=0&size=10&sort=fecha,desc
       body {"texto": "<query>", "seccion": []}
  -> {totalElements, content:[{id, titulo, seccion, organismo, ...}]}
Notes:
- size>=10 is mandatory; size=3 triggers a server-side BigDecimal crash.
- Ley/Decreto refs are ~unique per year (top-1 is gold). Orden numbers repeat
  across consellerias within a year, so we rank by the question's topic terms.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("dogv.resolver")

# tipo (normalized) -> regex alternation matching how it appears in a DOGV title
# (Spanish + Valencian forms). Order matters: check the two-word forms first.
_TIPO_TITLE_PATTERNS: list[tuple[str, str]] = [
    ("decreto ley", r"DECRETO[\s-]*LEY|DECRET[\s-]*LLEI"),
    ("ley", r"LEY|LLEI"),
    # plain decreto must NOT swallow "DECRETO LEY"
    ("decreto", r"DECRET(?:O)?(?!\s*(?:LEY|LLEI))"),
    ("orden", r"ORDEN|ORDRE"),
    ("resolucion", r"RESOLUCI[OÓ]N|RESOLUCI[OÓ]"),
    ("acuerdo", r"ACUERDO|ACORD"),
]

# Spoken/written tipo word in a *question* -> normalized tipo key above.
_TIPO_QUERY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdecreto[\s-]*ley\b|\bdecret[\s-]*llei\b", re.I), "decreto ley"),
    (re.compile(r"\bley(?:es)?\b|\bllei(?:s)?\b", re.I), "ley"),
    (re.compile(r"\bdecreto(?:s)?\b|\bdecret(?:s)?\b", re.I), "decreto"),
    (re.compile(r"\borden(?:es)?\b|\bordre(?:s)?\b", re.I), "orden"),
    (re.compile(r"\bresoluci[oó]n(?:es)?\b|\bresoluci[oó](?:ns)?\b", re.I), "resolucion"),
    (re.compile(r"\bacuerdo(?:s)?\b|\bacord(?:s)?\b", re.I), "acuerdo"),
]

_NUMBER_YEAR_RE = re.compile(r"\b(\d{1,4})/(\d{2,4})\b")

# tipo -> SQL ILIKE title prefixes (es + va), for checking corpus presence.
_TIPO_PREFIXES: dict[str, list[str]] = {
    "ley": ["LEY", "LLEI"],
    "decreto": ["DECRETO", "DECRET"],
    "decreto ley": ["DECRETO LEY", "DECRET LLEI"],
    "orden": ["ORDEN", "ORDRE"],
    "resolucion": ["RESOLUCIÓN", "RESOLUCIÓ", "RESOLUCION"],
    "acuerdo": ["ACUERDO", "ACORD"],
}

# Words that carry no topical signal when disambiguating (Orden case).
_STOP = {
    "que",
    "dice",
    "sobre",
    "cual",
    "cuál",
    "como",
    "cómo",
    "para",
    "del",
    "las",
    "los",
    "una",
    "uno",
    "qué",
    "regula",
    "establece",
    "dispone",
    "trata",
    "contenido",
    "objeto",
    "principal",
    "menciona",
    "mencionan",
    "fecha",
    "firmó",
    "ayudas",
    "medidas",
    "the",
    "and",
    "materia",
    "comunitat",
    "valenciana",
    "generalitat",
    "consell",
    "conselleria",
    "articulo",
    "artículo",
}


@dataclass
class Reference:
    """A disposition reference parsed out of a user question."""

    tipo: str | None  # normalized tipo key, e.g. "decreto" (may be None)
    numero: int
    anyo: int  # 4-digit year
    topic_terms: list[str] = field(default_factory=list)
    raw: str = ""
    date_day: int | None = None  # disposition day, when the question states one
    date_month: str | None = None  # disposition month token (lowercased, es or va)

    @property
    def num_year(self) -> str:
        return f"{self.numero}/{self.anyo}"

    def search_text(self) -> str:
        """Bare reference query. The DOGV search is AND-semantics, so topic terms
        that don't appear verbatim in the target title would return 0 hits — they
        are used only for ranking (see _topic_score), never in the query itself."""
        tipo_word = self.tipo.split()[0] if self.tipo else ""
        return f"{tipo_word} {self.num_year}".strip()


@dataclass
class ResolvedDoc:
    disposicion_id: int
    titulo: str
    fecha_publicacion: date
    fecha_disposicion: date | None
    seccion: str | None
    organismo: str | None
    score: float = 0.0


def _normalize_year(raw_year: str) -> int:
    if len(raw_year) == 4:
        return int(raw_year)
    # 2-digit year: assume 20xx for <= current decade-ish, else 19xx.
    y = int(raw_year)
    return 2000 + y if y < 80 else 1900 + y


def parse_reference(question: str) -> Reference | None:
    """Extract a single explicit disposition reference, or None.

    Conservative: requires an ``N/YYYY`` token. The tipo is inferred from a
    norm-word in the question when present; topic terms are the remaining
    content words (used to disambiguate Orden, whose numbers repeat).
    """
    if not question:
        return None
    m = _NUMBER_YEAR_RE.search(question)
    if not m:
        return None
    numero = int(m.group(1))
    anyo = _normalize_year(m.group(2))
    if anyo < 1980 or anyo > 2100:
        return None

    tipo: str | None = None
    # Prefer a tipo word that sits near the number; fall back to any in the text.
    window = question[max(0, m.start() - 30) : m.end() + 5]
    for rx, key in _TIPO_QUERY_MAP:
        if rx.search(window):
            tipo = key
            break
    if tipo is None:
        for rx, key in _TIPO_QUERY_MAP:
            if rx.search(question):
                tipo = key
                break

    day, month = _parse_disposition_date(question)
    return Reference(
        tipo=tipo,
        numero=numero,
        anyo=anyo,
        topic_terms=_topic_terms_of(question),
        raw=m.group(0),
        date_day=day,
        date_month=month,
    )


# "de 30 de octubre" / "de 9 de gener" — the disposition date that disambiguates
# norms whose N/YYYY repeats across consellerias (each numbers independently).
_DISP_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(?:de\s+|d['’])\s*"
    r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|"
    r"gener|febrer|mar[çc]|maig|juny|juliol|agost|setembre|octubre|novembre|desembre)\b",
    re.IGNORECASE,
)


def _parse_disposition_date(question: str) -> tuple[int | None, str | None]:
    """Day + month token of the disposition date stated in the question, or
    (None, None). Used to confirm a same-numbered norm in the corpus is actually
    the one asked for (the N/YYYY alone is not unique)."""
    m = _DISP_DATE_RE.search(question or "")
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).lower()


def _topic_terms_of(question: str) -> list[str]:
    """Content words of a question (>=4 chars, not stopwords, not the tipo word or
    a number/year) used to disambiguate which disposition a question is about."""
    topic_terms: list[str] = []
    for tok in re.findall(r"[\wáéíóúüçñ·'-]+", question.lower()):
        if len(tok) < 4 or tok in _STOP:
            continue
        if _NUMBER_YEAR_RE.search(tok) or tok.isdigit():
            continue
        if any(rx.search(tok) for rx, _ in _TIPO_QUERY_MAP):
            continue
        if tok not in topic_terms:
            topic_terms.append(tok)
    return topic_terms


# A primary disposition is a self-standing norm (vs a resolución/anuncio/corrección
# that merely cites one). Only these are valid "norm-target" principals.
_PRIMARY_TIPOS = {"ley", "decreto", "decreto ley", "orden"}

# A tipo word introduced by an article ("la ley", "el decreto", "del decret",
# "de la orden") or an interrogative determiner ("qué decreto", "quina llei") ->
# the question is ABOUT that norm, not just mentioning the word in passing.
_NAMED_NORM_RE = re.compile(
    r"\b(?:la|el|las|los|del|de\s+la|de\s+l['’]|d['’]|una?|aquesta?|aquell[ao]?|"
    r"qu[eé]|quin(?:a|s|es)?)\s+"
    r"(decreto[\s-]*ley|decret[\s-]*llei|decreto|decret|ley|llei|orden|ordre)\b",
    re.IGNORECASE,
)


@dataclass
class NamedNormTarget:
    """A norm referenced by type + topic but WITHOUT a number/year
    (e.g. "la Ley de la Función Pública Valenciana"). Lets us recover the
    principal citation when retrieval found the norm but synthesis cited a doc
    that only mentions it."""

    tipo: str  # normalized primary tipo (in _PRIMARY_TIPOS)
    topic_terms: list[str] = field(default_factory=list)
    raw: str = ""


def parse_named_norm_target(question: str) -> NamedNormTarget | None:
    """Detect a no-number "type + topic" norm target, or None.

    Deliberately conservative: only fires when there is NO N/YYYY token (those go
    through parse_reference), an article-introduced primary tipo word is present,
    and the question carries topic terms to match a title against."""
    if not question or _NUMBER_YEAR_RE.search(question):
        return None
    m = _NAMED_NORM_RE.search(question)
    if not m:
        return None
    tipo = None
    for rx, key in _TIPO_QUERY_MAP:
        if rx.search(m.group(1)):
            tipo = key
            break
    if tipo not in _PRIMARY_TIPOS:
        return None
    topic_terms = _topic_terms_of(question)
    if not topic_terms:
        return None
    return NamedNormTarget(tipo=tipo, topic_terms=topic_terms, raw=m.group(0))


def title_primary_tipo(title: str) -> str | None:
    """Normalized tipo if `title` is a primary norm (in _PRIMARY_TIPOS), else None.
    Classified by the leading tipo word, with decreto-ley checked before decreto."""
    if not title:
        return None
    for key, pat in _TIPO_TITLE_PATTERNS:
        if re.match(rf"^\s*(?:{pat})\b", title, re.I):
            return key if key in _PRIMARY_TIPOS else None
    return None


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def named_target_topic_overlap(target: NamedNormTarget, title: str) -> tuple[int, float]:
    """(# of the target's topic terms found in `title`, coverage ratio)."""
    if not target.topic_terms:
        return (0, 0.0)
    hay = _strip_accents(title.lower())
    matched = sum(1 for t in target.topic_terms if _strip_accents(t) in hay)
    return (matched, matched / len(target.topic_terms))


def title_num_year(title: str) -> str | None:
    """The N/YYYY token in a title (to tell es/va twins of one norm from distinct
    norms when several titles match a named target)."""
    m = _NUMBER_YEAR_RE.search(title or "")
    return m.group(0) if m else None


def _safe_page_size(size: int) -> int:
    """The portal computes total/size as a BigDecimal and crashes (HTTP 440,
    'Non-terminating decimal expansion') when the ratio doesn't terminate. Only
    2/5-smooth sizes (10, 20, 50, 100, ...) guarantee termination for every total.
    Snap up to the nearest safe value that is also >= the requested size."""
    for s in (10, 20, 50, 100, 200, 500, 1000):
        if s >= size:
            return s
    return 1000


def _query_lang(text: str) -> str:
    """Conservative language pick for the resolver. The corpus ingest pulls both
    languages for a date regardless, so this only decides which language's titles
    the search returns; default to Spanish (the gold-dominant query language) and
    switch to Valencian only on unambiguous markers. guess_language() is too eager
    on shared tokens like 'del'/'Consell' and would search the wrong index."""
    low = (text or "").lower()
    if "·" in low or "ç" in low or "l'" in low:
        return "va_va"
    return "es_es"


def search_dogv(
    texto: str, lang: str, *, size: int = 50, timeout: int | None = None
) -> list[dict[str, Any]]:
    """POST the DOGV full-text search; return the `content` list (may be empty)."""
    base = settings.dogv_base_url.rstrip("/")
    url = f"{base}/dogv-portal/dogv/search"
    params = {"lang": lang, "page": 0, "size": _safe_page_size(size), "sort": "fecha,desc"}
    body = {"texto": texto, "seccion": []}
    to = timeout if timeout is not None else getattr(settings, "request_timeout_seconds", 20)
    resp = requests.post(url, params=params, json=body, timeout=to)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return []
    return data.get("content") or []


def _title_matches_ref(titulo: str, ref: Reference) -> bool:
    if not titulo:
        return False
    if ref.num_year not in titulo:
        return False
    if ref.tipo:
        for key, pat in _TIPO_TITLE_PATTERNS:
            if key == ref.tipo:
                return bool(re.match(rf"^\s*(?:{pat})\b", titulo, re.I))
    return True


_VA_TITLE_RE = re.compile(r"^\s*(?:DECRET\b|LLEI\b|ORDRE\b|RESOLUCI[OÓ]\b|ACORD\b)", re.I)


def _title_language(titulo: str) -> str:
    """Guess a DOGV title's language from its leading tipo word (es vs va)."""
    return "va" if _VA_TITLE_RE.match(titulo or "") else "es"


def _topic_score(hit: dict[str, Any], ref: Reference) -> float:
    if not ref.topic_terms:
        return 0.0
    organismo = hit.get("organismo") or ""
    hay = f"{hit.get('titulo') or ''} {organismo}".lower()
    return float(sum(1 for t in ref.topic_terms if t in hay))


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    s = str(value)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def resolve(ref: Reference, lang: str | None = None) -> ResolvedDoc | None:
    """Resolve a Reference to a DOGV document, or None if not confidently found."""
    if ref is None:
        return None
    search_lang = lang or "es_es"
    try:
        # Bare reference; size large enough to hold all same-number dispositions
        # of the year (Orden numbers repeat across consellerias).
        hits = search_dogv(ref.search_text(), search_lang, size=50)
        exact = [h for h in hits if _title_matches_ref(h.get("titulo") or "", ref)]
        if not exact:
            # Fallback: drop the tipo word (handles tipo-less questions / odd titles).
            hits = search_dogv(ref.num_year, search_lang, size=50)
            exact = [h for h in hits if _title_matches_ref(h.get("titulo") or "", ref)]
    except Exception:
        logger.exception("resolver.search_failed ref=%s", ref.num_year)
        return None

    pool = exact
    if not pool:
        return None

    if len(pool) == 1:
        best = pool[0]
    else:
        # Rank by: topic-term overlap (Orden disambiguation), then a title in the
        # requested language (search returns both es+va), then original order.
        want_lang = "va" if str(search_lang).startswith("va") else "es"
        ordered = sorted(
            enumerate(pool),
            key=lambda iv: (
                -_topic_score(iv[1], ref),
                0 if _title_language(iv[1].get("titulo") or "") == want_lang else 1,
                iv[0],
            ),
        )
        best = ordered[0][1]

    disp_id = best.get("id")
    if disp_id is None:
        return None

    # Recover the authoritative publication date (the issue-date to ingest).
    fecha_pub = _parse_date(best.get("fechaPublicacion") or best.get("fechaDogv"))
    fecha_disp = _parse_date(best.get("fechaDisposicion"))
    titulo = (best.get("titulo") or "").strip()
    if fecha_pub is None:
        try:
            from scripts.sumario_ingest import fetch_disposicion_json

            detail = fetch_disposicion_json(disp_id, search_lang)
            fecha_pub = _parse_date(detail.get("fechaPublicacion") or detail.get("fechaDogv"))
            fecha_disp = fecha_disp or _parse_date(detail.get("fechaDisposicion"))
            titulo = titulo or (detail.get("titulo") or "").strip()
        except Exception:
            logger.exception("resolver.detail_failed id=%s", disp_id)

    if fecha_pub is None:
        return None

    seccion = best.get("seccion")
    if isinstance(seccion, dict):
        seccion = seccion.get("descripcion")

    return ResolvedDoc(
        disposicion_id=int(disp_id),
        titulo=titulo,
        fecha_publicacion=fecha_pub,
        fecha_disposicion=fecha_disp,
        seccion=seccion if isinstance(seccion, str) else None,
        organismo=best.get("organismo"),
        score=_topic_score(best, ref),
    )


def reference_matches_title(ref: Reference, titulo: str) -> bool:
    """Public: does this title carry the referenced tipo + number/year?"""
    return _title_matches_ref(titulo or "", ref)


def corpus_like_patterns(ref: Reference) -> list[str]:
    """SQL ILIKE patterns that match this disposition's title in dogv_documents.
    Used to decide whether the referenced norm is already in the corpus (a plain
    'DECRETO 3/2020%' prefix won't match 'DECRETO LEY 3/2020...')."""
    prefixes = _TIPO_PREFIXES.get(ref.tipo or "", [])
    if prefixes:
        return [f"{p} {ref.num_year}%" for p in prefixes]
    return [f"% {ref.num_year}%"]


def resolve_question(question: str) -> ResolvedDoc | None:
    """Convenience: parse + resolve in one call. None if no explicit reference."""
    ref = parse_reference(question)
    if ref is None:
        return None
    return resolve(ref, _query_lang(question))


_REF_IN_TITLE_RE = re.compile(r"\b\d{1,4}/\d{2,4}\b")
# How far after a "Ley N/YYYY" token to look for the law's name. DOGV titles read
# "Ley 1/2022, de 13 de abril, de la Generalitat, de Transparencia y Buen Gobierno"
# — the topic name sits within ~90 chars of the number.
_PRINCIPAL_NAME_WINDOW = 90


def _infer_principal_ref(titles: list[str], tipo: str, topic_terms: list[str]) -> Reference | None:
    """Pick the principal N/YYYY of `tipo` that the corpus titles name with the
    query's topic.

    A foundational law that is itself out of the corpus window is still *named* by
    the in-window norms that modify or develop it (e.g. "LEY 4/2024 ... de
    modificación de la Ley 1/2022, de 13 de abril, ... de Transparencia y Buen
    Gobierno"). A DOGV title always states a law's name right after its number, so a
    topic term landing in the window just after a number is strong evidence that
    that number is the principal asked about. We score every `tipo`-typed number by
    how many topic terms fall in its trailing window, summed across titles, and
    require the winner to dominate — otherwise we stay hands-off (a wrong principal
    would force-cite the wrong law)."""
    prefixes = _TIPO_PREFIXES.get(tipo or "")
    if not prefixes:
        return None
    tipo_ref_re = re.compile(rf"(?:{'|'.join(prefixes)})\s+(\d{{1,4}}/\d{{2,4}})", re.I)
    topic_norm = sorted({_strip_accents(t) for t in topic_terms if len(t) >= 4})
    if not topic_norm:
        return None

    # IDF over the gathered titles so a distinctive term ("transparencia") outweighs
    # a generic one ("publica", which appears in countless law names).
    n = max(len(titles), 1)
    df = {t: sum(1 for title in titles if t in _strip_accents(title.lower())) for t in topic_norm}
    weight = {t: math.log(1 + n / max(df[t], 1)) for t in topic_norm}

    # Score each number by its BEST trailing window (not a sum), so a number that
    # merely recurs with a generic term can't out-accumulate the real principal.
    best_score: dict[str, float] = {}
    freq: dict[str, int] = {}
    for title in titles:
        for m in tipo_ref_re.finditer(title):
            ref = m.group(1)
            window = title[m.end() : m.end() + _PRINCIPAL_NAME_WINDOW]
            nxt = _REF_IN_TITLE_RE.search(window)  # don't bleed into the next norm
            if nxt:
                window = window[: nxt.start()]
            wn = _strip_accents(window.lower())
            ws = sum(weight[t] for t in topic_norm if t in wn)
            if ws > 0:
                best_score[ref] = max(best_score.get(ref, 0.0), ws)
                freq[ref] = freq.get(ref, 0) + 1
    if not best_score:
        return None
    ranked = sorted(best_score, key=lambda r: (best_score[r], freq[r]), reverse=True)
    best = ranked[0]
    # The winner must be unambiguous: a lone candidate, or one that strictly beats
    # the runner-up's topic-proximity score.
    if len(ranked) > 1 and best_score[ranked[1]] >= best_score[best]:
        return None

    bm = _NUMBER_YEAR_RE.search(best)
    if bm is None:
        return None
    numero = int(bm.group(1))
    anyo = _normalize_year(bm.group(2))
    if anyo < 1980 or anyo > 2100:
        return None
    return Reference(tipo=tipo, numero=numero, anyo=anyo, topic_terms=topic_terms, raw=best)


def infer_reference_from_corpus(db, question: str) -> Reference | None:
    """Infer the explicit N/YYYY of a no-number "type + topic" norm target by
    reading how the corpus names it (see _infer_principal_ref), or None.

    Lets the on-demand fetch recover a foundational law asked for by name only
    ("la Ley de Transparencia" -> Ley 1/2022) when the law itself predates the
    window but in-window norms reference it. Conservative: needs an
    article-introduced primary tipo + topic terms, and a dominant principal."""
    target = parse_named_norm_target(question)
    if target is None:
        return None
    prefixes = _TIPO_PREFIXES.get(target.tipo or "")
    if not prefixes:
        return None
    terms = [t for t in target.topic_terms if len(t) >= 4]
    if not terms:
        return None
    like = " OR ".join(f"title ILIKE :t{i}" for i in range(len(terms)))
    params: dict = {f"t{i}": f"%{t}%" for i, t in enumerate(terms)}
    params["tre"] = rf"({'|'.join(prefixes)})\s+[0-9]+/[0-9]{{2,4}}"
    from sqlalchemy import text as _sa_text  # local import: keep module import-light

    rows = db.execute(
        _sa_text(f"SELECT title FROM dogv_documents WHERE title ~* :tre AND ({like}) LIMIT 120"),
        params,
    ).all()
    return _infer_principal_ref([r[0] for r in rows], target.tipo, target.topic_terms)
