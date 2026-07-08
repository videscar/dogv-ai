"""Enumeration-aware candidate augmentation.

Some questions ask the assistant to *list every* disposition matching a structured
filter — "cítame las disposiciones de ofertas de empleo A1 publicadas en mayo de
2026". Pure semantic retrieval is the wrong tool for an exhaustive list: it caps at
k and lets near-identical siblings crowd each other out (tester #30: gold ORDEN
24/2026 never entered the candidate pool because its siblings ORDEN 23/26 did). For
these queries we run a deterministic SQL pull over the stated month + category and
merge the results into the candidate pool, so the reranker/reader can see the whole
series instead of whichever sibling retrieval happened to surface.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date
import re
import unicodedata

from sqlalchemy import text as sa_text

from .dogv_resolver import _STOP, _strip_accents

# Month token (es + va) -> month number.
_MONTHS: dict[str, int] = {
    "enero": 1, "gener": 1,
    "febrero": 2, "febrer": 2,
    "marzo": 3, "marc": 3, "març": 3,
    "abril": 4,
    "mayo": 5, "maig": 5,
    "junio": 6, "juny": 6,
    "julio": 7, "juliol": 7,
    "agosto": 8, "agost": 8,
    "septiembre": 9, "setiembre": 9, "setembre": 9,
    "octubre": 10,
    "noviembre": 11, "novembre": 11,
    "diciembre": 12, "desembre": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
# "mayo de 2026", "maig de 2026", "mayo 2026".
_MONTH_YEAR_RE = re.compile(
    rf"\b({_MONTH_ALT})\b\s+(?:de\s+|d['’]\s*)?(\d{{4}})\b", re.IGNORECASE
)
# Group/subgroup codes ("A1", "grupo C2", "subgrupo A1").
_GROUP_CODE_RE = re.compile(r"\b([ABCDE][12])\b")
# Cues that a question wants a list of dispositions, not a single one.
_ENUM_CUE_RE = re.compile(
    r"\b(disposiciones|disposicions|c[ií]ta(?:me|m)?|c[ií]tame|cites|citeu|"
    r"enumera|lista(?:do)?|llista(?:t)?|rela[cs]i[oó]n\s+de|todas\s+las|totes\s+les|"
    r"qu[eé]\s+\w+\s+(?:se\s+)?(?:han\s+)?public)\b",
    re.IGNORECASE,
)


@dataclass
class EnumerationSpec:
    date_start: date
    date_end: date
    group_codes: list[str]
    content_terms: list[str] = field(default_factory=list)


def is_enumeration_query(question: str) -> bool:
    """True when the question asks to list dispositions of a stated month + category."""
    return parse_enumeration(question) is not None


def parse_enumeration(question: str) -> EnumerationSpec | None:
    """Parse an enumeration query into a month range + category, or None.

    Conservative: requires an explicit month+year AND an enumeration cue, so it
    never fires on an ordinary single-disposition question."""
    if not question:
        return None
    if not _ENUM_CUE_RE.search(question):
        return None
    m = _MONTH_YEAR_RE.search(question)
    if not m:
        return None
    month = _MONTHS.get(_strip_accents(m.group(1).lower()))
    if month is None:
        month = _MONTHS.get(m.group(1).lower())
    if month is None:
        return None
    year = int(m.group(2))
    if year < 2000 or year > 2100:
        return None
    last = calendar.monthrange(year, month)[1]
    group_codes = []
    for code in _GROUP_CODE_RE.findall(question):
        if code.upper() not in group_codes:
            group_codes.append(code.upper())
    content_terms = _content_terms(question)
    if not group_codes and not content_terms:
        return None
    return EnumerationSpec(
        date_start=date(year, month, 1),
        date_end=date(year, month, last),
        group_codes=group_codes,
        content_terms=content_terms,
    )


def _content_terms(question: str) -> list[str]:
    """Topical content words (>=4, not stopwords/months) for ranking the pull."""
    terms: list[str] = []
    for tok in re.findall(r"[\wáéíóúüçñ'-]+", question.lower()):
        if len(tok) < 4 or tok in _STOP:
            continue
        if tok.isdigit() or tok in _MONTHS or _strip_accents(tok) in _MONTHS:
            continue
        if tok not in terms:
            terms.append(tok)
    return terms


def augment_enumeration_candidates(
    db,
    spec: EnumerationSpec,
    language: str,
    existing_ids: set[int],
    cap: int,
) -> list[dict]:
    """Documents in the stated month + language matching the category, ranked by
    content-term overlap, excluding ones already in the candidate pool.

    Recall-oriented: a group code (when present) is a hard filter; the LLM rerank +
    answer node still decide which of these to actually cite."""
    if cap <= 0:
        return []
    clauses = ["i.date >= :start", "i.date <= :end", "i.language = :lang"]
    params: dict = {
        "start": spec.date_start,
        "end": spec.date_end,
        "lang": language,
    }
    if spec.group_codes:
        ors = []
        for j, code in enumerate(spec.group_codes):
            ors.append(f"d.title ILIKE :g{j}")
            params[f"g{j}"] = f"%{code}%"
        clauses.append("(" + " OR ".join(ors) + ")")
    elif spec.content_terms:
        ors = []
        for j, term in enumerate(spec.content_terms):
            ors.append(f"d.title ILIKE :c{j}")
            params[f"c{j}"] = f"%{term}%"
        clauses.append("(" + " OR ".join(ors) + ")")
    rows = db.execute(
        sa_text(
            "SELECT d.id, d.title, d.ref, i.date "
            "FROM dogv_documents d JOIN dogv_issues i ON d.issue_id = i.id "
            f"WHERE {' AND '.join(clauses)}"
        ),
        params,
    ).all()

    terms_norm = [_strip_accents(t) for t in spec.content_terms]
    scored: list[tuple[int, int, dict]] = []
    for doc_id, title, ref, issue_date in rows:
        if int(doc_id) in existing_ids:
            continue
        hay = _strip_accents((title or "").lower())
        overlap = sum(1 for t in terms_norm if t in hay)
        scored.append(
            (
                overlap,
                int(doc_id),
                {
                    "document_id": int(doc_id),
                    "title": title,
                    "ref": ref,
                    "issue_date": issue_date.isoformat() if issue_date else None,
                    "rrf_score": 0.0,
                    "enumeration_augmented": True,
                },
            )
        )
    # Highest content overlap first; stable by id for determinism.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for _, _, item in scored[:cap]]
