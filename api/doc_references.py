"""Extract explicit norm cross-references from a DOGV document's title/body.

DOGV documents routinely name the norm(s) they relate to in prose: "por la que
se resuelve la convocatoria aprobada por la Orden 12/2026...", "es modifica el
Decret 185/2018...", "Mediante la Resolución de 12 de marzo de 2026 ... se
convocaron...". This module finds those references so they can be persisted
in `doc_reference` (see sql/2026-07-doc-references.sql) and used at read time
to pull the referenced companion document into the read set.

Reuses the tipo-word / N-YYYY patterns from dogv_resolver.py rather than
duplicating them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sqlalchemy import text as sa_text

from .dogv_resolver import (
    _DISP_DATE_RE,
    _NUMBER_YEAR_RE,
    _TIPO_PREFIXES,
    _TIPO_TITLE_PATTERNS,
    _normalize_year,
)

logger = logging.getLogger("dogv.doc_references")

# How far before a matched N/YYYY (or a date) a tipo word may sit and still be
# considered "governing" it.
_TIPO_WINDOW = 40
# How much text around a match to scan for the verb that classifies the
# relationship (ref_kind).
_KIND_WINDOW = 120
# Only the title + the first slice of the body carry high-precision refs; deep
# body text is boilerplate/annex noise.
_BODY_SCAN_CHARS = 3000

_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("modifica", re.compile(r"modificaci[oó]|modifica[nr]?\b", re.I)),
    ("deroga", re.compile(r"derog[ao]|derogaci[oó]n", re.I)),
    ("resuelve", re.compile(r"\bresuel[vt]e|\bresoluci[oó]n de\b.*\bconvocatoria\b", re.I)),
    ("convoca", re.compile(r"convoca(?:toria|n|ron|do)?\b|aprobada por|aprovada per", re.I)),
    (
        "corrige",
        re.compile(r"correcci[oó]n de errores|correcci[oó] d'?errades|correcci[oó]|corrige", re.I),
    ),
]

_VA_TITLE_TIPO_RE = re.compile(r"^\s*(?:DECRET\b|LLEI\b|ORDRE\b|RESOLUCI[OÓ]\b|ACORD\b)", re.I)


@dataclass(frozen=True)
class ExtractedReference:
    ref_tipo: str
    ref_numero: int | None
    ref_anyo: int | None
    ref_kind: str
    raw_text: str
    disp_day: int | None = None
    disp_month: str | None = None
    disp_year: int | None = None
    disp_organismo: str | None = None

    @property
    def ref_key(self) -> str:
        """Always-non-null identity of the referenced norm, for dedup (see
        ref_key on the doc_reference table — Postgres UNIQUE treats NULL
        columns as always-distinct, which would break idempotent re-runs for
        the numero/anyo-less date-based Resolución references)."""
        if self.ref_numero is not None and self.ref_anyo is not None:
            return f"{self.ref_tipo}:{self.ref_numero}/{self.ref_anyo}"
        return f"{self.ref_tipo}:D{self.disp_day}-{self.disp_month}-{self.disp_year}"


def _classify_kind(text: str, start: int, end: int) -> str:
    window = text[max(0, start - _KIND_WINDOW) : end + _KIND_WINDOW]
    for kind, pat in _KIND_PATTERNS:
        if pat.search(window):
            return kind
    return "cita"


def _find_tipo_before(text: str, pos: int) -> str | None:
    """Which tipo word governs the token ending at `pos`? Checked within
    `_TIPO_WINDOW` chars before it (two-word forms first, per _TIPO_TITLE_PATTERNS
    ordering, so "decreto ley" isn't swallowed by plain "decreto")."""
    window = text[max(0, pos - _TIPO_WINDOW) : pos]
    for tipo, pat in _TIPO_TITLE_PATTERNS:
        if re.search(pat, window, re.I):
            return tipo
    return None


@dataclass(frozen=True)
class _SelfIdentity:
    numero: int | None
    anyo: int | None
    disp_day: int | None
    disp_month: str | None
    disp_year: int | None = None


def _self_identity(title: str) -> _SelfIdentity:
    """The document's own identifier — its N/YYYY (e.g. "ORDEN 12/2026, de
    ...") and/or its disposition date (e.g. "RESOLUCIÓN de 12 de marzo de
    2026...") — both read off the *start* of its own title. A doc's own
    identifier reappearing anywhere in its title/body (very common: titles
    are echoed verbatim at the top of the body) must never be emitted as a
    reference to another document."""
    numero = anyo = None
    m = re.match(
        r"^\s*(?:" + "|".join(p for _, p in _TIPO_TITLE_PATTERNS) + r")\s+(\d{1,4})/(\d{2,4})",
        title,
        re.I,
    )
    if m:
        numero, anyo = int(m.group(1)), _normalize_year(m.group(2))

    day = month = disp_year = None
    dm = re.match(
        r"^\s*(?:RESOLUCI[OÓ]N|RESOLUCI[OÓ])\s+de\s+(\d{1,2})\s+de\s+([a-záéíóúç]+)\s+de\s+(\d{4})",
        title,
        re.I,
    )
    if dm:
        day, month, disp_year = int(dm.group(1)), dm.group(2).lower(), int(dm.group(3))

    return _SelfIdentity(
        numero=numero, anyo=anyo, disp_day=day, disp_month=month, disp_year=disp_year
    )


def _extract_num_year_refs(text: str, self_id: _SelfIdentity) -> list[ExtractedReference]:
    out: list[ExtractedReference] = []
    for m in _NUMBER_YEAR_RE.finditer(text):
        tipo = _find_tipo_before(text, m.start())
        if tipo is None:
            continue
        numero = int(m.group(1))
        anyo = _normalize_year(m.group(2))
        if anyo < 1980 or anyo > 2100:
            continue
        if numero == self_id.numero and anyo == self_id.anyo:
            continue
        kind = _classify_kind(text, m.start(), m.end())
        raw = text[max(0, m.start() - _TIPO_WINDOW) : m.end()].strip()
        out.append(
            ExtractedReference(
                ref_tipo=tipo, ref_numero=numero, ref_anyo=anyo, ref_kind=kind, raw_text=raw
            )
        )
    return out


def _extract_resolucion_date_refs(text: str, self_id: _SelfIdentity) -> list[ExtractedReference]:
    """Resoluciones are frequently unnumbered and referenced by disposition
    date instead ("Resolución de 12 de marzo de 2026, de la ..."). Only fires
    when a "resolucion"-tipo word immediately precedes the date (<=20 chars)
    so we don't pick up unrelated dates in the body."""
    out: list[ExtractedReference] = []
    for m in _DISP_DATE_RE.finditer(text):
        tipo_window = text[max(0, m.start() - 20) : m.start()]
        if not re.search(r"RESOLUCI[OÓ]N|RESOLUCI[OÓ]", tipo_window, re.I):
            continue
        day = int(m.group(1))
        month = m.group(2).lower()
        year: int | None = None
        tail_end = m.end() + 10
        ym = re.match(r"\s*de\s+(\d{4})", text[m.end() : tail_end])
        if ym:
            year = int(ym.group(1))
            tail_end = m.end() + ym.end()
        organismo = None
        om = re.match(
            r",?\s*de\s+(?:la\s+|los\s+|las\s+)?([A-ZÁÉÍÓÚÑÇ][^,()]{4,60})",
            text[tail_end : tail_end + 90],
        )
        if om:
            organismo = om.group(1).strip()
        if day == self_id.disp_day and month == self_id.disp_month and year == self_id.disp_year:
            continue
        kind = _classify_kind(text, m.start(), m.end())
        raw = text[max(0, m.start() - 20) : m.end() + 10].strip()
        out.append(
            ExtractedReference(
                ref_tipo="resolucion",
                ref_numero=None,
                ref_anyo=None,
                ref_kind=kind,
                raw_text=raw,
                disp_day=day,
                disp_month=month,
                disp_year=year,
                disp_organismo=organismo,
            )
        )
    return out


def extract_references(title: str, body: str | None) -> list[ExtractedReference]:
    """All references found in `title` (highest precision) then the first
    `_BODY_SCAN_CHARS` of `body`. Self-references (the doc's own identifier,
    which titles/bodies commonly repeat) are skipped."""
    title = title or ""
    self_id = _self_identity(title)

    refs: list[ExtractedReference] = []
    refs.extend(_extract_num_year_refs(title, self_id))
    refs.extend(_extract_resolucion_date_refs(title, self_id))

    if body:
        snippet = body[:_BODY_SCAN_CHARS]
        refs.extend(_extract_num_year_refs(snippet, self_id))
        refs.extend(_extract_resolucion_date_refs(snippet, self_id))

    # Dedup on the identity that matches the DB unique constraint, keep first.
    seen: set[tuple] = set()
    deduped: list[ExtractedReference] = []
    for r in refs:
        key = (r.ref_key, r.ref_kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def _topic_overlap_pick(rows: list[tuple], source_title: str) -> int | None:
    """Among still-ambiguous candidates, prefer the one whose title shares the
    most >=5-char content words with the source document's own title (a
    convocatoria and its resolución de concesión share distinctive nouns, e.g.
    "premios"/"fallas"). Requires a strict winner."""
    src_words = {w for w in re.findall(r"[a-záéíóúüñ]{5,}", source_title.lower())}
    if not src_words:
        return None
    scored = []
    for doc_id, title in rows:
        words = set(re.findall(r"[a-záéíóúüñ]{5,}", (title or "").lower()))
        scored.append((doc_id, len(words & src_words)))
    scored.sort(key=lambda x: x[1], reverse=True)
    if len(scored) >= 2 and scored[0][1] > scored[1][1] and scored[0][1] > 0:
        return scored[0][0]
    if len(scored) == 1 and scored[0][1] > 0:
        return scored[0][0]
    return None


def resolve_target_document_id(
    db, source_document_id: int, ref: ExtractedReference, source_title: str = ""
) -> int | None:
    """Find the single in-corpus document this reference points at, or None if
    it isn't in corpus or is ambiguous (e.g. an Orden number that repeats
    across consellerias within a year — better to leave NULL than guess)."""
    prefixes = _TIPO_PREFIXES.get(ref.ref_tipo, [])
    if not prefixes:
        return None

    # upper(title) LIKE upper(pattern) (not ILIKE) so Postgres can use the
    # upper(title) text_pattern_ops index (sql/2026-07-doc-references.sql)
    # instead of a seq scan over the whole corpus per reference.
    if ref.ref_numero is not None and ref.ref_anyo is not None:
        like = " OR ".join(f"upper(title) LIKE upper(:p{i})" for i in range(len(prefixes)))
        params = {f"p{i}": f"{p} {ref.ref_numero}/{ref.ref_anyo}%" for i, p in enumerate(prefixes)}
    elif ref.disp_day is not None and ref.disp_month is not None and ref.disp_year is not None:
        like = " OR ".join(f"upper(title) LIKE upper(:p{i})" for i in range(len(prefixes)))
        params = {
            f"p{i}": f"{p} de {ref.disp_day} de {ref.disp_month} de {ref.disp_year}%"
            for i, p in enumerate(prefixes)
        }
    else:
        return None

    params["src"] = source_document_id
    rows = db.execute(
        sa_text(
            f"SELECT id, conselleria FROM dogv_documents WHERE id != :src AND ({like}) LIMIT 200"
        ),
        params,
    ).all()
    ids = {r[0] for r in rows}
    if len(ids) == 1:
        return ids.pop()
    if len(ids) > 1 and ref.disp_organismo:
        org = ref.disp_organismo.lower()
        narrowed_rows = [r for r in rows if r[1] and org in r[1].lower()]
        narrowed = {r[0] for r in narrowed_rows}
        if len(narrowed) == 1:
            return narrowed.pop()
        if len(narrowed) > 1 and source_title:
            title_rows = db.execute(
                sa_text("SELECT id, title FROM dogv_documents WHERE id = ANY(:ids)"),
                {"ids": list(narrowed)},
            ).all()
            picked = _topic_overlap_pick(title_rows, source_title)
            if picked is not None:
                return picked
    return None
