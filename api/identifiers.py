"""Structured-identifier extraction for the identifier layer.

The retrieval pipeline matches through lexical BM25 and semantic embeddings, and
both are structurally blind to a structured identifier (a code like
``GACUJIMA/2025/36`` is one atomic tsvector token; embeddings cannot tell twin
extracts apart). This module extracts those identifiers at ingest into the
``doc_identifier`` table (see sql/2026-07-doc-identifier.sql) so the query side
can look them up exactly (see the query lane, step 4).

Classes extracted here:
- ``code``: letter-prefixed slash expedient/project codes (``GACUJIMA/2025/36``,
  ``ERESAR/2026/39R07/0008``) and compact body project codes (``24I636``).
- ``bdns``: BDNS subsidy database IDs (``BDNS 895054``), body-only.
- ``norm``: the document's OWN norm identity (``DECRETO 74/2026``), read off the
  start of its title — this is what norm-pin resolves today by title ILIKE;
  storing it lets step 4 migrate norm-pin onto an exact lookup.

Deliberately NOT extracted:
- ``person`` — names are lexically matchable and already retrieve correctly
  (probes IP06/07/08 pass); pinning on common names carries the highest
  false-pin risk, so names stay lexical-only.
- ``ref`` — the DOGV publication ref is already a first-class, uniquely-indexed
  column on ``dogv_documents``; the query lane reads that column directly rather
  than duplicating it here.

Normalization is centralized (``normalize_code`` / ``normalize_norm_key``) so the
ingest side and the query-side detector produce identical keys — that identity
is the whole point of the exact-match lane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .doc_references import _self_identity
from .dogv_resolver import parse_references

# Title + the first slice of the body carry the high-precision identifiers; deep
# body text is boilerplate/annex noise. Matches doc_references._BODY_SCAN_CHARS.
_BODY_SCAN_CHARS = 3000

# Letter-prefixed slash code: a >=3-letter acronym followed by 1..4 "/group"
# segments (GACUJIMA/2025/36, ERESAR/2026/39R07/0008). Precision guard applied
# after the match: at least one segment must be a 4-digit year, which rejects
# ordinary prose slashes ("y/o", "km/h", "and/or") that lack a year.
_SLASH_CODE_RE = re.compile(r"\b([A-Za-z]{3,}(?:/[A-Za-z0-9]+){1,4})\b")
_YEAR_SEGMENT_RE = re.compile(r"^(?:19|20)\d{2}$")

# Compact body project ("I" = investigación) code: 2-digit year + 'I' + 3-4
# digits ("codi 25I656", "24I636"). Restricted to the 'I' letter on purpose:
# the broader \d{2}[A-Za-z]\d+ shape also matches budget-line codes (08R09,
# 28E050) which are not document identifiers and pollute the class.
_COMPACT_CODE_RE = re.compile(r"\b(\d{2}[Ii]\d{3,4})\b")

# BDNS subsidy database identifier ("BDNS 895054", "BDNS (Identif.): 895054").
_BDNS_RE = re.compile(r"\bBDNS\b[^0-9]{0,20}?(\d{5,7})\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractedIdentifier:
    id_kind: str  # 'code' | 'bdns' | 'norm'
    id_key: str  # normalized lookup key
    raw: str  # as it appeared in the text
    source: str  # 'title' | 'body'

    @property
    def dedup_key(self) -> tuple[str, str]:
        return (self.id_kind, self.id_key)


def normalize_code(raw: str) -> str:
    """Canonical key for a slash/compact code: lowercase, separators collapsed to
    a single '/'. The query side normalizes spaced/dashed renderings
    ('gacujima 2025 36', 'GACUJIMA-2025-36') to this same key."""
    parts = [p for p in re.split(r"[\s/\-]+", raw.strip()) if p]
    return "/".join(parts).lower()


def normalize_norm_key(tipo: str, numero: int, anyo: int) -> str:
    """Canonical key for a numbered norm-ref: '<tipo>/<numero>/<anyo>' lowercased.
    tipo is a canonical tipo key from dogv_resolver._TIPO_TITLE_PATTERNS."""
    return f"{tipo}/{numero}/{anyo}".lower()


# Any slash-joined alphanumeric run (GACUJIMA/2025/36, 2024/302/03, 74/2026).
_SLASH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+")


def code_token_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of structured-code tokens in ``text``: slash-joined
    alphanumeric runs that are letter-prefixed (GACUJIMA/2025/36) or have 3+
    groups (a compound expedient/ref like 2024/302/03). A 4-digit year embedded
    in such a token is part of the code, not a standalone date — callers use
    these spans to stop a code-year from seeding a date filter (see
    intent._infer_year_range).

    A plain two-number N/YYYY (Ley 39/2015, Decreto 74/2026) is deliberately NOT
    a code span: its year is the norm's real year and stays available as a date
    signal. This is the single source of truth for "where are the code tokens";
    modules that need it import this rather than re-implementing the regex.
    """
    spans: list[tuple[int, int]] = []
    for match in _SLASH_TOKEN_RE.finditer(text):
        token = match.group(0)
        if any(char.isalpha() for char in token) or token.count("/") >= 2:
            spans.append(match.span())
    return spans


def _slash_codes(text: str, source: str) -> list[ExtractedIdentifier]:
    out: list[ExtractedIdentifier] = []
    for m in _SLASH_CODE_RE.finditer(text):
        token = m.group(1)
        segments = token.split("/")
        # Precision guard: a real expedient code carries a 4-digit year segment.
        if not any(_YEAR_SEGMENT_RE.match(seg) for seg in segments):
            continue
        # Reject a plain two-number norm-ref (39/2015) that happens to have a
        # letter word glued by the tokenizer — those are handled by the norm
        # class, not code. A code has a genuine letter-only lead segment.
        if not re.match(r"^[A-Za-z]{3,}$", segments[0]):
            continue
        out.append(
            ExtractedIdentifier(
                id_kind="code", id_key=normalize_code(token), raw=token, source=source
            )
        )
    return out


def _compact_codes(text: str, source: str) -> list[ExtractedIdentifier]:
    out: list[ExtractedIdentifier] = []
    for m in _COMPACT_CODE_RE.finditer(text):
        token = m.group(1)
        out.append(
            ExtractedIdentifier(id_kind="code", id_key=token.lower(), raw=token, source=source)
        )
    return out


def _bdns_ids(text: str, source: str) -> list[ExtractedIdentifier]:
    out: list[ExtractedIdentifier] = []
    for m in _BDNS_RE.finditer(text):
        digits = m.group(1)
        out.append(
            ExtractedIdentifier(
                id_kind="bdns", id_key=digits, raw=m.group(0).strip(), source=source
            )
        )
    return out


def _norm_identity(title: str) -> list[ExtractedIdentifier]:
    """The document's own numbered norm identity (DECRETO 74/2026), read off the
    start of its title. Date-only Resoluciones (no numero) are not pinned here."""
    self_id = _self_identity(title or "")
    if self_id.tipo is None or self_id.numero is None or self_id.anyo is None:
        return []
    key = normalize_norm_key(self_id.tipo, self_id.numero, self_id.anyo)
    raw = f"{self_id.tipo} {self_id.numero}/{self_id.anyo}"
    return [ExtractedIdentifier(id_kind="norm", id_key=key, raw=raw, source="title")]


def extract_doc_identifiers(title: str, body: str | None) -> list[ExtractedIdentifier]:
    """All identifiers found in ``title`` (highest precision) then the first
    ``_BODY_SCAN_CHARS`` of ``body``. Deduplicated on (id_kind, id_key),
    keeping the first (title-sourced) occurrence."""
    title = title or ""
    ids: list[ExtractedIdentifier] = []

    ids.extend(_slash_codes(title, "title"))
    ids.extend(_compact_codes(title, "title"))
    ids.extend(_norm_identity(title))

    if body:
        snippet = body[:_BODY_SCAN_CHARS]
        ids.extend(_slash_codes(snippet, "body"))
        ids.extend(_compact_codes(snippet, "body"))
        ids.extend(_bdns_ids(snippet, "body"))

    seen: set[tuple[str, str]] = set()
    deduped: list[ExtractedIdentifier] = []
    for ident in ids:
        if ident.dedup_key in seen:
            continue
        seen.add(ident.dedup_key)
        deduped.append(ident)
    return deduped


# ---------------------------------------------------------------------------
# Query side: detect identifiers in a user question so the retrieval pin lane
# can look them up exactly against doc_identifier / the ref column.
# ---------------------------------------------------------------------------

# A maximal run of alnum groups joined by spaces/slashes/dashes. Candidate code
# windows are carved out of this run below — starting at each >=3-letter acronym
# so an acronym buried after leading words ("beca gacujima 2025 36") is still a
# candidate start. Deliberately liberal: precision comes from the exact match
# against doc_identifier, not this regex.
_LOOSE_CODE_RE = re.compile(r"\b([A-Za-z0-9]+(?:[\s/\-]+[A-Za-z0-9]+){1,10})\b")

# A real expedient code is an acronym + at most this many trailing groups
# (ERESAR/2026/39R07/0008 = acronym + 3); bounds the candidate windows.
_MAX_CODE_GROUPS = 5

# DOGV publication ref written year-first ("2026/4148"), matched against the
# indexed ref column. Distinct from a norm-ref (year-last, "74/2026").
_REF_COLUMN_RE = re.compile(r"\b(20\d{2}/\d{3,5})\b")


@dataclass(frozen=True)
class QueryIdentifier:
    id_kind: str  # 'code' | 'bdns' | 'norm' | 'ref'
    id_key: str


def _code_candidate_keys(question: str) -> set[str]:
    """Candidate code keys from a question. Within each loose run, a window is
    opened at every >=3-letter acronym and extended up to _MAX_CODE_GROUPS
    trailing groups; any window containing a 4-digit year is emitted. This yields
    the real code (gacujima/2025/36) even when it is buried after leading words
    ('beca gacujima 2025 36 uji') or trailed by junk — the windows that do not
    correspond to a real code simply match no row in doc_identifier."""
    keys: set[str] = set()
    for m in _LOOSE_CODE_RE.finditer(question):
        parts = [p for p in re.split(r"[\s/\-]+", m.group(1).strip()) if p]
        n = len(parts)
        for i in range(n):
            if not re.fullmatch(r"[A-Za-z]{3,}", parts[i]):
                continue  # a code must start with a >=3-letter acronym
            for j in range(i + 1, min(n, i + 1 + _MAX_CODE_GROUPS)):
                window = parts[i : j + 1]
                if any(_YEAR_SEGMENT_RE.fullmatch(seg) for seg in window):
                    keys.add("/".join(window).lower())
    for m in _COMPACT_CODE_RE.finditer(question):
        keys.add(m.group(1).lower())
    return keys


def detect_query_identifiers(question: str) -> list[QueryIdentifier]:
    """All structured identifiers mentioned in ``question``, normalized to the
    same keys the ingest side stores, for exact lookup by the pin lane."""
    if not question:
        return []
    out: list[QueryIdentifier] = []
    seen: set[tuple[str, str]] = set()

    def _add(kind: str, key: str) -> None:
        if key and (kind, key) not in seen:
            seen.add((kind, key))
            out.append(QueryIdentifier(kind, key))

    for key in _code_candidate_keys(question):
        _add("code", key)
    for m in _BDNS_RE.finditer(question):
        _add("bdns", m.group(1))
    for ref in parse_references(question):
        if ref.tipo and ref.numero is not None and ref.anyo is not None:
            _add("norm", normalize_norm_key(ref.tipo, ref.numero, ref.anyo))
    for m in _REF_COLUMN_RE.finditer(question):
        _add("ref", m.group(1))
    return out
