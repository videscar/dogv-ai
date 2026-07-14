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
            ExtractedIdentifier(
                id_kind="code", id_key=token.lower(), raw=token, source=source
            )
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
