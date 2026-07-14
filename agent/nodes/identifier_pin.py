"""Exact-match identifier pin lane (identifier layer, query side).

Structured identifiers (codes like GACUJIMA/2025/36, BDNS subsidy IDs, the DOGV
publication ref) are invisible to lexical BM25 (one atomic token) and to
embeddings (twin extracts are near-identical). When a question names one, this
lane looks it up EXACTLY against the pre-extracted doc_identifier table (or the
ref column), and pins the matched documents into the read set — bypassing the
whole ranking pool, which cannot separate the twins, and bypassing the intent
date filter (an exact identifier is unambiguous regardless of date).

This runs before the gated second hop (agent/nodes/second_hop.py) and feeds the
same read-set pin (norm_pin_doc_ids) and RC1-protection (second_hop_protect_ids)
channels. Guarded by a match-count cap so a non-discriminating key (a family
prefix that somehow resolves to many docs) never pins.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sa_text

from agent.nodes.retrieve_pool import PoolResult
from api.db import SessionLocal
from api.identifiers import detect_query_identifiers

logger = logging.getLogger("dogv.graph")

# An identifier key that resolves to more than this many documents is not
# discriminating (a real code maps to the es+va editions of one act — 2 docs);
# above the cap we decline to pin rather than flood the read set.
_MATCH_CAP = 6

# Only precise, copy-verbatim machine identifiers are pinned. Norm-refs are
# deliberately excluded: a question's cited "Decret 167/2025" is often a human
# error (the eval v2-042 question attributes DANA rental aid to a decree that is
# actually about appointments), so an exact norm pin collides with premise
# correction. norm-refs stay on the existing title-retrieval / norm-pin path;
# folding them onto this lane is a later, separately eval-gated step.
_PINNABLE_KINDS = frozenset({"code", "bdns", "ref"})

_DOC_ROW_SQL = sa_text(
    """
    SELECT dd.id AS document_id, dd.title, dd.ref, dd.type, dd.doc_kind,
           dd.doc_subkind, dd.pdf_url, dd.html_url, di.date AS issue_date
    FROM dogv_documents dd
    JOIN dogv_issues di ON di.id = dd.issue_id
    WHERE dd.id = ANY(:ids)
    ORDER BY di.date DESC, dd.id ASC
    """
)

_IDENTIFIER_SQL = sa_text(
    "SELECT document_id FROM doc_identifier WHERE id_kind = :kind AND id_key = :key"
)
_REF_SQL = sa_text("SELECT id FROM dogv_documents WHERE ref = :key")


def _resolve_ids(db, kind: str, key: str) -> list[int]:
    stmt = _REF_SQL if kind == "ref" else _IDENTIFIER_SQL
    params = {"key": key} if kind == "ref" else {"kind": kind, "key": key}
    return [int(row[0]) for row in db.execute(stmt, params).all()]


def apply_identifier_pins(
    question: str, pool: PoolResult, request_id: str | None = None
) -> tuple[PoolResult, list[int], list[int], dict[str, Any] | None]:
    """Pin documents whose exact structured identifier is named in `question`.

    Returns (pool, pin_ids, protect_ids, profile). pin_ids go into the read-set
    pin (norm_pin_doc_ids); protect_ids shield the pinned docs from RC1 edition
    suppression (an exact identifier match is the target, never a stale sibling).
    profile is None when nothing pinned (no extra work signalled)."""
    identifiers = [qi for qi in detect_query_identifiers(question) if qi.id_kind in _PINNABLE_KINDS]
    if not identifiers:
        return pool, [], [], None

    existing = {int(d["document_id"]) for d in pool.fused}
    fired: list[str] = []
    matched_ids: list[int] = []
    with SessionLocal() as db:
        for qi in identifiers:
            ids = _resolve_ids(db, qi.id_kind, qi.id_key)
            if not ids or len(ids) > _MATCH_CAP:
                continue
            fired.append(f"{qi.id_kind}:{qi.id_key}")
            matched_ids.extend(ids)
        matched_ids = list(dict.fromkeys(matched_ids))  # dedup, keep order
        if not matched_ids:
            return pool, [], [], None
        rows = db.execute(_DOC_ROW_SQL, {"ids": matched_ids}).mappings().all()

    # Pin in the deterministic SQL order (date DESC, id ASC) — not the
    # match/detect order — so the read set is reproducible for es/va edition
    # pairs that share a date.
    pinned: list[int] = []
    for row in rows:
        doc = dict(row)
        doc_id = int(doc["document_id"])
        if doc_id not in existing:
            doc["rrf_score"] = 0.0
            pool.fused.append(doc)
            existing.add(doc_id)
        # Pin + protect whether newly added or already in the pool: pool
        # membership alone does not guarantee the doc reaches the reader (the
        # ranking that placed it there cannot separate the twins).
        if doc_id not in pinned:
            pinned.append(doc_id)

    if not pinned:
        return pool, [], [], None
    logger.info(
        "retrieve.identifier_pin req=%s fired=%s pinned=%s", request_id, fired, pinned
    )
    return pool, pinned, pinned, {"fired": True, "identifiers": fired, "pinned": pinned}
