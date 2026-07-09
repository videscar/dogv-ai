"""RC1 — sibling-edition recency suppression.

The DOGV re-publishes near-identical documents across issues: annual convocatòries,
recurring nomenaments, successive subvention rounds. Retrieval surfaces the whole
family, and the reader — fed all of them — can answer from a stale year (e.g. a 2025
appointment when the user means the current one). This module detects such families by
document-embedding near-duplication *across different issue dates* and keeps only the
newest edition of each in the read set.

The signal is doc-embedding cosine, not title text: sibling editions often reword their
titles (different underlying selective process, different annex) yet stay tight in
embedding space, while two distinct same-day publications that happen to share a long
title boilerplate are protected by the different-date rule.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from .rerank import parse_issue_date


def _find(parent: dict[int, int], x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def group_editions(pairs: Iterable[tuple[int, int]]) -> list[set[int]]:
    """Union-find the sibling pairs into edition families. Only ids that appear in at
    least one pair are grouped; singletons are not returned."""
    parent: dict[int, int] = {}
    for a, b in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, set[int]] = {}
    for node in parent:
        groups.setdefault(_find(parent, node), set()).add(node)
    return [g for g in groups.values() if len(g) > 1]


def stale_edition_ids(
    sibling_pairs: Iterable[tuple[int, int]],
    issue_date_by_doc: dict[int, date | None],
) -> set[int]:
    """Doc ids that are older editions of a detected family — i.e. every family member
    whose issue date is strictly before the family's newest issue date. Members sharing
    the newest date are all kept (concurrent, distinct). Members with an unknown date are
    never marked stale (we cannot prove they are older)."""
    stale: set[int] = set()
    for group in group_editions(sibling_pairs):
        dates = [issue_date_by_doc.get(d) for d in group]
        known = [d for d in dates if d is not None]
        if not known:
            continue
        newest = max(known)
        for doc_id in group:
            d = issue_date_by_doc.get(doc_id)
            if d is not None and d < newest:
                stale.add(doc_id)
    return stale


def edition_sibling_pairs(
    db,
    doc_ids: list[int],
    issue_date_by_doc: dict[int, date | None],
    sim_threshold: float,
) -> list[tuple[int, int]]:
    """Pairs of candidate docs whose document embeddings are near-duplicate
    (cosine >= sim_threshold) AND that carry different issue dates. The different-date
    filter is what keeps distinct same-day publications (which can share a title
    boilerplate and rank very high in similarity) from being treated as editions."""
    if len(doc_ids) < 2:
        return []
    from sqlalchemy import text as sa_text

    rows = (
        db.execute(
            sa_text(
                """
            SELECT a.document_id AS a, b.document_id AS b,
                   1 - (a.embedding <=> b.embedding) AS cos
            FROM rag_doc a
            JOIN rag_doc b ON a.document_id < b.document_id
            WHERE a.document_id = ANY(:ids)
              AND b.document_id = ANY(:ids)
              AND a.embedding IS NOT NULL
              AND b.embedding IS NOT NULL
            """
            ),
            {"ids": doc_ids},
        )
        .mappings()
        .all()
    )
    pairs: list[tuple[int, int]] = []
    for row in rows:
        if float(row["cos"]) < sim_threshold:
            continue
        a, b = int(row["a"]), int(row["b"])
        da, dbb = issue_date_by_doc.get(a), issue_date_by_doc.get(b)
        if da is None or dbb is None or da == dbb:
            continue
        pairs.append((a, b))
    return pairs


def suppress_stale_editions(
    db,
    ordered_doc_ids: list[int],
    candidates: list[dict[str, Any]],
    *,
    sim_threshold: float,
    scan_n: int,
    protected_ids: Iterable[int] = (),
) -> tuple[list[int], set[int]]:
    """Return (kept_doc_ids, dropped_ids). Scans the top `scan_n` of `ordered_doc_ids`,
    detects edition families, and drops older siblings — except any in `protected_ids`
    (e.g. a norm-pinned disposition). Order of the kept ids is preserved."""
    if len(ordered_doc_ids) < 2:
        return ordered_doc_ids, set()
    issue_date_by_doc: dict[int, date | None] = {}
    for item in candidates:
        did = item.get("document_id")
        if did is not None:
            issue_date_by_doc[int(did)] = parse_issue_date(item.get("issue_date"))
    scan_ids = [int(d) for d in ordered_doc_ids[:scan_n]]
    pairs = edition_sibling_pairs(db, scan_ids, issue_date_by_doc, sim_threshold)
    if not pairs:
        return ordered_doc_ids, set()
    stale = stale_edition_ids(pairs, issue_date_by_doc)
    protected = {int(x) for x in protected_ids}
    stale -= protected
    if not stale:
        return ordered_doc_ids, set()
    kept = [int(d) for d in ordered_doc_ids if int(d) not in stale]
    return kept, stale
