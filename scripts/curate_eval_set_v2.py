from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.query_expansion import is_relative_time_query


@dataclass
class DocCoverage:
    has_chunks: bool
    has_title_embedding: bool
    has_doc_embedding: bool


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _doc_id(entry: dict[str, Any]) -> int | None:
    if "doc_id" in entry and entry["doc_id"] is not None:
        try:
            return int(entry["doc_id"])
        except (TypeError, ValueError):
            return None
    if isinstance(entry.get("doc_ids"), list) and entry["doc_ids"]:
        try:
            return int(entry["doc_ids"][0])
        except (TypeError, ValueError):
            return None
    return None


def _fetch_coverage(doc_ids: list[int]) -> dict[int, DocCoverage]:
    if not doc_ids:
        return {}
    with SessionLocal() as db:
        chunk_rows = db.execute(
            sa_text(
                """
                SELECT document_id, COUNT(*) AS chunk_count
                FROM rag_chunk
                WHERE document_id = ANY(:doc_ids)
                GROUP BY document_id
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
        chunk_counts = {int(row["document_id"]): int(row["chunk_count"]) for row in chunk_rows}

        title_rows = db.execute(
            sa_text(
                """
                SELECT document_id
                FROM rag_title
                WHERE document_id = ANY(:doc_ids)
                AND embedding IS NOT NULL
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
        title_embeds = {int(row["document_id"]) for row in title_rows}

        doc_rows = db.execute(
            sa_text(
                """
                SELECT document_id
                FROM rag_doc
                WHERE document_id = ANY(:doc_ids)
                AND embedding IS NOT NULL
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
        doc_embeds = {int(row["document_id"]) for row in doc_rows}

    coverage: dict[int, DocCoverage] = {}
    for doc_id in doc_ids:
        coverage[doc_id] = DocCoverage(
            has_chunks=chunk_counts.get(doc_id, 0) > 0,
            has_title_embedding=doc_id in title_embeds,
            has_doc_embedding=doc_id in doc_embeds,
        )
    return coverage


def _group_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("doc_kind") or "Otros"),
        str(entry.get("doc_subkind") or "SinSubtipo"),
        str(entry.get("language") or "es_es"),
    )


def _round_robin(groups: dict[tuple[str, str, str], list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    buckets = {key: deque(items) for key, items in groups.items() if items}
    order = deque(sorted(buckets.keys()))
    selected: list[dict[str, Any]] = []
    while buckets and len(selected) < limit:
        key = order[0]
        order.rotate(-1)
        queue = buckets.get(key)
        if not queue:
            buckets.pop(key, None)
            order = deque([k for k in order if k in buckets])
            continue
        selected.append(queue.popleft())
        if not queue:
            buckets.pop(key, None)
            order = deque([k for k in order if k in buckets])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/eval_set_curated_last12m.json")
    parser.add_argument("--candidate", default="data/eval_set_candidate_v2.json")
    parser.add_argument("--output", default="data/eval_set_curated_last12m_v2.json")
    parser.add_argument("--max-size", type=int, default=100)
    parser.add_argument("--require-embeddings", action="store_true")
    parser.add_argument("--exclude-temporal", action="store_true")
    args = parser.parse_args()

    base = _load_json(Path(args.base))
    candidate = _load_json(Path(args.candidate))

    base_by_doc: dict[int, dict[str, Any]] = {}
    for entry in base:
        if args.exclude_temporal and is_relative_time_query(entry.get("question", "")):
            continue
        doc_id = _doc_id(entry)
        if doc_id is None:
            continue
        base_by_doc[doc_id] = dict(entry)

    candidate_by_doc: dict[int, dict[str, Any]] = {}
    for entry in candidate:
        if args.exclude_temporal and is_relative_time_query(entry.get("question", "")):
            continue
        doc_id = _doc_id(entry)
        if doc_id is None:
            continue
        if doc_id in base_by_doc:
            continue
        candidate_by_doc[doc_id] = dict(entry)

    coverage = _fetch_coverage(list(candidate_by_doc.keys()))
    filtered_candidates: list[dict[str, Any]] = []
    for doc_id, entry in candidate_by_doc.items():
        cov = coverage.get(doc_id)
        if not cov or not cov.has_chunks:
            continue
        if args.require_embeddings and not (cov.has_title_embedding and cov.has_doc_embedding):
            continue
        filtered_candidates.append(entry)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in filtered_candidates:
        groups[_group_key(entry)].append(entry)

    for items in groups.values():
        items.sort(key=lambda item: str(item.get("issue_date") or ""), reverse=True)

    max_size = max(len(base_by_doc), args.max_size)
    remaining = max_size - len(base_by_doc)
    selected_candidates = _round_robin(groups, remaining)

    merged = list(base_by_doc.values()) + selected_candidates
    for idx, entry in enumerate(merged, start=1):
        entry["id"] = idx

    Path(args.output).write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Wrote {len(merged)} entries to {args.output}")


if __name__ == "__main__":
    main()
