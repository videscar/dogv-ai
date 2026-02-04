from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal


def _load_eval_ids(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    doc_ids: set[int] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None and isinstance(item.get("doc_ids"), list) and item["doc_ids"]:
            doc_id = item["doc_ids"][0]
        if doc_id is None:
            continue
        try:
            doc_ids.add(int(doc_id))
        except (TypeError, ValueError):
            continue
    return sorted(doc_ids)


def _fetch_coverage(doc_ids: list[int]) -> list[dict[str, Any]]:
    if not doc_ids:
        return []
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

        meta_rows = db.execute(
            sa_text(
                """
                SELECT dd.id AS document_id, dd.title, di.date AS issue_date, di.language
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                WHERE dd.id = ANY(:doc_ids)
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
        meta = {
            int(row["document_id"]): {
                "title": (row.get("title") or "").strip(),
                "issue_date": row.get("issue_date"),
                "language": row.get("language"),
            }
            for row in meta_rows
        }

    results = []
    for doc_id in doc_ids:
        info = meta.get(doc_id, {})
        chunk_count = chunk_counts.get(doc_id, 0)
        has_chunks = chunk_count > 0
        has_title_embed = doc_id in title_embeds
        has_doc_embed = doc_id in doc_embeds
        results.append(
            {
                "document_id": doc_id,
                "issue_date": info.get("issue_date"),
                "language": info.get("language"),
                "title": info.get("title"),
                "chunk_count": chunk_count,
                "has_chunks": has_chunks,
                "has_title_embedding": has_title_embed,
                "has_doc_embedding": has_doc_embed,
                "missing_any": not (has_chunks and has_title_embed and has_doc_embed),
            }
        )
    return results


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No documents to audit.")
        return
    header = (
        "document_id",
        "issue_date",
        "language",
        "chunk_count",
        "title_embed",
        "doc_embed",
        "missing_any",
        "title",
    )
    print(" | ".join(header))
    print("-" * 120)
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["document_id"]),
                    str(row.get("issue_date") or ""),
                    str(row.get("language") or ""),
                    str(row.get("chunk_count") or 0),
                    "yes" if row.get("has_title_embedding") else "no",
                    "yes" if row.get("has_doc_embedding") else "no",
                    "yes" if row.get("missing_any") else "no",
                    (row.get("title") or "")[:80],
                ]
            )
        )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/eval_set_curated_last12m.json",
        help="Path to eval set JSON",
    )
    parser.add_argument("--output", help="Optional output file (csv or json)")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    doc_ids = _load_eval_ids(input_path)
    rows = _fetch_coverage(doc_ids)

    if args.format == "table":
        _print_table(rows)
    elif args.format == "csv":
        if not args.output:
            raise SystemExit("--output is required for csv format")
        _write_csv(rows, Path(args.output))
    elif args.format == "json":
        if not args.output:
            raise SystemExit("--output is required for json format")
        _write_json(rows, Path(args.output))

    missing = [row for row in rows if row.get("missing_any")]
    if missing:
        dates = sorted({row["issue_date"] for row in missing if row.get("issue_date")})
        if dates:
            start = dates[0]
            end = dates[-1]
            print(
                f"\nMissing coverage for {len(missing)} docs. "
                f"Suggested rebuild range: python scripts/build_chunks.py {start} {end} --force"
            )
        else:
            print(f"\nMissing coverage for {len(missing)} docs with unknown dates.")


if __name__ == "__main__":
    main()
