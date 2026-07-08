from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.auto_ingest import (
    check_gap_source_publications,
    mark_gap_source_resolved,
    record_gap_source_failure,
)
from api.db import SessionLocal
from api.ingest_pipeline import run_pipeline

logger = logging.getLogger("dogv.retry_gap_source")


def _parse_languages(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _failures_table_exists(db) -> bool:
    exists = db.execute(
        sa_text("SELECT to_regclass('ingest_gap_source_failures')")
    ).scalar()
    return bool(exists)


def _load_due_rows(db, limit: int, languages: list[str]) -> list[dict]:
    query = """
    SELECT
        issue_date,
        language,
        attempts,
        last_error,
        last_checked_at,
        next_retry_at
    FROM ingest_gap_source_failures
    WHERE resolved_at IS NULL
      AND (next_retry_at IS NULL OR next_retry_at <= :now)
    """
    params: dict[str, object] = {
        "now": datetime.now(timezone.utc),
        "limit": int(limit),
    }
    if languages:
        query += "\n  AND language = ANY(:languages)"
        params["languages"] = languages
    query += "\nORDER BY COALESCE(next_retry_at, last_checked_at, now()) ASC\nLIMIT :limit"
    rows = db.execute(sa_text(query), params).mappings().all()
    return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--languages", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db = SessionLocal()
    try:
        if not _failures_table_exists(db):
            print(
                "Missing table ingest_gap_source_failures. "
                "Apply sql/2026-04-gap-source-failures.sql first."
            )
            return 2

        languages = _parse_languages(args.languages)
        rows = _load_due_rows(db, limit=max(1, args.limit), languages=languages)
        if not rows:
            print("No due gap source failures to retry.")
            return 0

        summary = {
            "checked": 0,
            "source_failed": 0,
            "resolved_no_docs": 0,
            "ingested": 0,
            "would_ingest": 0,
            "pipeline_failures": 0,
        }

        for row in rows:
            issue_date = row["issue_date"]
            language = row["language"]
            summary["checked"] += 1
            logger.info(
                "retry.check date=%s lang=%s attempts=%s",
                issue_date,
                language,
                row.get("attempts"),
            )

            has_docs = check_gap_source_publications(issue_date, language, db=db)
            if has_docs is None:
                summary["source_failed"] += 1
                continue

            if not has_docs:
                if args.dry_run:
                    logger.info(
                        "retry.no_docs date=%s lang=%s dry_run=true",
                        issue_date,
                        language,
                    )
                else:
                    mark_gap_source_resolved(db, issue_date, language)
                summary["resolved_no_docs"] += 1
                continue

            if args.dry_run:
                logger.info(
                    "retry.source_positive date=%s lang=%s action=would_ingest",
                    issue_date,
                    language,
                )
                summary["would_ingest"] += 1
                continue

            try:
                run_pipeline(issue_date, issue_date, [language])
                mark_gap_source_resolved(db, issue_date, language)
                summary["ingested"] += 1
                logger.info(
                    "retry.ingest_done date=%s lang=%s",
                    issue_date,
                    language,
                )
            except Exception as exc:
                summary["pipeline_failures"] += 1
                logger.exception(
                    "retry.ingest_failed date=%s lang=%s",
                    issue_date,
                    language,
                )
                record_gap_source_failure(
                    db,
                    issue_date,
                    language,
                    f"retry_pipeline_failed: {exc}",
                )

        print("Retry summary:")
        for key, value in summary.items():
            print(f"- {key}: {value}")

        if summary["source_failed"] or summary["pipeline_failures"]:
            return 1
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
