"""
Maintain hot/warm DOGV indices.

Usage:
    python scripts/maintain_indices.py --bootstrap
    python scripts/maintain_indices.py --daily
"""

import sys
from datetime import date, timedelta

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import get_settings
from api.db import SessionLocal
from api.ingest_pipeline import run_pipeline

settings = get_settings()


def _subtract_months(d: date, months: int) -> date:
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def purge_older_than(cutoff: date) -> None:
    db = SessionLocal()
    try:
        db.execute(
            sa_text(
                """
                DELETE FROM rag_chunk
                WHERE document_id IN (
                    SELECT dd.id
                    FROM dogv_documents dd
                    JOIN dogv_issues di ON di.id = dd.issue_id
                    WHERE di.date < :cutoff
                )
                """
            ),
            {"cutoff": cutoff},
        )
        db.execute(
            sa_text(
                """
                DELETE FROM rag_title
                WHERE document_id IN (
                    SELECT dd.id
                    FROM dogv_documents dd
                    JOIN dogv_issues di ON di.id = dd.issue_id
                    WHERE di.date < :cutoff
                )
                """
            ),
            {"cutoff": cutoff},
        )
        db.execute(
            sa_text(
                """
                DELETE FROM rag_doc
                WHERE document_id IN (
                    SELECT dd.id
                    FROM dogv_documents dd
                    JOIN dogv_issues di ON di.id = dd.issue_id
                    WHERE di.date < :cutoff
                )
                """
            ),
            {"cutoff": cutoff},
        )
        db.execute(
            sa_text(
                """
                DELETE FROM dogv_documents
                WHERE issue_id IN (
                    SELECT id FROM dogv_issues WHERE date < :cutoff
                )
                """
            ),
            {"cutoff": cutoff},
        )
        db.execute(sa_text("DELETE FROM dogv_issues WHERE date < :cutoff"), {"cutoff": cutoff})
        db.commit()
        print(f"Purged data older than {cutoff}")
    finally:
        db.close()


def main():
    mode = "--daily" if "--daily" in sys.argv else "--bootstrap"
    today = date.today()
    warm_start = _subtract_months(today, settings.warm_index_months)

    if mode == "--bootstrap":
        run_pipeline(warm_start, today, ["es_es", "va_va"])
    else:
        recent_start = today - timedelta(days=settings.auto_ingest_max_days)
        run_pipeline(recent_start, today, ["es_es", "va_va"])

    purge_older_than(warm_start)


if __name__ == "__main__":
    main()
