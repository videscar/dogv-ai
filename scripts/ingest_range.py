"""
Ingest a range of DOGV issues (sumario + documents) into the DB.

Usage:
    python scripts/ingest_range.py YYYY-MM-DD YYYY-MM-DD [lang]

Example:
    python scripts/ingest_range.py 2024-01-01 2024-01-31 es_es
"""

import sys
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.models import DogvIssue
from scripts.sumario_ingest import fetch_sumario_json, upsert_issue
from scripts.extract_documents import process_issue


def daterange(start_date, end_date):
    """Yield dates from start_date to end_date inclusive."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def ingest_issue_for_date(db: Session, date_str: str, lang: str) -> DogvIssue | None:
    try:
        print(f"=== {date_str} ({lang}) ===")
        data = fetch_sumario_json(date_str, lang)
    except Exception as e:
        print(f"[ERROR] Fetch failed for {date_str} {lang}: {e}")
        return None

    issue = upsert_issue(db, date_str, lang, data)
    print(f"Stored issue id={issue.id}, numero={issue.numero}, title={issue.title[:80] if issue.title else None}")
    return issue


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/ingest_range.py YYYY-MM-DD YYYY-MM-DD [lang]")
        raise SystemExit(1)

    start_str = sys.argv[1]
    end_str = sys.argv[2]
    lang = sys.argv[3] if len(sys.argv) > 3 else "es_es"

    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d").date()

    db = SessionLocal()
    try:
        total_issues = 0
        total_docs = 0

        for d in daterange(start_date, end_date):
            date_str = d.strftime("%Y-%m-%d")
            issue = ingest_issue_for_date(db, date_str, lang)
            if not issue:
                continue

            if issue.numero is None:
                print(f"No DOGV issue for {date_str} ({lang}) – skipping (weekend/holiday?)")
                continue

            total_issues += 1
            docs_count = process_issue(db, issue)
            total_docs += docs_count

        print(f"\nDone. Issues stored: {total_issues}, documents stored: {total_docs}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
