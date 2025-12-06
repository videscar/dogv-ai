"""
Debug helper: print the JSON for one stored DOGV issue.

Usage:
    python scripts/debug_issue_json.py YYYY-MM-DD [lang]
"""

import sys
import json
from datetime import datetime

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401
from api.db import SessionLocal
from api.models import DogvIssue


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_issue_json.py YYYY-MM-DD [lang]")
        raise SystemExit(1)

    date_str = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "es_es"
    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    db = SessionLocal()
    try:
        issue = (
            db.query(DogvIssue)
            .filter(DogvIssue.date == date, DogvIssue.language == lang)
            .one_or_none()
        )
        if not issue:
            print("No issue found for that date/lang in DB.")
            return

        print("Top-level keys:", list(issue.raw_json.keys()))
        print("\nPretty JSON (truncated):")
        print(json.dumps(issue.raw_json, indent=2, ensure_ascii=False)[:4000])
    finally:
        db.close()


if __name__ == "__main__":
    main()
