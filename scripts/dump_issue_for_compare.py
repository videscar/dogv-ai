"""
Print all documents for a given issue (by numero or date) so you can
compare them against the DOGV web 'sumario'.

Usage:
    python scripts/dump_issue_for_compare.py --numero 9757
    python scripts/dump_issue_for_compare.py --date 2024-01-02
"""

import argparse
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.models import DogvIssue, DogvDocument


def get_issue(db: Session, numero: str | None, date_str: str | None) -> DogvIssue | None:
    q = db.query(DogvIssue).filter(DogvIssue.numero.isnot(None))
    if numero:
        q = q.filter(DogvIssue.numero == numero)
    if date_str:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        q = q.filter(DogvIssue.date == date)
    return q.order_by(DogvIssue.date.asc()).first()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numero", help="DOGV issue number, e.g. 9757")
    parser.add_argument("--date", help="Issue date YYYY-MM-DD")
    args = parser.parse_args()

    if not args.numero and not args.date:
        parser.error("You must pass --numero or --date")

    db = SessionLocal()
    try:
        issue = get_issue(db, args.numero, args.date)
        if not issue:
            print("No matching issue found.")
            return

        print(f"Issue id={issue.id}  numero={issue.numero}  date={issue.date}  lang={issue.language}")
        print(f"Title: {issue.title}\n")

        docs = (
            db.query(DogvDocument)
            .filter(DogvDocument.issue_id == issue.id)
            .order_by(DogvDocument.section.asc(), DogvDocument.id.asc())
            .all()
        )

        print(f"Documents in this issue: {len(docs)}\n")
        current_section = None
        for d in docs:
            if d.section != current_section:
                current_section = d.section
                print("\n=== SECTION:", current_section, "===\n")
            print(f"- conselleria : {d.conselleria}")
            print(f"  title       : {d.title}")
            print(f"  pdf_url     : {d.pdf_url}")
            print(f"  text_len    : {len(d.text) if getattr(d, 'text', None) else 0}")
            print()
    finally:
        db.close()


if __name__ == "__main__":
    main()
