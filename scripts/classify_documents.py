"""
Classify DOGV documents by topic/subtopic using gpt-oss-20b.

Usage:
    python scripts/classify_documents.py [YYYY-MM-DD [YYYY-MM-DD]]
"""

import argparse
from datetime import datetime

from sqlalchemy.orm import Session, load_only

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.classify import classify_document
from api.db import SessionLocal
from api.models import DogvDocument, DogvIssue


def _count_target_documents(db: Session, start_date=None, end_date=None) -> int:
    q = (
        db.query(DogvDocument.id)
        .join(DogvIssue)
        .filter(DogvDocument.text.isnot(None))
        .filter(DogvDocument.doc_kind.is_(None))
    )
    if start_date:
        q = q.filter(DogvIssue.date >= start_date)
    if end_date:
        q = q.filter(DogvIssue.date <= end_date)
    return q.count()


def iter_target_document_ids(db: Session, start_date=None, end_date=None, batch_size: int = 200):
    last_id = 0
    while True:
        q = (
            db.query(DogvDocument.id)
            .join(DogvIssue)
            .filter(DogvDocument.text.isnot(None))
            .filter(DogvDocument.doc_kind.is_(None))
            .filter(DogvDocument.id > last_id)
        )
        if start_date:
            q = q.filter(DogvIssue.date >= start_date)
        if end_date:
            q = q.filter(DogvIssue.date <= end_date)
        q = q.order_by(DogvDocument.id.asc()).limit(batch_size)
        ids = [row[0] for row in q.all()]
        if not ids:
            break
        yield ids
        last_id = ids[-1]


def classify_range(
    db: Session,
    start_date=None,
    end_date=None,
    batch_size: int = 200,
    commit_every: int = 200,
) -> int:
    total = _count_target_documents(db, start_date, end_date)
    print(f"Found {total} documents to classify")
    updated = 0

    for ids in iter_target_document_ids(db, start_date, end_date, batch_size=batch_size):
        docs = (
            db.query(DogvDocument)
            .options(load_only(DogvDocument.id, DogvDocument.title, DogvDocument.text))
            .filter(DogvDocument.id.in_(ids))
            .all()
        )
        for doc in docs:
            try:
                result = classify_document(doc.title, doc.text)
                doc.doc_kind = result.get("doc_kind") or "Otros"
                doc.doc_subkind = result.get("doc_subkind")
                doc.doc_kind_confidence = result.get("confidence")
                doc.doc_tags = {"tags": result.get("tags") or []}
            except Exception as exc:
                print(f"[warn] classify failed for doc {doc.id}: {exc}")
                doc.doc_kind = "Otros"
                doc.doc_subkind = None
                doc.doc_kind_confidence = None
                doc.doc_tags = {"tags": []}
            updated += 1
            if updated % commit_every == 0:
                db.commit()
                print(f"[progress] classified {updated} docs")
    db.commit()
    print(f"Done. Classified {updated} documents.")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("end_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--commit-every", type=int, default=200)
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    db = SessionLocal()
    try:
        classify_range(
            db,
            start_date,
            end_date,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
