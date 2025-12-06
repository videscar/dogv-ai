"""
Extract text from cached DOGV PDFs into the database.

Usage:
    python scripts/extract_text.py [YYYY-MM-DD [YYYY-MM-DD]]
If dates are provided, filter documents by issue date range (inclusive).
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session
from pypdf import PdfReader

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.models import DogvDocument, DogvIssue
from scripts.download_pdfs import build_dest_path


def ensure_text_column(db: Session):
    """
    Make sure the text column exists (helpful for existing SQLite DBs).
    """
    cols = [row[1] for row in db.execute(sa_text("PRAGMA table_info(dogv_documents)"))]
    if "text" not in cols:
        db.execute(sa_text("ALTER TABLE dogv_documents ADD COLUMN text TEXT"))
        db.commit()
        print("Added text column to dogv_documents")


def extract_text_from_pdf(path: Path) -> Optional[str]:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
        text = "\n".join(parts).strip()
        return text
    except Exception as e:
        print(f"[error] reading {path}: {e}")
        return None


def iter_target_documents(db: Session, start_date=None, end_date=None):
    q = (
        db.query(DogvDocument)
        .join(DogvIssue)
        .filter(DogvDocument.text.is_(None))
    )
    if start_date:
        q = q.filter(DogvIssue.date >= start_date)
    if end_date:
        q = q.filter(DogvIssue.date <= end_date)
    return q.all()


def main():
    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 else None
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else None

    db = SessionLocal()
    try:
        ensure_text_column(db)
        docs = iter_target_documents(db, start_date, end_date)
        print(f"Found {len(docs)} documents needing text")

        updated = 0
        for idx, doc in enumerate(docs, start=1):
            if not doc.pdf_url:
                print(f"[skip] doc id={doc.id} has no pdf_url")
                continue

            pdf_path = build_dest_path(doc.pdf_url)
            if not pdf_path.exists():
                print(f"[skip] missing PDF for doc id={doc.id}: {pdf_path}")
                continue

            text = extract_text_from_pdf(pdf_path)
            if text is None:
                continue

            doc.text = text
            updated += 1

            if updated % 20 == 0:
                db.commit()
                print(f"[progress] committed {updated} docs")

        db.commit()
        print(f"Done. Updated text for {updated} documents.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
