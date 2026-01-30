"""
Extract text from cached DOGV PDFs into the database.

Usage:
    python scripts/extract_text.py [YYYY-MM-DD [YYYY-MM-DD]]
If dates are provided, filter documents by issue date range (inclusive).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session, load_only
from pypdf import PdfReader

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.dogv_urls import build_html_url
from api.models import DogvDocument, DogvIssue
from api.text_cleaning import clean_html
from scripts.download_assets import build_pdf_path, build_html_path, download_html


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


def extract_text_from_html(path: Path) -> Optional[str]:
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_html(html)
        return text.strip()
    except Exception as e:
        print(f"[error] reading HTML {path}: {e}")
        return None


def _count_target_documents(db: Session, start_date=None, end_date=None) -> int:
    q = (
        db.query(DogvDocument)
        .options(
            load_only(
                DogvDocument.id,
                DogvDocument.html_url,
                DogvDocument.pdf_url,
                DogvDocument.text,
                DogvDocument.text_source,
                DogvDocument.text_updated_at,
            )
        )
        .join(DogvIssue)
        .filter(DogvDocument.text.is_(None))
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
            .filter(DogvDocument.text.is_(None))
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


def extract_range(
    db: Session,
    start_date=None,
    end_date=None,
    batch_size: int = 200,
    commit_every: int = 200,
    fetch_missing_html: bool = False,
) -> int:
    total = _count_target_documents(db, start_date, end_date)
    print(f"Found {total} documents needing text")

    updated = 0
    for ids in iter_target_document_ids(db, start_date, end_date, batch_size=batch_size):
        docs = (
            db.query(DogvDocument)
            .options(
                load_only(
                    DogvDocument.id,
                    DogvDocument.html_url,
                    DogvDocument.pdf_url,
                    DogvDocument.text,
                    DogvDocument.text_source,
                    DogvDocument.text_updated_at,
                )
            )
            .filter(DogvDocument.id.in_(ids))
            .all()
        )
        for doc in docs:
            text = None
            source = None

            if doc.html_url:
                html_path = build_html_path(doc.html_url)
                if not html_path.exists():
                    if fetch_missing_html:
                        full_url = build_html_url(doc.html_url)
                        download_html(full_url, html_path)
                if html_path.exists():
                    text = extract_text_from_html(html_path)
                    if text and len(text) > 200:
                        source = "html"
                    else:
                        text = None

            if text is None and doc.pdf_url:
                pdf_path = build_pdf_path(doc.pdf_url)
                if not pdf_path.exists():
                    print(f"[skip] missing PDF for doc id={doc.id}: {pdf_path}")
                    continue
                text = extract_text_from_pdf(pdf_path)
                if text:
                    source = "pdf"

            if text is None:
                print(f"[skip] no text extracted for doc id={doc.id}")
                continue

            doc.text = text
            doc.text_source = source
            doc.text_updated_at = datetime.now(timezone.utc)
            updated += 1

            if updated % commit_every == 0:
                db.commit()
                print(f"[progress] committed {updated} docs")

    db.commit()
    print(f"Done. Updated text for {updated} documents.")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("end_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--commit-every", type=int, default=200)
    parser.add_argument("--fetch-html", action="store_true")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    db = SessionLocal()
    try:
        extract_range(
            db,
            start_date,
            end_date,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
            fetch_missing_html=args.fetch_html,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
