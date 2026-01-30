"""
Download DOGV document HTML (preferred) and PDF (fallback) into local caches.

Usage:
    python scripts/download_assets.py [YYYY-MM-DD [YYYY-MM-DD]]
If dates are provided, filter documents by issue date range (inclusive).
"""

import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.dogv_urls import build_html_url, build_pdf_url
from api.models import DogvDocument, DogvIssue
from scripts.download_utils import build_dest_path as _build_dest_path, download_to_path

CACHE_HTML_ROOT = Path("data/html_cache")
CACHE_PDF_ROOT = Path("data/pdf_cache")
CHUNK_SIZE_HTML = 1024 * 256
CHUNK_SIZE_PDF = 1024 * 1024


def build_html_path(html_url: str) -> Path:
    return _build_dest_path(html_url, CACHE_HTML_ROOT)


def build_pdf_path(pdf_url: str) -> Path:
    return _build_dest_path(pdf_url, CACHE_PDF_ROOT)


def download_html(full_url: str, dest: Path) -> bool:
    return download_to_path(full_url, dest, chunk_size=CHUNK_SIZE_HTML)


def download_pdf(full_url: str, dest: Path) -> bool:
    return download_to_path(full_url, dest, chunk_size=CHUNK_SIZE_PDF)


def iter_documents(db: Session, start_date=None, end_date=None):
    q = (
        db.query(DogvDocument)
        .join(DogvIssue)
        .filter((DogvDocument.html_url.isnot(None)) | (DogvDocument.pdf_url.isnot(None)))
    )
    if start_date:
        q = q.filter(DogvIssue.date >= start_date)
    if end_date:
        q = q.filter(DogvIssue.date <= end_date)
    return q.all()


def download_assets(db: Session, start_date=None, end_date=None) -> None:
    docs = iter_documents(db, start_date, end_date)
    if start_date or end_date:
        print(f"Found {len(docs)} documents with assets in date range")
    else:
        print(f"Found {len(docs)} documents with assets")

    html_downloaded = 0
    pdf_downloaded = 0
    for doc in docs:
        used_html = False
        if doc.html_url:
            html_path = build_html_path(doc.html_url)
            if html_path.exists():
                used_html = True
            else:
                full_url = build_html_url(doc.html_url)
                if full_url and download_html(full_url, html_path):
                    html_downloaded += 1
                    used_html = True

        if used_html:
            continue

        if doc.pdf_url:
            pdf_path = build_pdf_path(doc.pdf_url)
            if pdf_path.exists():
                continue
            full_url = build_pdf_url(doc.pdf_url)
            if full_url and download_pdf(full_url, pdf_path):
                pdf_downloaded += 1

    print(f"Done. Downloaded HTML={html_downloaded} PDF={pdf_downloaded}.")


def main():
    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 else None
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date() if len(sys.argv) > 2 else None

    db = SessionLocal()
    try:
        download_assets(db, start_date, end_date)
    finally:
        db.close()


if __name__ == "__main__":
    main()
