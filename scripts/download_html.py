"""
Download DOGV document HTML into a local cache under data/html_cache.

Usage:
    python scripts/download_html.py [YYYY-MM-DD [YYYY-MM-DD]]
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
from api.dogv_urls import build_html_url
from api.models import DogvDocument, DogvIssue
from scripts.download_utils import build_dest_path as _build_dest_path, download_to_path

CACHE_ROOT = Path("data/html_cache")
CHUNK_SIZE = 1024 * 256


def build_dest_path(html_url: str) -> Path:
    return _build_dest_path(html_url, CACHE_ROOT)


def download_html(full_url: str, dest: Path) -> bool:
    return download_to_path(full_url, dest, chunk_size=CHUNK_SIZE)


def iter_documents(db: Session, start_date=None, end_date=None):
    q = (
        db.query(DogvDocument)
        .join(DogvIssue)
        .filter(DogvDocument.html_url.isnot(None))
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
        docs = iter_documents(db, start_date, end_date)
        if start_date or end_date:
            print(f"Found {len(docs)} documents with html_url in date range")
        else:
            print(f"Found {len(docs)} documents with html_url")
        downloaded = 0
        for doc in docs:
            html_url = doc.html_url
            if not html_url:
                continue
            full_url = build_html_url(html_url)
            if not full_url:
                continue
            dest = build_dest_path(html_url)
            if download_html(full_url, dest):
                downloaded += 1
        print(f"Done. Downloaded {downloaded} HTML files.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
