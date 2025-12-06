"""
Download DOGV document PDFs into a local cache under data/pdf_cache.

Usage:
    python scripts/download_pdfs.py
"""

import sys
from pathlib import Path

import requests
from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import get_settings
from api.db import SessionLocal
from api.models import DogvDocument

settings = get_settings()
CACHE_ROOT = Path("data/pdf_cache")


def build_full_url(pdf_url: str) -> str:
    """
    Build the full PDF URL.
    The public PDFs live under /datos/, so we prefix that path unless the URL is absolute.
    """
    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        return pdf_url
    base = settings.dogv_base_url.rstrip("/")
    return f"{base}/datos/{pdf_url.lstrip('/')}"


def build_dest_path(pdf_url: str) -> Path:
    """
    Convert a DOGV pdf_url like '/2024/01/15/pdf/2024_12087.pdf' into
    data/pdf_cache/2024/01/15/2024_12087.pdf. If the pattern differs,
    fall back to mirroring the URL path under pdf_cache.
    """
    url_path = Path(pdf_url.lstrip("/"))
    parts = url_path.parts
    if len(parts) >= 4:
        return CACHE_ROOT / parts[0] / parts[1] / parts[2] / parts[-1]
    return CACHE_ROOT / url_path


def download_pdf(full_url: str, dest: Path) -> bool:
    """Download a single PDF to dest. Returns True if downloaded, False if skipped."""
    if dest.exists():
        print(f"[skip] exists: {dest}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(full_url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[error] {full_url}: {e}")
        return False

    dest.write_bytes(resp.content)
    print(f"[ok] {dest} ({len(resp.content)} bytes)")
    return True


def iter_documents(db: Session):
    return db.query(DogvDocument).filter(DogvDocument.pdf_url.isnot(None)).all()


def main():
    db = SessionLocal()
    try:
        docs = iter_documents(db)
        print(f"Found {len(docs)} documents with pdf_url")
        downloaded = 0
        for doc in docs:
            pdf_url = doc.pdf_url
            if not pdf_url:
                continue
            full_url = build_full_url(pdf_url)
            dest = build_dest_path(pdf_url)
            if download_pdf(full_url, dest):
                downloaded += 1
        print(f"Done. Downloaded {downloaded} PDFs.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
