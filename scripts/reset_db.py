"""
Reset DOGV database tables for a fresh ingest.

Usage:
    python scripts/reset_db.py [--with-cache]
"""

import shutil
from pathlib import Path
import sys

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal


def main():
    with_cache = "--with-cache" in sys.argv[1:]

    db = SessionLocal()
    try:
        for table in ("rag_chunk", "rag_title", "rag_doc", "qa_traces", "dogv_documents", "dogv_issues"):
            exists = db.execute(sa_text("SELECT to_regclass(:name)"), {"name": table}).scalar()
            if exists:
                db.execute(sa_text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
        db.commit()
        print("Database tables truncated.")
    finally:
        db.close()

    if with_cache:
        for path in (Path("data/pdf_cache"), Path("data/html_cache"), Path("data/sumarios")):
            if path.exists():
                shutil.rmtree(path)
                print(f"Removed {path}")


if __name__ == "__main__":
    main()
