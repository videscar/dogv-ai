"""
Run the full ingestion pipeline (sumario -> documents -> text -> classify -> chunks).

Text extraction prefers the portal HTML body and pulls the PDF on demand only
for annex documents, so there is no separate bulk asset-download step.

Usage:
    python scripts/ingest_pipeline.py YYYY-MM-DD YYYY-MM-DD [langs]
Example:
    python scripts/ingest_pipeline.py 2025-01-01 2025-01-31 es_es,va_va
"""

import sys
from datetime import datetime

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.ingest_pipeline import run_pipeline


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/ingest_pipeline.py YYYY-MM-DD YYYY-MM-DD [langs]")
        raise SystemExit(1)

    start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
    end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
    langs = sys.argv[3].split(",") if len(sys.argv) > 3 else ["es_es", "va_va"]
    force_chunks = "--force" in sys.argv[1:]

    run_pipeline(start_date, end_date, langs, force_chunks=force_chunks)


if __name__ == "__main__":
    main()
