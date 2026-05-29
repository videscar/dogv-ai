from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from api.db import SessionLocal
from scripts.sumario_ingest import fetch_sumario_json, upsert_issue
from scripts.extract_documents import process_issue
from scripts.extract_text import extract_range
from scripts.classify_documents import classify_range
from scripts.build_chunks import build_chunks_for_range



def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def ingest_range(db: Session, start_date: date, end_date: date, lang: str) -> int:
    total_issues = 0
    for d in daterange(start_date, end_date):
        date_str = d.strftime("%Y-%m-%d")
        try:
            data = fetch_sumario_json(date_str, lang)
        except Exception as e:
            print(f"[warn] sumario fetch failed {date_str} {lang}: {e}")
            continue
        issue = upsert_issue(db, date_str, lang, data)
        process_issue(db, issue)
        total_issues += 1
    return total_issues


def run_pipeline(start_date: date, end_date: date, languages: list[str], force_chunks: bool = False) -> None:
    db = SessionLocal()
    try:
        for lang in languages:
            ingest_range(db, start_date, end_date, lang)

        # No bulk asset download: text now comes from the HTML body for ~90% of
        # documents, and extract_range fetches/parses the PDF on demand only for
        # the annex minority that needs it.
        extract_range(db, start_date, end_date)
        classify_range(db, start_date, end_date)
        build_chunks_for_range(db, start_date, end_date, force=force_chunks)
    finally:
        db.close()
