from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from api.config import get_settings
from api.db import SessionLocal
from scripts.build_chunks import build_chunks_for_range
from scripts.classify_documents import classify_range
from scripts.extract_documents import process_issue
from scripts.extract_text import extract_range
from scripts.sumario_ingest import fetch_sumario_json, upsert_issue

logger = logging.getLogger("dogv.ingest")


def daterange(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _complete_sibling_edition(db: Session, issue, lang: str) -> int:
    """Capture the sibling edition the date-sumario JSON dropped (bis-collision dates).

    The portal's `?date=` sumario returns only ONE of an ordinary+bis pair, so the
    other edition's dispositions are silently lost. After the JSON edition is
    ingested we enumerate the sibling from its sumario PDF and create rows for any
    dispositions still missing; the pipeline's batch extract/classify/chunk then
    processes them. Best-effort — never breaks normal ingest. Cheap on the common
    no-sibling date (one PDF fetch -> 404)."""
    from api.auto_ingest import upsert_disposicion_row  # lazy: avoid import cycle
    from api.bis_editions import missing_sibling_disposicion_ids

    numero = (issue.numero or "").strip()
    if not numero:
        return 0
    is_bis = str((issue.raw_json or {}).get("esBis")).lower() == "true"
    try:
        missing = missing_sibling_disposicion_ids(issue.date.isoformat(), numero, lang, is_bis)
    except Exception:
        logger.exception("bis.sibling_enumerate_failed date=%s lang=%s", issue.date, lang)
        return 0
    created = 0
    for disp_id in missing.values():
        try:
            if upsert_disposicion_row(db, disp_id, lang, source_tag="sibling_edition"):
                created += 1
        except Exception:
            logger.exception("bis.sibling_row_failed disp_id=%s lang=%s", disp_id, lang)
    if created:
        db.commit()
        logger.info("bis.sibling_recovered date=%s lang=%s rows=%d", issue.date, lang, created)
    return created


def ingest_range(db: Session, start_date: date, end_date: date, lang: str) -> int:
    complete_bis = getattr(get_settings(), "ingest_complete_bis_editions", True)
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
        if complete_bis:
            _complete_sibling_edition(db, issue, lang)
        total_issues += 1
    return total_issues


def run_pipeline(
    start_date: date, end_date: date, languages: list[str], force_chunks: bool = False
) -> None:
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
