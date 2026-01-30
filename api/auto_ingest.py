from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from .config import get_settings
from .ingest_pipeline import run_pipeline

settings = get_settings()


def _month_bounds(target: date) -> tuple[date, date]:
    start = target.replace(day=1)
    if target.month == 12:
        next_month = target.replace(year=target.year + 1, month=1, day=1)
    else:
        next_month = target.replace(month=target.month + 1, day=1)
    end = next_month - timedelta(days=1)
    return start, end


def get_issue_bounds(db: Session) -> tuple[date | None, date | None]:
    row = db.execute(
        sa_text("SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM dogv_issues")
    ).mappings().one()
    return row["min_date"], row["max_date"]


def ensure_recent_ingested(days: int, languages: list[str]) -> None:
    today = date.today()
    start = today - timedelta(days=days)
    run_pipeline(start, today, languages)


def ensure_month_ingested(target: date, languages: list[str]) -> None:
    start, end = _month_bounds(target)
    run_pipeline(start, end, languages)


def ensure_range_ingested(start: date, end: date, languages: list[str]) -> None:
    run_pipeline(start, end, languages)
