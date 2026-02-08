from __future__ import annotations

import copy
from datetime import date, datetime, timedelta, timezone
import logging
import threading
from typing import Any

import requests
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from .config import get_settings
from .db import SessionLocal
from .ingest_pipeline import run_pipeline
from .temporal import local_today

settings = get_settings()
logger = logging.getLogger("dogv.auto_ingest")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_languages(raw: str | None) -> list[str]:
    if not raw:
        return ["es_es", "va_va"]
    parsed = [token.strip() for token in raw.split(",") if token.strip()]
    return parsed or ["es_es", "va_va"]


_status_lock = threading.Lock()
_startup_status: dict[str, Any] = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "success": None,
    "error": None,
    "skipped": False,
    "skip_reason": None,
    "window_start": None,
    "today": None,
    "ranges": [],
    "before": {"min_date": None, "max_date": None},
    "after": {"min_date": None, "max_date": None},
    "lag_days": None,
    "gap_repair": {
        "enabled": True,
        "candidate_pairs": 0,
        "checked_pairs": 0,
        "source_positive_pairs": 0,
        "failed_source_checks": 0,
        "repair_dates": 0,
        "repair_ranges": [],
    },
}
_startup_thread: threading.Thread | None = None


def _set_startup_status(**values: Any) -> None:
    with _status_lock:
        _startup_status.update(values)


def get_startup_sync_status() -> dict[str, Any]:
    with _status_lock:
        return copy.deepcopy(_startup_status)


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


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _subtract_months(day: date, months: int) -> date:
    months = max(1, int(months))
    year = day.year
    month = day.month - months
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, min(day.day, _days_in_month(year, month)))


def _merge_ranges(ranges: list[tuple[date, date]]) -> list[tuple[date, date]]:
    valid = sorted((start, end) for start, end in ranges if start <= end)
    if not valid:
        return []
    merged: list[tuple[date, date]] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= (last_end + timedelta(days=1)):
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return merged


def _purge_older_than(db: Session, cutoff: date) -> None:
    db.execute(
        sa_text(
            """
            DELETE FROM rag_chunk
            WHERE document_id IN (
                SELECT dd.id
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                WHERE di.date < :cutoff
            )
            """
        ),
        {"cutoff": cutoff},
    )
    db.execute(
        sa_text(
            """
            DELETE FROM rag_title
            WHERE document_id IN (
                SELECT dd.id
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                WHERE di.date < :cutoff
            )
            """
        ),
        {"cutoff": cutoff},
    )
    db.execute(
        sa_text(
            """
            DELETE FROM rag_doc
            WHERE document_id IN (
                SELECT dd.id
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                WHERE di.date < :cutoff
            )
            """
        ),
        {"cutoff": cutoff},
    )
    db.execute(
        sa_text(
            """
            DELETE FROM dogv_documents
            WHERE issue_id IN (
                SELECT id FROM dogv_issues WHERE date < :cutoff
            )
            """
        ),
        {"cutoff": cutoff},
    )
    db.execute(sa_text("DELETE FROM dogv_issues WHERE date < :cutoff"), {"cutoff": cutoff})
    db.commit()


def get_freshness_status() -> dict[str, Any]:
    today = local_today(settings.temporal_timezone)
    with SessionLocal() as db:
        min_date, max_date = get_issue_bounds(db)
    lag_days = (today - max_date).days if isinstance(max_date, date) else None
    return {
        "today": today.isoformat(),
        "min_issue_date": min_date.isoformat() if isinstance(min_date, date) else None,
        "max_issue_date": max_date.isoformat() if isinstance(max_date, date) else None,
        "lag_days": lag_days,
    }


def _compute_startup_ranges(
    min_date: date | None,
    max_date: date | None,
    window_start: date,
    today: date,
) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    if max_date is None:
        ranges.append((window_start, today))
    elif max_date < today:
        catchup_start = max(window_start, max_date + timedelta(days=1))
        if catchup_start <= today:
            ranges.append((catchup_start, today))

    if min_date is None or min_date > window_start:
        fill_end = today if min_date is None else min(today, min_date - timedelta(days=1))
        if window_start <= fill_end:
            ranges.append((window_start, fill_end))
    return _merge_ranges(ranges)


def _iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _load_doc_counts(
    db: Session, start: date, end: date, languages: list[str]
) -> dict[tuple[date, str], int]:
    rows = db.execute(
        sa_text(
            """
            SELECT di.date AS issue_date, di.language AS issue_language, COUNT(dd.id) AS doc_count
            FROM dogv_issues di
            LEFT JOIN dogv_documents dd ON dd.issue_id = di.id
            WHERE di.date BETWEEN :start_date AND :end_date
            GROUP BY di.date, di.language
            """
        ),
        {"start_date": start, "end_date": end},
    ).mappings()
    language_set = {lang.strip() for lang in languages if lang.strip()}
    counts: dict[tuple[date, str], int] = {}
    for row in rows:
        issue_language = (row["issue_language"] or "").strip()
        if issue_language not in language_set:
            continue
        issue_date = row["issue_date"]
        if not isinstance(issue_date, date):
            continue
        counts[(issue_date, issue_language)] = int(row["doc_count"] or 0)
    return counts


def _source_has_publications(
    issue_date: date,
    language: str,
    cache: dict[tuple[date, str], bool | None],
) -> bool | None:
    key = (issue_date, language)
    if key in cache:
        return cache[key]
    base_url = settings.dogv_base_url.rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/dogv-portal/dogv",
            params={"date": issue_date.isoformat(), "lang": language},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        has_docs = bool(payload.get("disposiciones") or [])
        cache[key] = has_docs
        return has_docs
    except Exception:
        logger.exception("gap.repair source check failed date=%s lang=%s", issue_date.isoformat(), language)
        cache[key] = None
        return None


def _compute_gap_repair_ranges(
    db: Session,
    start: date,
    end: date,
    languages: list[str],
) -> tuple[list[tuple[date, date]], dict[str, Any]]:
    if not bool(settings.auto_ingest_startup_repair_gaps):
        return [], {
            "enabled": False,
            "candidate_pairs": 0,
            "checked_pairs": 0,
            "source_positive_pairs": 0,
            "failed_source_checks": 0,
            "repair_dates": 0,
            "repair_ranges": [],
        }

    language_list = [lang.strip() for lang in languages if lang.strip()]
    doc_counts = _load_doc_counts(db, start, end, language_list)
    candidates: list[tuple[date, str]] = []
    for issue_date in _iter_dates(start, end):
        for language in language_list:
            docs = doc_counts.get((issue_date, language))
            if docs is None or docs == 0:
                candidates.append((issue_date, language))

    source_cache: dict[tuple[date, str], bool | None] = {}
    repair_dates: set[date] = set()
    checked_pairs = 0
    positive_pairs = 0
    failed_checks = 0

    for issue_date, language in candidates:
        has_docs = _source_has_publications(issue_date, language, source_cache)
        checked_pairs += 1
        if has_docs is None:
            failed_checks += 1
            continue
        if has_docs:
            positive_pairs += 1
            repair_dates.add(issue_date)

    ranges = _merge_ranges([(day, day) for day in sorted(repair_dates)])
    summary = {
        "enabled": True,
        "candidate_pairs": len(candidates),
        "checked_pairs": checked_pairs,
        "source_positive_pairs": positive_pairs,
        "failed_source_checks": failed_checks,
        "repair_dates": len(repair_dates),
        "repair_ranges": [{"start": s.isoformat(), "end": e.isoformat()} for s, e in ranges],
    }
    return ranges, summary


def startup_sync(languages: list[str] | None = None) -> dict[str, Any]:
    default_gap_repair = {
        "enabled": bool(settings.auto_ingest_startup_repair_gaps),
        "candidate_pairs": 0,
        "checked_pairs": 0,
        "source_positive_pairs": 0,
        "failed_source_checks": 0,
        "repair_dates": 0,
        "repair_ranges": [],
    }
    if not settings.auto_ingest_enabled:
        _set_startup_status(
            state="finished",
            started_at=_utc_now_iso(),
            finished_at=_utc_now_iso(),
            success=True,
            skipped=True,
            skip_reason="auto_ingest_disabled",
            gap_repair=default_gap_repair,
        )
        return get_startup_sync_status()
    if not settings.auto_ingest_startup_enabled:
        _set_startup_status(
            state="finished",
            started_at=_utc_now_iso(),
            finished_at=_utc_now_iso(),
            success=True,
            skipped=True,
            skip_reason="startup_sync_disabled",
            gap_repair=default_gap_repair,
        )
        return get_startup_sync_status()

    selected_languages = languages or _parse_languages(settings.auto_ingest_languages)
    keep_months = max(1, int(settings.auto_ingest_keep_months))
    today = local_today(settings.temporal_timezone)
    window_start = _subtract_months(today, keep_months)
    started_at = _utc_now_iso()

    _set_startup_status(
        state="running",
        started_at=started_at,
        finished_at=None,
        success=None,
        error=None,
        skipped=False,
        skip_reason=None,
        window_start=window_start.isoformat(),
        today=today.isoformat(),
        ranges=[],
        before={"min_date": None, "max_date": None},
        after={"min_date": None, "max_date": None},
        lag_days=None,
        gap_repair=default_gap_repair,
    )

    lock_session = SessionLocal()
    lock_acquired = False
    try:
        lock_acquired = bool(
            lock_session.execute(
                sa_text("SELECT pg_try_advisory_lock(:lock_id) AS acquired"),
                {"lock_id": int(settings.auto_ingest_startup_lock_id)},
            ).scalar()
        )
        lock_session.commit()
        if not lock_acquired:
            _set_startup_status(
                state="finished",
                finished_at=_utc_now_iso(),
                success=True,
                skipped=True,
                skip_reason="lock_not_acquired",
                gap_repair=default_gap_repair,
            )
            return get_startup_sync_status()

        min_before, max_before = get_issue_bounds(lock_session)
        ranges = _compute_startup_ranges(min_before, max_before, window_start, today)
        gap_ranges, gap_summary = _compute_gap_repair_ranges(
            lock_session,
            window_start,
            today,
            selected_languages,
        )

        logger.info(
            "startup.sync begin languages=%s window_start=%s today=%s ranges=%s gap_ranges=%s",
            selected_languages,
            window_start.isoformat(),
            today.isoformat(),
            [(s.isoformat(), e.isoformat()) for s, e in ranges],
            [(s.isoformat(), e.isoformat()) for s, e in gap_ranges],
        )

        for start, end in ranges:
            run_pipeline(start, end, selected_languages)
        for start, end in gap_ranges:
            run_pipeline(start, end, selected_languages)

        if settings.auto_ingest_startup_purge_old:
            _purge_older_than(lock_session, window_start)

        min_after, max_after = get_issue_bounds(lock_session)
        lag_days = (today - max_after).days if isinstance(max_after, date) else None
        _set_startup_status(
            state="finished",
            finished_at=_utc_now_iso(),
            success=True,
            ranges=[{"start": s.isoformat(), "end": e.isoformat()} for s, e in ranges],
            before={
                "min_date": min_before.isoformat() if isinstance(min_before, date) else None,
                "max_date": max_before.isoformat() if isinstance(max_before, date) else None,
            },
            after={
                "min_date": min_after.isoformat() if isinstance(min_after, date) else None,
                "max_date": max_after.isoformat() if isinstance(max_after, date) else None,
            },
            lag_days=lag_days,
            gap_repair=gap_summary,
        )
        logger.info(
            "startup.sync done after_max=%s lag_days=%s gap_repair=%s",
            max_after.isoformat() if isinstance(max_after, date) else None,
            lag_days,
            gap_summary,
        )
    except Exception as exc:
        logger.exception("startup.sync failed")
        _set_startup_status(
            state="finished",
            finished_at=_utc_now_iso(),
            success=False,
            error=str(exc),
        )
    finally:
        if lock_acquired:
            try:
                lock_session.execute(
                    sa_text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": int(settings.auto_ingest_startup_lock_id)},
                )
                lock_session.commit()
            except Exception:
                logger.exception("startup.sync unlock failed")
        lock_session.close()

    return get_startup_sync_status()


def start_startup_sync(languages: list[str] | None = None) -> None:
    global _startup_thread

    if bool(settings.auto_ingest_startup_blocking):
        startup_sync(languages=languages)
        return

    with _status_lock:
        if _startup_thread and _startup_thread.is_alive():
            return

        def _runner() -> None:
            startup_sync(languages=languages)

        _startup_thread = threading.Thread(target=_runner, name="dogv-startup-sync", daemon=True)
        _startup_thread.start()


def ensure_recent_ingested(days: int, languages: list[str]) -> None:
    today = local_today(settings.temporal_timezone)
    start = today - timedelta(days=days)
    run_pipeline(start, today, languages)


def ensure_month_ingested(target: date, languages: list[str]) -> None:
    start, end = _month_bounds(target)
    run_pipeline(start, end, languages)


def ensure_range_ingested(start: date, end: date, languages: list[str]) -> None:
    run_pipeline(start, end, languages)
