from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from sqlalchemy import text as sa_text

from agent.shared import QAState, return_with_profile
from api.auto_ingest import (
    ensure_range_ingested,
    ensure_recent_ingested,
    get_issue_bounds,
    get_startup_sync_status,
)
from api.config import get_settings
from api.db import SessionLocal
from api.temporal import local_today

settings = get_settings()
logger = logging.getLogger("dogv.graph")

DEFAULT_LANGS = ["es_es", "va_va"]


def _missing_ranges_from_bounds(
    start: date,
    end: date,
    min_date: date | None,
    max_date: date | None,
) -> list[tuple[date, date]]:
    if start > end:
        return []
    if min_date is None or max_date is None:
        return [(start, end)]

    missing: list[tuple[date, date]] = []
    if start < min_date:
        left_end = min(end, min_date - timedelta(days=1))
        if start <= left_end:
            missing.append((start, left_end))
    if end > max_date:
        right_start = max(start, max_date + timedelta(days=1))
        if right_start <= end:
            missing.append((right_start, end))
    return missing


def online_ingest_node(state: QAState) -> QAState:
    started_at = time.monotonic()
    request_id = state.get("request_id")
    try:
        if not settings.auto_ingest_enabled:
            elapsed = time.monotonic() - started_at
            logger.info("ingest.skip req=%s reason=disabled elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "ingest",
                {"online_ingest_done": True},
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="disabled",
            )
        startup_state = (get_startup_sync_status().get("state") or "").lower()
        if startup_state == "running":
            elapsed = time.monotonic() - started_at
            logger.info(
                "ingest.skip req=%s reason=startup_sync_running elapsed=%.2fs",
                request_id,
                elapsed,
            )
            return return_with_profile(
                state,
                "ingest",
                {"online_ingest_done": True},
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="startup_sync_running",
            )

        intent = state.get("intent") or {}
        filters = state.get("filters")
        since_date = intent.get("since_date")
        until_date = intent.get("until_date")
        if since_date is None and filters:
            since_date = filters.since_date
        if until_date is None and filters:
            until_date = filters.until_date
        if since_date and until_date and since_date > until_date:
            since_date, until_date = until_date, since_date

        if since_date or until_date:
            ingest_start = since_date or until_date
            ingest_end = until_date or since_date
            if ingest_start and ingest_end:
                with SessionLocal() as db:
                    min_date, max_date = get_issue_bounds(db)

                missing_ranges = _missing_ranges_from_bounds(
                    ingest_start,
                    ingest_end,
                    min_date=min_date,
                    max_date=max_date,
                )
                if not missing_ranges:
                    elapsed = time.monotonic() - started_at
                    logger.info(
                        "ingest.skip req=%s reason=range_already_covered start=%s end=%s min=%s max=%s elapsed=%.2fs",
                        request_id,
                        ingest_start,
                        ingest_end,
                        min_date,
                        max_date,
                        elapsed,
                    )
                else:
                    for missing_start, missing_end in missing_ranges:
                        ensure_range_ingested(missing_start, missing_end, DEFAULT_LANGS)
        else:
            today = local_today(settings.temporal_timezone)
            with SessionLocal() as db:
                row = db.execute(sa_text("SELECT MAX(date) AS max_date FROM dogv_issues")).mappings().one()
                max_date = row["max_date"]
            if not max_date or max_date < today:
                ensure_recent_ingested(settings.auto_ingest_max_days, DEFAULT_LANGS)
        elapsed = time.monotonic() - started_at
        logger.info("ingest.done req=%s elapsed=%.2fs", request_id, elapsed)
        return return_with_profile(
            state,
            "ingest",
            {"online_ingest_done": True},
            elapsed_seconds=round(elapsed, 3),
            status="done",
        )
    except Exception:
        logger.exception("ingest.error req=%s elapsed=%.2fs", request_id, time.monotonic() - started_at)
        raise
