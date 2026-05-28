from __future__ import annotations

from datetime import timedelta
import logging
import time

from sqlalchemy import text as sa_text

from agent.shared import QAState, return_with_profile
from api.auto_ingest import ensure_month_ingested
from api.config import get_settings
from api.db import SessionLocal

settings = get_settings()
logger = logging.getLogger("dogv.graph")

DEFAULT_LANGS = ["es_es", "va_va"]


def backfill_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    try:
        if not settings.backfill_enabled:
            elapsed = time.monotonic() - start
            logger.info("backfill.skip req=%s reason=disabled elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "backfill",
                {"backfill_attempted": True},
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="disabled",
            )
        if state.get("candidate_docs"):
            elapsed = time.monotonic() - start
            logger.info("backfill.skip req=%s reason=candidates elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "backfill",
                {"backfill_attempted": True},
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="candidates",
            )
        if state.get("backfill_attempted"):
            elapsed = time.monotonic() - start
            logger.info("backfill.skip req=%s reason=already_attempted elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "backfill",
                {"backfill_attempted": True},
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="already_attempted",
            )

        with SessionLocal() as db:
            row = db.execute(
                sa_text("SELECT MIN(date) AS min_date FROM dogv_issues")
            ).mappings().one()
            min_date = row["min_date"]

        if min_date:
            target = min_date - timedelta(days=1)
            ensure_month_ingested(target, DEFAULT_LANGS)

        elapsed = time.monotonic() - start
        logger.info("backfill.done req=%s elapsed=%.2fs", request_id, elapsed)
        return return_with_profile(
            state,
            "backfill",
            {"backfill_attempted": True},
            elapsed_seconds=round(elapsed, 3),
            status="done",
        )
    except Exception:
        logger.exception("backfill.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
