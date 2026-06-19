from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text as sa_text

from .config import get_settings
from .db import SessionLocal

settings = get_settings()
logger = logging.getLogger("dogv.traces")


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)


def store_trace(payload: dict[str, Any]) -> None:
    if not settings.trace_enabled:
        return

    # Tracing is observability only: it must never take down the answer path.
    # Any failure here is logged and swallowed so a successful answer is still returned.
    try:
        db = SessionLocal()
        try:
            db.execute(
                sa_text(
                    """
                    INSERT INTO qa_traces (question, filters, plan, lanes, fusion, rerank, evidence, answer)
                    VALUES (:question, CAST(:filters AS jsonb), CAST(:plan AS jsonb), CAST(:lanes AS jsonb),
                            CAST(:fusion AS jsonb), CAST(:rerank AS jsonb), CAST(:evidence AS jsonb), :answer)
                    """
                ),
                {
                    "question": payload.get("question"),
                    "filters": _dump(payload.get("filters") or {}),
                    "plan": _dump(payload.get("plan") or {}),
                    "lanes": _dump(payload.get("lanes") or {}),
                    "fusion": _dump(payload.get("fusion") or {}),
                    "rerank": _dump(payload.get("rerank") or {}),
                    "evidence": _dump(payload.get("evidence") or []),
                    "answer": payload.get("answer"),
                },
            )
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.exception("store_trace failed; answer returned without persisting trace")
