from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text as sa_text

from .config import get_settings
from .db import SessionLocal

settings = get_settings()


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)


def store_trace(payload: dict[str, Any]) -> None:
    if not settings.trace_enabled:
        return

    db = SessionLocal()
    try:
        db.execute(
            sa_text(
                """
                INSERT INTO qa_traces (question, filters, plan, lanes, fusion, rerank, evidence, answer)
                VALUES (:question, :filters::jsonb, :plan::jsonb, :lanes::jsonb, :fusion::jsonb,
                        :rerank::jsonb, :evidence::jsonb, :answer)
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
