from __future__ import annotations

from typing import Any

from .auto_ingest import get_freshness_status, get_startup_sync_status


def build_readiness_payload() -> dict[str, Any]:
    startup_sync = get_startup_sync_status()
    startup_state = (startup_sync.get("state") or "").lower()

    freshness: dict[str, Any] | None = None
    freshness_error = False
    try:
        freshness = get_freshness_status()
    except Exception:
        freshness_error = True

    if startup_state == "running":
        return {
            "ready": False,
            "status": "warming",
            "reason": "startup_sync_running",
            "freshness": freshness,
            "startup_sync": startup_sync,
        }
    if startup_state == "finished" and startup_sync.get("success") is False:
        return {
            "ready": False,
            "status": "degraded",
            "reason": "startup_sync_failed",
            "freshness": freshness,
            "startup_sync": startup_sync,
        }
    if freshness_error:
        return {
            "ready": False,
            "status": "degraded",
            "reason": "freshness_unavailable",
            "freshness": None,
            "startup_sync": startup_sync,
        }
    return {
        "ready": True,
        "status": "ready",
        "reason": None,
        "freshness": freshness,
        "startup_sync": startup_sync,
    }
