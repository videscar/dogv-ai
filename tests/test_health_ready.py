from __future__ import annotations

from fastapi.testclient import TestClient


READY_PAYLOAD = {
    "ready": True,
    "status": "ready",
    "reason": None,
    "freshness": {
        "today": "2026-02-23",
        "min_issue_date": "2026-02-01",
        "max_issue_date": "2026-02-23",
        "lag_days": 0,
    },
    "startup_sync": {
        "state": "finished",
        "success": True,
    },
}

WARMING_PAYLOAD = {
    "ready": False,
    "status": "warming",
    "reason": "startup_sync_running",
    "freshness": {
        "today": "2026-02-23",
        "min_issue_date": "2026-02-01",
        "max_issue_date": "2026-02-22",
        "lag_days": 1,
    },
    "startup_sync": {
        "state": "running",
        "success": None,
    },
}


def _client(monkeypatch, payload: dict):
    import api.main as main

    monkeypatch.setattr(main, "build_readiness_payload", lambda: payload)
    monkeypatch.setattr(main, "start_startup_sync", lambda: None)
    return TestClient(main.app)


def test_ready_endpoint_returns_readiness_payload(monkeypatch):
    with _client(monkeypatch, READY_PAYLOAD) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == READY_PAYLOAD


def test_health_exposes_core_fields_when_ready(monkeypatch):
    with _client(monkeypatch, READY_PAYLOAD) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["freshness"] == READY_PAYLOAD["freshness"]
    assert body["startup_sync"] == READY_PAYLOAD["startup_sync"]
    assert "readiness" not in body


def test_health_includes_readiness_hint_when_not_ready(monkeypatch):
    with _client(monkeypatch, WARMING_PAYLOAD) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["readiness"] == {
        "ready": False,
        "status": "warming",
        "reason": "startup_sync_running",
    }
