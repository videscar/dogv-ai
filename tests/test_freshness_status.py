from __future__ import annotations

from datetime import date

import api.auto_ingest as auto_ingest


class _DummySession:
    def __enter__(self):
        return object()

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def test_freshness_lag_days_is_clamped_to_zero(monkeypatch):
    monkeypatch.setattr(auto_ingest, "SessionLocal", _DummySession)
    monkeypatch.setattr(auto_ingest, "local_today", lambda _tz: date(2026, 2, 25))
    monkeypatch.setattr(
        auto_ingest,
        "get_issue_bounds",
        lambda _db: (date(2026, 2, 20), date(2026, 2, 26)),
    )

    payload = auto_ingest.get_freshness_status()
    assert payload["lag_days"] == 0


def test_freshness_lag_days_positive_when_source_is_older(monkeypatch):
    monkeypatch.setattr(auto_ingest, "SessionLocal", _DummySession)
    monkeypatch.setattr(auto_ingest, "local_today", lambda _tz: date(2026, 2, 25))
    monkeypatch.setattr(
        auto_ingest,
        "get_issue_bounds",
        lambda _db: (date(2026, 2, 1), date(2026, 2, 23)),
    )

    payload = auto_ingest.get_freshness_status()
    assert payload["lag_days"] == 2
