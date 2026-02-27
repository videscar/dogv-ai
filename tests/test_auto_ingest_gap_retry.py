from __future__ import annotations

from datetime import date

import requests

import api.auto_ingest as auto_ingest


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict:
        return self._payload


def _set_retry_config(monkeypatch, retries: int = 3, backoff: float = 0.0) -> None:
    monkeypatch.setattr(auto_ingest.settings, "auto_ingest_gap_check_retries", retries)
    monkeypatch.setattr(auto_ingest.settings, "auto_ingest_gap_check_backoff_seconds", backoff)
    monkeypatch.setattr(auto_ingest.settings, "dogv_base_url", "https://dogv.gva.es")
    monkeypatch.setattr(auto_ingest.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(auto_ingest.random, "uniform", lambda _a, _b: 0.0)


def test_source_check_retries_transient_errors_then_succeeds(monkeypatch):
    _set_retry_config(monkeypatch, retries=3, backoff=0.0)

    calls = {"count": 0}

    def _fake_get(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.Timeout("temporary timeout")
        return _FakeResponse({"disposiciones": [{"id": 1}]})

    monkeypatch.setattr(auto_ingest.requests, "get", _fake_get)
    monkeypatch.setattr(auto_ingest, "_record_gap_source_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto_ingest, "_mark_gap_source_resolved", lambda *args, **kwargs: None)

    target_date = date(2026, 2, 20)
    cache: dict[tuple[date, str], bool | None] = {}
    result = auto_ingest._source_has_publications(target_date, "es_es", cache, db=None)

    assert result is True
    assert calls["count"] == 3
    assert cache[(target_date, "es_es")] is True


def test_source_check_records_failure_after_retries(monkeypatch):
    _set_retry_config(monkeypatch, retries=3, backoff=0.0)

    calls = {"count": 0}

    def _fake_get(*_args, **_kwargs):
        calls["count"] += 1
        raise requests.Timeout("still failing")

    recorded: list[tuple] = []

    def _record(*args, **kwargs):
        recorded.append((args, kwargs))

    monkeypatch.setattr(auto_ingest.requests, "get", _fake_get)
    monkeypatch.setattr(auto_ingest, "_record_gap_source_failure", _record)
    monkeypatch.setattr(auto_ingest, "_mark_gap_source_resolved", lambda *args, **kwargs: None)

    target_date = date(2026, 2, 21)
    cache: dict[tuple[date, str], bool | None] = {}
    result = auto_ingest._source_has_publications(target_date, "va_va", cache, db=None)

    assert result is None
    assert calls["count"] == 3
    assert len(recorded) == 1
    assert cache[(target_date, "va_va")] is None


def test_source_check_does_not_retry_non_transient_http_404(monkeypatch):
    _set_retry_config(monkeypatch, retries=5, backoff=0.0)

    calls = {"count": 0}

    def _fake_get(*_args, **_kwargs):
        calls["count"] += 1
        return _FakeResponse({"disposiciones": []}, status_code=404)

    recorded: list[tuple] = []

    def _record(*args, **kwargs):
        recorded.append((args, kwargs))

    monkeypatch.setattr(auto_ingest.requests, "get", _fake_get)
    monkeypatch.setattr(auto_ingest, "_record_gap_source_failure", _record)
    monkeypatch.setattr(auto_ingest, "_mark_gap_source_resolved", lambda *args, **kwargs: None)

    target_date = date(2026, 2, 22)
    cache: dict[tuple[date, str], bool | None] = {}
    result = auto_ingest._source_has_publications(target_date, "es_es", cache, db=None)

    assert result is None
    assert calls["count"] == 1
    assert len(recorded) == 1
    assert cache[(target_date, "es_es")] is None


def test_gap_repair_scan_window_is_capped(monkeypatch):
    monkeypatch.setattr(auto_ingest.settings, "auto_ingest_startup_repair_gaps", True)
    monkeypatch.setattr(auto_ingest.settings, "auto_ingest_gap_repair_scan_max_days", 7)
    monkeypatch.setattr(auto_ingest, "_load_doc_counts", lambda *args, **kwargs: {})

    seen_pairs: list[tuple[date, str]] = []

    def _fake_source(issue_date: date, language: str, cache, db=None):
        seen_pairs.append((issue_date, language))
        return False

    monkeypatch.setattr(auto_ingest, "_source_has_publications", _fake_source)

    start = date(2026, 1, 1)
    end = date(2026, 1, 31)
    _, summary = auto_ingest._compute_gap_repair_ranges(
        db=object(),
        start=start,
        end=end,
        languages=["es_es", "va_va"],
    )

    assert summary["scan_window_start"] == "2026-01-25"
    assert summary["scan_window_end"] == "2026-01-31"
    assert summary["candidate_pairs"] == 14
    assert len(seen_pairs) == 14
