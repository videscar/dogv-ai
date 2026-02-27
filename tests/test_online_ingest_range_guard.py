from __future__ import annotations

from datetime import date

import agent.graph as graph


class _DummySession:
    def __enter__(self):
        return object()

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def test_missing_ranges_returns_full_range_when_bounds_absent():
    missing = graph._missing_ranges_from_bounds(
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        min_date=None,
        max_date=None,
    )
    assert missing == [(date(2025, 1, 1), date(2025, 12, 31))]


def test_missing_ranges_returns_empty_when_range_is_covered():
    missing = graph._missing_ranges_from_bounds(
        start=date(2025, 4, 1),
        end=date(2025, 9, 30),
        min_date=date(2025, 1, 1),
        max_date=date(2025, 12, 31),
    )
    assert missing == []


def test_missing_ranges_returns_both_edges_when_needed():
    missing = graph._missing_ranges_from_bounds(
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        min_date=date(2025, 3, 1),
        max_date=date(2025, 10, 31),
    )
    assert missing == [
        (date(2025, 1, 1), date(2025, 2, 28)),
        (date(2025, 11, 1), date(2025, 12, 31)),
    ]


def test_online_ingest_skips_when_range_is_already_covered(monkeypatch):
    monkeypatch.setattr(graph.settings, "auto_ingest_enabled", True)
    monkeypatch.setattr(graph, "get_startup_sync_status", lambda: {"state": "idle"})
    monkeypatch.setattr(graph, "SessionLocal", lambda: _DummySession())
    monkeypatch.setattr(
        graph,
        "get_issue_bounds",
        lambda _db: (date(2025, 1, 1), date(2025, 12, 31)),
    )

    calls: list[tuple[date, date, list[str]]] = []

    def _capture(start: date, end: date, languages: list[str]) -> None:
        calls.append((start, end, languages))

    monkeypatch.setattr(graph, "ensure_range_ingested", _capture)

    result = graph.online_ingest_node(
        {
            "request_id": "test-covered",
            "intent": {
                "since_date": date(2025, 1, 1),
                "until_date": date(2025, 12, 31),
            },
        }
    )

    assert result["online_ingest_done"] is True
    assert calls == []


def test_online_ingest_ingests_only_missing_edge_ranges(monkeypatch):
    monkeypatch.setattr(graph.settings, "auto_ingest_enabled", True)
    monkeypatch.setattr(graph, "get_startup_sync_status", lambda: {"state": "idle"})
    monkeypatch.setattr(graph, "SessionLocal", lambda: _DummySession())
    monkeypatch.setattr(
        graph,
        "get_issue_bounds",
        lambda _db: (date(2025, 3, 1), date(2025, 10, 31)),
    )

    calls: list[tuple[date, date, list[str]]] = []

    def _capture(start: date, end: date, languages: list[str]) -> None:
        calls.append((start, end, languages))

    monkeypatch.setattr(graph, "ensure_range_ingested", _capture)

    result = graph.online_ingest_node(
        {
            "request_id": "test-missing-edges",
            "intent": {
                "since_date": date(2025, 1, 1),
                "until_date": date(2025, 12, 31),
            },
        }
    )

    assert result["online_ingest_done"] is True
    assert calls == [
        (date(2025, 1, 1), date(2025, 2, 28), graph.DEFAULT_LANGS),
        (date(2025, 11, 1), date(2025, 12, 31), graph.DEFAULT_LANGS),
    ]
