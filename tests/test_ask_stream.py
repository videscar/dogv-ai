from __future__ import annotations

from fastapi.testclient import TestClient


NOT_READY_PAYLOAD = {
    "ready": False,
    "status": "warming",
    "reason": "startup_sync_running",
    "freshness": None,
    "startup_sync": {"state": "running", "success": None},
}


def _client(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "start_startup_sync", lambda: None)
    return main, TestClient(main.app)


def _parse_sse(text: str) -> list[tuple[str, str]]:
    """Return [(event, data)] from a raw SSE body."""
    events: list[tuple[str, str]] = []
    event = "message"
    data: list[str] = []
    for line in text.splitlines():
        if line == "":
            if data:
                events.append((event, "\n".join(data)))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].lstrip())
    if data:
        events.append((event, "\n".join(data)))
    return events


def test_ask_stream_emits_stages_then_result(monkeypatch):
    main, client = _client(monkeypatch)

    def _stream(_payload, stream_mode):
        assert stream_mode == ["updates", "values"]
        # Interleave updates (per-node deltas) and values (running full state),
        # exactly as langgraph emits for a list stream_mode.
        yield ("updates", {"analyze_intent": {"intent": {"language": "es"}}})
        yield ("values", {"intent": {"language": "es"}})
        yield ("updates", {"retrieve_candidates": {"candidate_docs": []}})
        yield ("values", {"intent": {"language": "es"}, "candidate_docs": []})
        yield ("updates", {"answer_node": {"answer": "respuesta", "citations": []}})
        yield (
            "values",
            {
                "answer": "respuesta",
                "citations": [
                    {
                        "document_id": 11,
                        "title": "Documento",
                        "ref": "2026/11",
                        "issue_date": "2026-02-23",
                        "pdf_url": "/pdf/11",
                        "html_url": "/html/11",
                    }
                ],
                "candidate_docs": [],
                "intent": {"language": "es"},
                "filters": None,
            },
        )

    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", False)
    monkeypatch.setattr(main.graph, "stream", _stream)
    monkeypatch.setattr(main, "store_trace", lambda _payload: None)

    with client:
        response = client.post("/ask/stream", json={"question": "hola", "debug": False})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    kinds = [e for e, _ in events]

    # Three known nodes -> three stage events, then result, then done.
    assert kinds == ["stage", "stage", "stage", "result", "done"]

    import json

    stage_labels = [json.loads(d)["label"] for e, d in events if e == "stage"]
    assert stage_labels == [
        "Analizando la consulta",
        "Recuperando documentos del DOGV",
        "Redactando la respuesta",
    ]

    result = next(json.loads(d) for e, d in events if e == "result")
    assert result["answer"] == "respuesta"
    assert len(result["citations"]) == 1


def test_ask_stream_503_when_readiness_gate_blocks(monkeypatch):
    main, client = _client(monkeypatch)
    streamed = {"called": False}

    def _stream(_payload, stream_mode):
        streamed["called"] = True
        yield ("values", {})

    monkeypatch.setattr(main, "build_readiness_payload", lambda: NOT_READY_PAYLOAD)
    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", True)
    monkeypatch.setattr(main.graph, "stream", _stream)

    with client:
        response = client.post("/ask/stream", json={"question": "hola", "debug": False})

    assert response.status_code == 503
    assert response.json()["detail"] == NOT_READY_PAYLOAD
    assert streamed["called"] is False


def test_ask_stream_emits_error_event_on_failure(monkeypatch):
    main, client = _client(monkeypatch)

    def _stream(_payload, stream_mode):
        yield ("updates", {"analyze_intent": {"intent": {}}})
        raise RuntimeError("boom")

    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", False)
    monkeypatch.setattr(main.graph, "stream", _stream)
    monkeypatch.setattr(main, "store_trace", lambda _payload: None)

    with client:
        response = client.post("/ask/stream", json={"question": "hola", "debug": False})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    kinds = [e for e, _ in events]
    assert "error" in kinds
    assert "result" not in kinds
