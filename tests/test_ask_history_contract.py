from __future__ import annotations

from fastapi.testclient import TestClient


def _client(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "start_startup_sync", lambda: None)
    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", False)
    monkeypatch.setattr(main, "store_trace", lambda _payload: None)
    return main, TestClient(main.app)


def _capture_invoke(main, monkeypatch):
    """Stub graph.invoke to record the input state and return a minimal result."""
    captured: dict = {}

    def _invoke(state):
        captured["state"] = state
        return {"answer": "ok", "citations": []}

    monkeypatch.setattr(main.graph, "invoke", _invoke)
    return captured


def test_ask_without_history_threads_empty_list(monkeypatch):
    main, client = _client(monkeypatch)
    captured = _capture_invoke(main, monkeypatch)

    resp = client.post("/ask", json={"question": "hola"})

    assert resp.status_code == 200
    assert captured["state"]["question"] == "hola"
    assert captured["state"]["history"] == []


def test_ask_threads_history_in_order(monkeypatch):
    main, client = _client(monkeypatch)
    captured = _capture_invoke(main, monkeypatch)

    history = [
        {"role": "user", "content": "primera"},
        {"role": "assistant", "content": "respuesta"},
    ]
    resp = client.post("/ask", json={"question": "segunda", "history": history})

    assert resp.status_code == 200
    assert captured["state"]["history"] == history


def test_ask_truncates_history_to_max_turns(monkeypatch):
    main, client = _client(monkeypatch)
    monkeypatch.setattr(main.settings, "ask_history_max_turns", 2)
    captured = _capture_invoke(main, monkeypatch)

    history = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    resp = client.post("/ask", json={"question": "q", "history": history})

    assert resp.status_code == 200
    threaded = captured["state"]["history"]
    assert len(threaded) == 2
    assert [t["content"] for t in threaded] == ["m4", "m5"]


def test_ask_rejects_malformed_history_role(monkeypatch):
    main, client = _client(monkeypatch)
    _capture_invoke(main, monkeypatch)

    resp = client.post(
        "/ask",
        json={"question": "q", "history": [{"role": "system", "content": "x"}]},
    )
    assert resp.status_code == 422


def test_ask_rejects_history_missing_content(monkeypatch):
    main, client = _client(monkeypatch)
    _capture_invoke(main, monkeypatch)

    resp = client.post("/ask", json={"question": "q", "history": [{"role": "user"}]})
    assert resp.status_code == 422


def test_ask_stream_threads_history(monkeypatch):
    main, client = _client(monkeypatch)
    captured: dict = {}

    def _stream(state, stream_mode):
        captured["state"] = state
        yield ("updates", {"answer_node": {"answer": "ok", "citations": []}})
        yield ("values", {"answer": "ok", "citations": [], "intent": {}, "filters": None})

    monkeypatch.setattr(main.graph, "stream", _stream)

    history = [{"role": "user", "content": "antes"}]
    with client:
        resp = client.post(
            "/ask/stream", json={"question": "ahora", "history": history}
        )

    assert resp.status_code == 200
    assert captured["state"]["history"] == history
