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


def test_ask_returns_503_when_readiness_gate_enabled(monkeypatch):
    main, client = _client(monkeypatch)
    invoked = {"called": False}

    def _invoke(_payload):
        invoked["called"] = True
        return {"answer": "should not run", "citations": []}

    monkeypatch.setattr(main, "build_readiness_payload", lambda: NOT_READY_PAYLOAD)
    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", True)
    monkeypatch.setattr(main.graph, "invoke", _invoke)

    with client:
        response = client.post("/ask", json={"question": "hola", "debug": False})

    assert response.status_code == 503
    assert response.json()["detail"] == NOT_READY_PAYLOAD
    assert invoked["called"] is False


def test_ask_runs_when_readiness_gate_disabled(monkeypatch):
    main, client = _client(monkeypatch)
    invoked = {"called": False}

    def _invoke(payload):
        invoked["called"] = True
        assert payload["question"] == "hola"
        return {
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
            "selected_doc_ids": [11],
            "evidence": [],
            "intent": {"language": "es"},
            "filters": None,
        }

    monkeypatch.setattr(main, "build_readiness_payload", lambda: NOT_READY_PAYLOAD)
    monkeypatch.setattr(main.settings, "demo_enforce_ready_gate", False)
    monkeypatch.setattr(main.graph, "invoke", _invoke)
    monkeypatch.setattr(main, "store_trace", lambda _payload: None)

    with client:
        response = client.post("/ask", json={"question": "hola", "debug": False})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "respuesta"
    assert len(body["citations"]) == 1
    assert invoked["called"] is True
