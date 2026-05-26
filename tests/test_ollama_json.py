from __future__ import annotations

import api.ollama as ollama


def test_json_from_text_parses_fenced_json_block():
    text = """
Respuesta:
```json
{"answer":"ok","citations":[101]}
```
"""
    parsed = ollama._json_from_text(text)
    assert parsed == {"answer": "ok", "citations": [101]}


def test_json_from_text_extracts_first_balanced_object():
    text = 'preface {"answer":"valor con {llaves}","citations":[7]} trailing'
    parsed = ollama._json_from_text(text)
    assert parsed == {"answer": "valor con {llaves}", "citations": [7]}


def test_chat_json_uses_json_format(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponse:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, object]:
            return {"message": {"content": '{"answer":"ok","citations":[1]}'}}

    def _fake_post(url, json, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(ollama.requests, "post", _fake_post)

    client = ollama.OllamaClient(base_url="http://localhost:11435", model="fake", timeout=5)
    result = client.chat_json([{"role": "user", "content": "hi"}], temperature=0.0)

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload.get("format") == "json"
    assert result == {"answer": "ok", "citations": [1]}
