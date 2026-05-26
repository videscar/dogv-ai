from __future__ import annotations

import api.llm as llm


def test_json_from_text_parses_plain_object():
    assert llm._json_from_text('{"answer":"ok","citations":[1]}') == {
        "answer": "ok",
        "citations": [1],
    }


def test_json_from_text_parses_fenced_block():
    text = """
Aqui tienes:
```json
{"answer":"ok","citations":[101]}
```
"""
    assert llm._json_from_text(text) == {"answer": "ok", "citations": [101]}


def test_json_from_text_extracts_first_balanced_object():
    text = 'preface {"answer":"valor con {llaves}","citations":[7]} trailing'
    assert llm._json_from_text(text) == {
        "answer": "valor con {llaves}",
        "citations": [7],
    }


def test_json_from_text_strips_think_blocks_before_parsing():
    text = "<think>step 1...step 2...</think>{\"answer\":\"ok\",\"citations\":[1]}"
    assert llm._json_from_text(text) == {"answer": "ok", "citations": [1]}


def test_json_from_text_returns_none_for_non_object():
    assert llm._json_from_text("[1,2,3]") is None
    assert llm._json_from_text("") is None
    assert llm._json_from_text("not json at all") is None


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, object]:
        return self._payload


def _chat_response(content: str) -> _FakeResponse:
    return _FakeResponse(
        {
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        }
    )


def test_chat_json_requests_openai_json_response_format(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_post(url, json, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _chat_response('{"answer":"ok","citations":[1]}')

    monkeypatch.setattr(llm.requests, "post", _fake_post)

    client = llm.LlmClient(base_url="http://localhost:9999", model="fake", timeout=5, max_tokens=128)
    result = client.chat_json([{"role": "user", "content": "hi"}])

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "fake"
    assert payload["max_tokens"] == 128
    assert payload["response_format"] == {"type": "json_object"}
    assert captured["url"] == "http://localhost:9999/v1/chat/completions"
    assert result == {"answer": "ok", "citations": [1]}


def test_chat_json_retries_without_response_format_when_parser_fails(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_post(url, json, timeout):  # noqa: ANN001
        calls.append(json)
        if len(calls) == 1:
            return _chat_response("garbage that is not json")
        return _chat_response('{"answer":"ok","citations":[]}')

    monkeypatch.setattr(llm.requests, "post", _fake_post)

    client = llm.LlmClient(base_url="http://localhost:9999", model="fake", timeout=5, max_tokens=128)
    result = client.chat_json([{"role": "user", "content": "hi"}])

    assert len(calls) == 2
    assert calls[0].get("response_format") == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert result == {"answer": "ok", "citations": []}


def test_chat_json_raises_when_both_passes_fail(monkeypatch):
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda url, json, timeout: _chat_response("still not json"),
    )

    client = llm.LlmClient(base_url="http://localhost:9999", model="fake", timeout=5, max_tokens=128)
    try:
        client.chat_json([{"role": "user", "content": "hi"}])
    except ValueError as exc:
        assert "Failed to parse JSON" in str(exc)
    else:
        raise AssertionError("expected ValueError")
