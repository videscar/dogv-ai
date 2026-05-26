from __future__ import annotations

import api.embed as embed


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict[str, object]:
        return self._payload


def _embed_response(vectors: list[list[float]]) -> _FakeResponse:
    return _FakeResponse(
        {
            "data": [
                {"embedding": vec, "index": idx, "object": "embedding"}
                for idx, vec in enumerate(vectors)
            ],
            "model": "fake-embed",
            "object": "list",
        }
    )


def test_embed_single_posts_to_v1_embeddings(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_post(url, json, timeout):  # noqa: ANN001
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _embed_response([[0.1, 0.2, 0.3]])

    # Reset cache to avoid pollution across tests
    embed._cached_embed.cache_clear()
    monkeypatch.setattr(embed.requests, "post", _fake_post)

    client = embed.EmbedClient(base_url="http://localhost:9999", model="fake-embed", timeout=5)
    vector = client.embed("hola")

    assert captured["url"] == "http://localhost:9999/v1/embeddings"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "fake-embed"
    assert payload["input"] == ["hola"]
    assert vector == [0.1, 0.2, 0.3]


def test_embed_batch_returns_one_vector_per_input(monkeypatch):
    def _fake_post(url, json, timeout):  # noqa: ANN001
        assert json["input"] == ["a", "b", "c"]
        return _embed_response([[0.1], [0.2], [0.3]])

    monkeypatch.setattr(embed.requests, "post", _fake_post)

    client = embed.EmbedClient(base_url="http://localhost:9999", model="fake-embed", timeout=5)
    vectors = client.embed_batch(["a", "b", "c"])

    assert vectors == [[0.1], [0.2], [0.3]]


def test_embed_batch_returns_empty_for_no_inputs(monkeypatch):
    def _fake_post(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("should not call HTTP when input list is empty")

    monkeypatch.setattr(embed.requests, "post", _fake_post)

    client = embed.EmbedClient(base_url="http://localhost:9999", model="fake-embed", timeout=5)
    assert client.embed_batch([]) == []


def test_embed_batch_raises_on_count_mismatch(monkeypatch):
    def _fake_post(url, json, timeout):  # noqa: ANN001
        return _embed_response([[0.1]])  # only 1 vector for 2 inputs

    monkeypatch.setattr(embed.requests, "post", _fake_post)

    client = embed.EmbedClient(base_url="http://localhost:9999", model="fake-embed", timeout=5)
    try:
        client.embed_batch(["a", "b"])
    except ValueError as exc:
        assert "returned 1 vectors for 2 inputs" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_embed_single_caches_repeated_calls(monkeypatch):
    calls = {"count": 0}

    def _fake_post(url, json, timeout):  # noqa: ANN001
        calls["count"] += 1
        return _embed_response([[0.5, 0.5]])

    embed._cached_embed.cache_clear()
    monkeypatch.setattr(embed.requests, "post", _fake_post)

    client = embed.EmbedClient(base_url="http://localhost:9999", model="fake-embed", timeout=5)
    client.embed("repeat")
    client.embed("repeat")
    client.embed("repeat")

    assert calls["count"] == 1
