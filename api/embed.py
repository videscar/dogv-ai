from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from functools import lru_cache

import requests

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("dogv.embed")


def _post_embeddings(
    base_url: str,
    model: str,
    inputs: list[str],
    timeout: int,
) -> list[list[float]]:
    payload = {"model": model, "input": inputs}
    resp = requests.post(
        f"{base_url}/v1/embeddings",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") or []
    vectors: list[list[float]] = []
    for item in items:
        vector = item.get("embedding")
        if vector is None:
            raise ValueError("embeddings response item missing 'embedding'")
        vectors.append(vector)
    if len(vectors) != len(inputs):
        raise ValueError(
            f"embeddings response returned {len(vectors)} vectors for {len(inputs)} inputs"
        )
    return vectors


@lru_cache(maxsize=2048)
def _cached_embed(base_url: str, model: str, text: str, timeout: int) -> tuple[float, ...]:
    start = time.monotonic()
    ok = False
    try:
        vectors = _post_embeddings(base_url, model, [text], timeout)
        ok = True
        return tuple(vectors[0])
    finally:
        logger.info(
            "embed.single ok=%s model=%s chars=%s elapsed=%.2fs",
            ok,
            model,
            len(text),
            time.monotonic() - start,
        )


class EmbedClient:
    """OpenAI-compatible embeddings client backed by a llama-server process."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.embed_base_url).rstrip("/")
        self.model = model or settings.embed_model
        self.timeout = timeout if timeout is not None else settings.embed_timeout

    def embed(self, text: str) -> list[float]:
        return list(_cached_embed(self.base_url, self.model, text, self.timeout))

    def embed_batch(self, inputs: Iterable[str]) -> list[list[float]]:
        texts = list(inputs)
        if not texts:
            return []
        start = time.monotonic()
        ok = False
        try:
            vectors = _post_embeddings(self.base_url, self.model, texts, self.timeout)
            ok = True
            return vectors
        finally:
            logger.info(
                "embed.batch ok=%s model=%s batch=%s elapsed=%.2fs",
                ok,
                self.model,
                len(texts),
                time.monotonic() - start,
            )
