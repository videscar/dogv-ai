from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from typing import Any, Iterable

import requests

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("dogv.ollama")


def _json_from_text(text: str) -> dict[str, Any] | None:
    """
    Best-effort JSON extraction from model output.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


@lru_cache(maxsize=2048)
def _cached_embed(base_url: str, model: str, prompt: str, timeout: int) -> list[float]:
    start = time.monotonic()
    ok = False
    payload = {"model": model, "prompt": prompt}
    try:
        resp = requests.post(
            f"{base_url}/api/embeddings",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if embedding is None:
            raise ValueError("Ollama embeddings response missing 'embedding'")
        ok = True
        return embedding
    finally:
        logger.info(
            "ollama.embed ok=%s model=%s chars=%s elapsed=%.2fs",
            ok,
            model,
            len(prompt),
            time.monotonic() - start,
        )


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.embed_model = embed_model or settings.ollama_embed_model
        self.timeout = timeout if timeout is not None else settings.ollama_timeout

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        start = time.monotonic()
        ok = False
        options: dict[str, Any] = {"temperature": temperature}
        if settings.ollama_num_ctx:
            options["num_ctx"] = settings.ollama_num_ctx
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            ok = True
            return data.get("message", {}).get("content", "").strip()
        finally:
            logger.info(
                "ollama.chat ok=%s model=%s messages=%s elapsed=%.2fs",
                ok,
                self.model,
                len(messages),
                time.monotonic() - start,
            )

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        text = self.chat(messages, temperature=temperature)
        parsed = _json_from_text(text)
        if parsed is None:
            raise ValueError(f"Failed to parse JSON from model output: {text[:500]}")
        return parsed

    def embed_batch(self, inputs: Iterable[str]) -> list[list[float]]:
        texts = list(inputs)
        if not texts:
            return []

        start = time.monotonic()
        ok = False
        payload = {"model": self.embed_model, "input": texts}
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 404:
                embeddings = []
                for text in texts:
                    embeddings.append(self.embed(text))
                ok = True
                return embeddings

            resp.raise_for_status()
            data = resp.json()
            ok = True
            return data.get("embeddings", [])
        finally:
            logger.info(
                "ollama.embed_batch ok=%s model=%s batch=%s elapsed=%.2fs",
                ok,
                self.embed_model,
                len(texts),
                time.monotonic() - start,
            )

    def embed(self, text: str) -> list[float]:
        return _cached_embed(self.base_url, self.embed_model, text, self.timeout)
