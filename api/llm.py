from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from .config import get_settings

settings = get_settings()
logger = logging.getLogger("dogv.llm")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None


def _strip_think_blocks(text: str) -> str:
    # llama-server with --reasoning-format auto already routes <think> to a
    # separate reasoning_content field, but defensively strip in case the
    # server config changes or a stray block leaks into message.content.
    return _THINK_BLOCK_RE.sub("", text or "").strip()


def _json_from_text(text: str) -> dict[str, Any] | None:
    text = _strip_think_blocks(text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    candidate = _extract_first_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    return None


class LlmClient:
    """OpenAI-compatible chat client backed by a llama-server process."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.timeout = timeout if timeout is not None else settings.llm_timeout
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        start = time.monotonic()
        ok = False
        # Unsloth Qwen3.6-27B sampling presets. The model card's thinking-mode preset
        # is temp=1.0/top_p=0.95, but at temp=1.0 the grounded synthesis is unstable
        # (it flips between the correct fact and "no consta" on identical input), so we
        # honor the caller's temperature in thinking mode too and keep the preset's
        # top_p. Callers that want the model-card default can still pass temperature=1.0.
        if enable_thinking:
            sample_temp = temperature
            sample_top_p = 0.95
        else:
            sample_temp = temperature
            sample_top_p = 0.8
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": sample_temp,
            "top_p": sample_top_p,
            "top_k": 20,
            "presence_penalty": 1.5,
            "min_p": 0.0,
            "stream": False,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if not enable_thinking:
            # Qwen3 + llama.cpp jinja template: disables <think> generation entirely.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            ok = True
            return content
        finally:
            logger.info(
                "llm.chat ok=%s model=%s messages=%s format=%s think=%s elapsed=%.2fs",
                ok,
                self.model,
                len(messages),
                "json" if response_format else "none",
                "on" if enable_thinking else "off",
                time.monotonic() - start,
            )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        enable_thinking: bool = True,
    ) -> dict[str, Any]:
        text = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            enable_thinking=enable_thinking,
        )
        parsed = _json_from_text(text)
        if parsed is None:
            # Fallback for servers/models that ignore response_format.
            text = self.chat(messages, temperature=temperature, enable_thinking=enable_thinking)
            parsed = _json_from_text(text)
        if parsed is None:
            raise ValueError(f"Failed to parse JSON from model output: {text[:500]}")
        return parsed
