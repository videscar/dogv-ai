from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


class BackendError(RuntimeError):
    """Base error for backend calls."""


class BackendTimeoutError(BackendError):
    """Backend request timed out."""


class BackendUnavailableError(BackendError):
    """Backend service is unreachable."""


class BackendProtocolError(BackendError):
    """Backend response is malformed."""


@dataclass
class BackendHttpError(BackendError):
    status_code: int
    detail: Any
    message: str

    def __str__(self) -> str:
        return f"{self.status_code}: {self.message}"


@dataclass
class DogvApiClient:
    base_url: str
    timeout_seconds: float | None = 60.0

    async def get_ready(self) -> dict[str, Any]:
        return await self._request_json("GET", "/ready")

    async def ask(
        self,
        question: str,
        debug: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        payload = {"question": question, "debug": debug, "history": history or []}
        return await self._request_json("POST", "/ask", json=payload)

    async def ask_stream(
        self,
        question: str,
        debug: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (event_type, data) tuples from the /ask/stream SSE endpoint.

        Event types: 'stage' (progress), 'result' (final answer payload),
        'done', 'error'. Connection/timeout/HTTP problems raise the same
        Backend* errors as ask(), so callers can fall back to the blocking path.
        """
        base_url = self.base_url.rstrip("/")
        # No read timeout between events: synthesis can run minutes with no event
        # in between; keep a bounded connect timeout so an unreachable backend
        # still fails fast.
        read = None if self.timeout_seconds is None else float(self.timeout_seconds)
        timeout = httpx.Timeout(connect=10.0, read=read, write=10.0, pool=10.0)
        payload = {"question": question, "debug": debug, "history": history or []}
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
                async with client.stream("POST", "/ask/stream", json=payload) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        raise BackendHttpError(
                            status_code=response.status_code,
                            detail=_response_detail(response),
                            message=response.reason_phrase or "Backend error",
                        )
                    event = "message"
                    data_lines: list[str] = []
                    async for line in response.aiter_lines():
                        if line == "":
                            if data_lines:
                                yield event, _parse_sse_data("\n".join(data_lines))
                            event, data_lines = "message", []
                            continue
                        if line.startswith(":"):
                            continue  # SSE comment / keep-alive
                        if line.startswith("event:"):
                            event = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:") :].lstrip())
                    if data_lines:  # flush a trailing event without blank line
                        yield event, _parse_sse_data("\n".join(data_lines))
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("Request timed out") from exc
        except httpx.NetworkError as exc:
            raise BackendUnavailableError("Backend unavailable") from exc

    async def _request_json(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_url = self.base_url.rstrip("/")
        timeout = None if self.timeout_seconds is None else float(self.timeout_seconds)
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
                response = await client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("Request timed out") from exc
        except httpx.NetworkError as exc:
            raise BackendUnavailableError("Backend unavailable") from exc

        if response.status_code >= 400:
            detail = _response_detail(response)
            raise BackendHttpError(
                status_code=response.status_code,
                detail=detail,
                message=response.reason_phrase or "Backend error",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendProtocolError("Backend returned non-JSON response") from exc

        if not isinstance(payload, dict):
            raise BackendProtocolError("Backend returned unexpected JSON payload")
        return payload


def _parse_sse_data(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _response_detail(response: httpx.Response) -> Any:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("detail")
        return payload
    except ValueError:
        return response.text
