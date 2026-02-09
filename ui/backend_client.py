from __future__ import annotations

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
    timeout_seconds: float = 60.0

    async def get_ready(self) -> dict[str, Any]:
        return await self._request_json("GET", "/ready")

    async def ask(self, question: str, debug: bool = False) -> dict[str, Any]:
        payload = {"question": question, "debug": debug}
        return await self._request_json("POST", "/ask", json=payload)

    async def _request_json(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_url = self.base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=self.timeout_seconds) as client:
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


def _response_detail(response: httpx.Response) -> Any:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("detail")
        return payload
    except ValueError:
        return response.text

