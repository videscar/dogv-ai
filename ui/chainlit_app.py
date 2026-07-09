from __future__ import annotations

import os
from typing import Any


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_chainlit_data_layer() -> None:
    if not _truthy(os.getenv("CHAINLIT_ENABLE_DATA_LAYER")):
        os.environ.pop("DATABASE_URL", None)
        return

    db_url = os.getenv("DATABASE_URL", "")
    if not db_url.startswith("postgresql+"):
        return

    # Chainlit uses asyncpg and requires postgresql:// or postgres:// DSNs.
    dsn = os.getenv("DOGV_DB_DSN")
    if dsn and dsn.startswith(("postgresql://", "postgres://")):
        os.environ["DATABASE_URL"] = dsn
        return

    os.environ["DATABASE_URL"] = db_url.replace("postgresql+psycopg2://", "postgresql://", 1)


_configure_chainlit_data_layer()

import chainlit as cl

# Chainlit loads .env on import, so normalize again after import.
_configure_chainlit_data_layer()

try:
    from ui.backend_client import (
        BackendHttpError,
        BackendTimeoutError,
        BackendUnavailableError,
        DogvApiClient,
    )
except ModuleNotFoundError:
    # Chainlit file-based execution may not include repo root on sys.path.
    from backend_client import (  # type: ignore
        BackendHttpError,
        BackendTimeoutError,
        BackendUnavailableError,
        DogvApiClient,
    )


BACKEND_URL = os.getenv("CHAINLIT_BACKEND_URL", "http://127.0.0.1:8000")


def _parse_request_timeout(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return 60.0
    lowered = raw.lower()
    if lowered in {"none", "inf", "infinite", "infinity"}:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return 60.0
    if parsed <= 0:
        return None
    return parsed


REQUEST_TIMEOUT_SECONDS = _parse_request_timeout(os.getenv("DEMO_REQUEST_TIMEOUT_SECONDS"))

TIMEOUT_MESSAGE = (
    "La consulta ha tardado demasiado y se ha cancelado por timeout. "
    "Inténtalo de nuevo en unos segundos."
)
UNAVAILABLE_MESSAGE = (
    "No se puede conectar con el backend DOGV ahora mismo. "
    "Comprueba que la API está levantada y vuelve a intentarlo."
)
GENERIC_ERROR_MESSAGE = "Se ha producido un error en el backend. Inténtalo de nuevo."


def _client() -> DogvApiClient:
    return DogvApiClient(base_url=BACKEND_URL, timeout_seconds=REQUEST_TIMEOUT_SECONDS)


def _readiness_message(readiness: dict[str, Any]) -> str:
    status = str(readiness.get("status") or "").lower()
    reason = str(readiness.get("reason") or "").lower()
    if status == "warming" or reason == "startup_sync_running":
        return (
            "El sistema está preparando el índice (startup sync en curso). "
            "Espera un momento y vuelve a enviar la consulta."
        )
    if status == "degraded":
        return (
            "El backend no está listo para responder de forma fiable "
            "(estado degradado). Revisa `/ready` y vuelve a intentarlo."
        )
    return "El backend todavía no está listo para tráfico de demo."


def _extract_readiness_from_error(exc: BackendHttpError) -> dict[str, Any] | None:
    if exc.status_code != 503:
        return None
    detail = exc.detail
    if isinstance(detail, dict) and "ready" in detail:
        return detail
    return None


def _render_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = ["### Citas"]
    for idx, item in enumerate(citations, start=1):
        title = (
            item.get("title") or item.get("ref") or f"Documento {item.get('document_id')}"
        ).strip()
        url = item.get("html_url") or item.get("pdf_url")
        metadata_parts = []
        if item.get("issue_date"):
            metadata_parts.append(f"fecha: {item['issue_date']}")
        if item.get("ref"):
            metadata_parts.append(f"ref: {item['ref']}")
        if item.get("document_id") is not None:
            metadata_parts.append(f"doc: {item['document_id']}")
        meta = f" ({' | '.join(metadata_parts)})" if metadata_parts else ""
        if url:
            lines.append(f"{idx}. [{title}]({url}){meta}")
        else:
            lines.append(f"{idx}. {title}{meta}")
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("dogv_client", _client())
    cl.user_session.set("history", [])
    client: DogvApiClient = cl.user_session.get("dogv_client")
    try:
        readiness = await client.get_ready()
    except BackendTimeoutError:
        await cl.Message(content=TIMEOUT_MESSAGE).send()
        return
    except BackendUnavailableError:
        await cl.Message(content=UNAVAILABLE_MESSAGE).send()
        return

    if not readiness.get("ready"):
        await cl.Message(content=_readiness_message(readiness)).send()


STREAM_ENABLED = _truthy(os.getenv("DEMO_STREAM_ENABLED", "1"))


def _history_max_messages() -> int:
    """Stored messages cap = 2 * max turns (one user + one assistant per turn)."""
    try:
        turns = int(os.getenv("DEMO_HISTORY_MAX_TURNS", "6"))
    except ValueError:
        turns = 6
    return max(0, turns) * 2


def _get_history() -> list[dict[str, str]]:
    return list(cl.user_session.get("history") or [])


def _append_history(question: str, answer: str) -> None:
    history = _get_history()
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    cap = _history_max_messages()
    cl.user_session.set("history", history[-cap:] if cap else [])


def _render_answer(response: dict[str, Any]) -> str:
    answer = (response.get("answer") or "").strip() or "No hay respuesta disponible."
    citations_block = _render_citations(response.get("citations") or [])
    return answer if not citations_block else f"{answer}\n\n{citations_block}"


def _progress_content(done_labels: list[str], finished: bool = False) -> str:
    lines = ["**Procesando tu consulta…**", ""]
    lines += [f"- ✅ {label}" for label in done_labels]
    if not finished:
        lines.append("- ⏳ …")
    return "\n".join(lines)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="Escribe una pregunta para consultar DOGV.").send()
        return

    client: DogvApiClient = cl.user_session.get("dogv_client") or _client()
    history = _get_history()
    if STREAM_ENABLED:
        answer = await _answer_streaming(client, question, history)
    else:
        answer = await _answer_blocking(client, question, history)
    if answer:
        _append_history(question, answer)


async def _answer_streaming(
    client: DogvApiClient, question: str, history: list[dict[str, str]]
) -> str | None:
    """Stream an answer; return the plain answer text (no citations) for history."""
    progress = cl.Message(content=_progress_content([]))
    await progress.send()
    done_labels: list[str] = []
    result: dict[str, Any] | None = None
    try:
        async for event, data in client.ask_stream(question, history=history):
            if event == "stage":
                label = str(data.get("label") or "").strip()
                if label and label not in done_labels:
                    done_labels.append(label)
                    progress.content = _progress_content(done_labels)
                    await progress.update()
            elif event == "result":
                result = data
            elif event == "error":
                await progress.remove()
                await cl.Message(content=GENERIC_ERROR_MESSAGE).send()
                return
    except BackendTimeoutError:
        await progress.remove()
        await cl.Message(content=TIMEOUT_MESSAGE).send()
        return
    except BackendUnavailableError:
        await progress.remove()
        await cl.Message(content=UNAVAILABLE_MESSAGE).send()
        return
    except BackendHttpError as exc:
        await progress.remove()
        readiness = _extract_readiness_from_error(exc)
        if readiness:
            await cl.Message(content=_readiness_message(readiness)).send()
            return
        await cl.Message(content=f"{GENERIC_ERROR_MESSAGE} (HTTP {exc.status_code})").send()
        return

    await progress.remove()
    if result is None:
        await cl.Message(content=GENERIC_ERROR_MESSAGE).send()
        return None
    await cl.Message(content=_render_answer(result)).send()
    return (result.get("answer") or "").strip() or None


async def _answer_blocking(
    client: DogvApiClient, question: str, history: list[dict[str, str]]
) -> str | None:
    try:
        response = await client.ask(question, history=history)
    except BackendTimeoutError:
        await cl.Message(content=TIMEOUT_MESSAGE).send()
        return
    except BackendUnavailableError:
        await cl.Message(content=UNAVAILABLE_MESSAGE).send()
        return
    except BackendHttpError as exc:
        readiness = _extract_readiness_from_error(exc)
        if readiness:
            await cl.Message(content=_readiness_message(readiness)).send()
            return
        await cl.Message(content=f"{GENERIC_ERROR_MESSAGE} (HTTP {exc.status_code})").send()
        return

    await cl.Message(content=_render_answer(response)).send()
    return (response.get("answer") or "").strip() or None
