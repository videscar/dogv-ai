from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import date
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from agent.graph import build_graph

from .auto_ingest import start_startup_sync
from .build_info import build_info
from .config import enabled_lanes, get_settings
from .db import SessionLocal
from .models import DogvDocument, DogvIssue
from .readiness import build_readiness_payload
from .traces import store_trace

# Langchain-core expects langchain.debug to exist in some versions.
try:
    import langchain  # type: ignore

    if not hasattr(langchain, "debug"):
        langchain.debug = False  # type: ignore[attr-defined]
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dogv.api")

app = FastAPI(title="DOGV AI – Local Legal Assistant")
graph = build_graph()
settings = get_settings()

# Stamp the running code + eval-relevant settings into the logs at startup, so any
# run (especially multi-hour eval jobs) can be tied to the exact commit + config
# that produced it. git_dirty=True means uncommitted code is serving requests.
_BUILD = build_info()
logger.info(
    "build.info git_sha=%s branch=%s dirty=%s claim_guard=%s synthesis_thinking=%s repair_mode=%s",
    _BUILD.get("git_sha"),
    _BUILD.get("git_branch"),
    _BUILD.get("git_dirty"),
    settings.answer_claim_guard_mode,
    settings.ask_synthesis_thinking,
    settings.answer_repair_mode,
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class IssueSummary(BaseModel):
    id: int
    date: date
    language: str
    numero: str | None
    title: str | None

    model_config = ConfigDict(from_attributes=True)


class DocumentSummary(BaseModel):
    id: int
    issue_id: int
    section: str | None
    ref: str | None
    conselleria: str | None
    title: str | None
    type: str | None
    pdf_url: str | None
    html_url: str | None

    model_config = ConfigDict(from_attributes=True)


class Turn(BaseModel):
    """One prior conversation message, supplied by the client (server is stateless)."""

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    debug: bool = False
    # Prior turns, oldest first. Client-owned; server keeps no session state. The
    # last `ask_history_max_turns` are used (older turns are dropped defensively).
    history: list[Turn] = Field(default_factory=list)

    def history_dicts(self) -> list[dict[str, str]]:
        """Last N turns as plain dicts for the graph state (most-recent-bounded)."""
        max_turns = max(0, int(getattr(settings, "ask_history_max_turns", 6) or 0))
        turns = self.history[-max_turns:] if max_turns else []
        return [{"role": t.role, "content": t.content} for t in turns]


class Citation(BaseModel):
    document_id: int
    title: str | None
    ref: str | None
    issue_date: date | None
    pdf_url: str | None
    html_url: str | None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    debug: dict | None = None


@app.get("/health")
def health():
    readiness = build_readiness_payload()
    payload: dict[str, object] = {
        "status": "ok",
        "freshness": readiness.get("freshness"),
        "startup_sync": readiness.get("startup_sync"),
        "build": _BUILD,
        "eval_settings": {
            "answer_claim_guard_mode": settings.answer_claim_guard_mode,
            "ask_synthesis_thinking": settings.ask_synthesis_thinking,
            "ask_synthesis_temperature": settings.ask_synthesis_temperature,
            "answer_repair_mode": settings.answer_repair_mode,
            "answer_validator_enabled": settings.answer_validator_enabled,
        },
    }
    if not readiness.get("ready"):
        payload["readiness"] = {
            "ready": readiness.get("ready"),
            "status": readiness.get("status"),
            "reason": readiness.get("reason"),
        }
    return payload


@app.on_event("startup")
def _startup_sync():
    try:
        start_startup_sync()
    except Exception:
        logger.exception("startup.sync.error")


@app.get("/ready")
def ready():
    return build_readiness_payload()


@app.get("/issues", response_model=list[IssueSummary])
def list_issues(db: Session = Depends(get_db)):
    issues = db.query(DogvIssue).order_by(DogvIssue.date.desc(), DogvIssue.language.asc()).all()
    return issues


@app.get("/issues/{issue_id}/documents", response_model=list[DocumentSummary])
def list_issue_documents(issue_id: int, db: Session = Depends(get_db)):
    issue = db.query(DogvIssue).filter(DogvIssue.id == issue_id).one_or_none()
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")

    docs = (
        db.query(DogvDocument)
        .filter(DogvDocument.issue_id == issue_id)
        .order_by(DogvDocument.id.asc())
        .all()
    )
    return docs


# Human-readable (Spanish) labels for each graph node, surfaced as live progress
# steps by the streaming endpoint. Nodes not listed here emit no stage event.
_STAGE_LABELS: dict[str, str] = {
    "analyze_intent": "Analizando la consulta",
    "temporal_guard": "Comprobando el marco temporal",
    "online_ingest": "Buscando documentos recientes",
    "retrieve_candidates": "Recuperando documentos del DOGV",
    "backfill": "Buscando en el archivo histórico del DOGV",
    "rerank_titles": "Seleccionando los documentos más relevantes",
    "read_docs": "Leyendo los documentos",
    "answer_node": "Redactando la respuesta",
}


def _build_ask_response(result: dict[str, Any], payload: AskRequest) -> dict[str, Any]:
    """Turn a finished graph state into the /ask response, persisting a trace.

    Shared by /ask and /ask/stream so both return the identical payload shape.
    """
    filters = result.get("filters")
    candidates = result.get("candidate_docs") or []
    fusion = [
        {
            "document_id": int(item.get("document_id")),
            "rrf_score": float(item.get("rrf_score") or 0.0),
            "issue_date": item.get("issue_date"),
            "title": item.get("title"),
        }
        for item in candidates
        if item.get("document_id") is not None
    ]
    store_trace(
        {
            "question": payload.question,
            "filters": filters.__dict__ if hasattr(filters, "__dict__") else filters,
            "plan": {"intent": result.get("intent")},
            "lanes": sorted(enabled_lanes(settings)),
            "fusion": {"candidates": fusion},
            "rerank": result.get("selected_doc_ids"),
            "evidence": result.get("evidence"),
            "answer": result.get("answer"),
        }
    )
    response: dict[str, Any] = {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
    if payload.debug:
        response["debug"] = {
            "intent": result.get("intent"),
            "candidate_docs": result.get("candidate_docs"),
            "evidence": result.get("evidence"),
            "profile": result.get("profile"),
            "read_payload": result.get("read_payload"),
        }
    return response


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if settings.demo_enforce_ready_gate:
        readiness = build_readiness_payload()
        if not readiness.get("ready"):
            raise HTTPException(status_code=503, detail=readiness)

    request_id = uuid.uuid4().hex[:8]
    history = payload.history_dicts()
    start = time.monotonic()
    logger.info(
        "ask.start req=%s chars=%s debug=%s history_turns=%s",
        request_id,
        len(payload.question),
        payload.debug,
        len(history),
    )
    try:
        result = graph.invoke(
            {
                "question": payload.question,
                "request_id": request_id,
                "debug": payload.debug,
                "history": history,
            }
        )
    except Exception:
        logger.exception("ask.error req=%s", request_id)
        raise
    finally:
        logger.info("ask.end req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
    return _build_ask_response(result, payload)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.post("/ask/stream")
def ask_stream(payload: AskRequest):
    """Server-Sent Events variant of /ask.

    Emits a `stage` event as each pipeline node completes (so the UI can show
    live progress instead of a silent spinner), then a single `result` event
    carrying the same payload as /ask, and finally a `done` event. On failure a
    single `error` event is emitted in-stream (the HTTP status is already 200 by
    the time streaming starts), except the readiness gate which 503s up front.
    """
    if settings.demo_enforce_ready_gate:
        readiness = build_readiness_payload()
        if not readiness.get("ready"):
            raise HTTPException(status_code=503, detail=readiness)

    request_id = uuid.uuid4().hex[:8]
    history = payload.history_dicts()

    def event_stream() -> Iterator[str]:
        start = time.monotonic()
        logger.info(
            "ask.stream.start req=%s chars=%s debug=%s history_turns=%s",
            request_id,
            len(payload.question),
            payload.debug,
            len(history),
        )
        final_state: dict[str, Any] = {}
        try:
            for mode, chunk in graph.stream(
                {
                    "question": payload.question,
                    "request_id": request_id,
                    "debug": payload.debug,
                    "history": history,
                },
                stream_mode=["updates", "values"],
            ):
                if mode == "values":
                    # Full state after each super-step; the last one is final.
                    final_state = chunk
                    continue
                # mode == "updates": {node_name: state_delta} for the node(s) that
                # just ran. Emit a friendly progress label for known nodes.
                for node in chunk or {}:
                    label = _STAGE_LABELS.get(node)
                    if label:
                        yield _sse("stage", {"node": node, "label": label})
            response = _build_ask_response(final_state, payload)
            yield _sse("result", response)
            yield _sse("done", {"elapsed": round(time.monotonic() - start, 2)})
        except Exception as exc:  # surface the failure in-stream, then log it
            logger.exception("ask.stream.error req=%s", request_id)
            yield _sse("error", {"message": "internal_error", "detail": str(exc)})
        finally:
            logger.info(
                "ask.stream.end req=%s elapsed=%.2fs",
                request_id,
                time.monotonic() - start,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
