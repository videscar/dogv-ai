from __future__ import annotations

from datetime import date
import logging
import time
import uuid
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from .config import enabled_lanes, get_settings
from .db import SessionLocal
from .models import DogvDocument, DogvIssue
from .traces import store_trace
from agent.graph import build_graph

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


class AskRequest(BaseModel):
    question: str
    debug: bool = False


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
    return {"status": "ok"}


@app.get("/issues", response_model=list[IssueSummary])
def list_issues(db: Session = Depends(get_db)):
    issues = (
        db.query(DogvIssue)
        .order_by(DogvIssue.date.desc(), DogvIssue.language.asc())
        .all()
    )
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


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    request_id = uuid.uuid4().hex[:8]
    start = time.monotonic()
    logger.info(
        "ask.start req=%s chars=%s debug=%s",
        request_id,
        len(payload.question),
        payload.debug,
    )
    result = None
    try:
        result = graph.invoke({"question": payload.question, "request_id": request_id})
    except Exception:
        logger.exception("ask.error req=%s", request_id)
        raise
    finally:
        logger.info("ask.end req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
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
    response = {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
    if payload.debug:
        response["debug"] = {
            "intent": result.get("intent"),
            "candidate_docs": result.get("candidate_docs"),
            "evidence": result.get("evidence"),
        }
    return response
