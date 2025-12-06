import re
from datetime import date
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from .config import get_settings
from .db import SessionLocal
from .llm_client import generate_answer_with_ollama
from .models import DogvIssue, DogvDocument

settings = get_settings()

app = FastAPI(title="DOGV AI – Local Legal Assistant")


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

    class Config:
        orm_mode = True


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

    class Config:
        orm_mode = True


# simple stopwords for ES + VA + domain noise
STOPWORDS = {
    # Spanish
    "para", "sobre", "donde", "dónde", "cuando", "cuándo", "cual", "cuál",
    "cuales", "cuáles", "como", "cómo", "que", "qué", "hay", "haya",
    "este", "esta", "estos", "estas", "aqui", "aquí", "alli", "allí",
    "entre", "tras", "segun", "según", "dentro", "fuera", "hacia", "desde",
    "cualquiera", "algunos", "algunas", "varios", "varias",

    # Valencià / català
    "quines", "quin", "quina", "quins", "quan", "on", "com",
    "aquest", "aquesta", "aquests", "aquestes", "aquell", "aquella",

    # Domain-very-common
    "comunitat", "valenciana", "generalitat", "valencia", "alicante",
    "castello", "castellon", "espanya", "españa",
}


def _tokenize_query(q: str) -> list[str]:
    raw = re.findall(r"\w+", q.lower())
    tokens: list[str] = []
    for t in raw:
        if len(t) < 4:
            continue
        if t in STOPWORDS:
            continue
        tokens.append(t)
    if not tokens:
        tokens = [t for t in raw if len(t) >= 3]
    return tokens


class SearchRequest(BaseModel):
    q: str
    since_date: date | None = None
    until_date: date | None = None
    section: str | None = None
    conselleria: str | None = None
    limit: int = 20


class SearchHit(BaseModel):
    issue_id: int
    issue_date: date
    issue_numero: str | None
    section: str | None
    conselleria: str | None
    title: str | None
    pdf_url: str | None
    score: float
    snippet: str | None

    class Config:
        orm_mode = True


class AskRequest(BaseModel):
    question: str
    language: str | None = "auto"
    mode: str | None = "auto"
    since_date: date | None = None

class AskResponse(BaseModel):
    answer: str
    used_documents: List[SearchHit]

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


def _make_snippet(text: str | None, q: str, max_len: int = 260) -> str | None:
    if not text:
        return None

    text_lower = text.lower()
    q_lower = q.lower().strip()
    if not q_lower:
        return text[:max_len]

    idx = text_lower.find(q_lower)
    if idx == -1:
        return text[:max_len]

    start = max(idx - max_len // 3, 0)
    end = min(start + max_len, len(text))
    snippet = text[start:end].strip()

    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _run_search(db: Session, req: SearchRequest) -> list[SearchHit]:
    q = (req.q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' must not be empty")

    tokens = _tokenize_query(q)
    if not tokens:
        raise HTTPException(status_code=400, detail="No usable tokens extracted from query")

    q_lower = q.lower()

    # Domain cues
    want_aids = any(t.startswith("ayud") or t.startswith("ajud") for t in tokens)
    want_calls = any("convocatori" in t for t in tokens)
    want_police = any("polici" in t for t in tokens)

    query = (
        db.query(DogvDocument, DogvIssue)
        .join(DogvIssue, DogvDocument.issue_id == DogvIssue.id)
    )

    if req.since_date:
        query = query.filter(DogvIssue.date >= req.since_date)
    if req.until_date:
        query = query.filter(DogvIssue.date <= req.until_date)

    if req.section:
        query = query.filter(DogvDocument.section.ilike(f"%{req.section}%"))
    if req.conselleria:
        query = query.filter(DogvDocument.conselleria.ilike(f"%{req.conselleria}%"))

    conditions = []
    for t in tokens:
        like_t = f"%{t}%"
        conditions.append(DogvDocument.title.ilike(like_t))
        conditions.append(DogvDocument.conselleria.ilike(like_t))
        conditions.append(DogvDocument.text.ilike(like_t))

    query = query.filter(or_(*conditions))

    candidates = query.all()

    hits: list[SearchHit] = []

    for doc, issue in candidates:
        title = doc.title or ""
        cons = doc.conselleria or ""
        text = getattr(doc, "text", None) or ""

        title_lower = title.lower()
        cons_lower = cons.lower()
        text_lower = text.lower()

        lower_all = f"{title_lower} {cons_lower} {text_lower}"

        matched_tokens: set[str] = set()
        title_hits = 0
        cons_hits = 0
        text_hits = 0

        for t in tokens:
            if t in title_lower:
                matched_tokens.add(t)
                title_hits += 1
            elif t in cons_lower:
                matched_tokens.add(t)
                cons_hits += 1
            elif t in text_lower:
                matched_tokens.add(t)
                text_hits += 1

        base = len(matched_tokens) * 1.5 + title_hits * 0.7 + cons_hits * 0.4 + text_hits * 0.2

        if want_aids:
            if "ayud" in lower_all or "ajud" in lower_all:
                base += 5.0

        if want_calls:
            if "convocatori" in lower_all:
                base += 3.0

        if want_police:
            if "polici" in lower_all:
                base += 3.0

        base += issue.date.toordinal() / 365000.0

        snippet = _make_snippet(text, q)

        hits.append(
            SearchHit(
                issue_id=issue.id,
                issue_date=issue.date,
                issue_numero=issue.numero,
                section=doc.section,
                conselleria=doc.conselleria,
                title=doc.title,
                pdf_url=doc.pdf_url,
                score=base,
                snippet=snippet,
            )
        )

    hits.sort(key=lambda h: (h.score, h.issue_date), reverse=True)
    return hits


@app.post("/search", response_model=List[SearchHit])
def search_documents(req: SearchRequest, db: Session = Depends(get_db)):
    hits = _run_search(db, req)
    return hits[: req.limit]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    search_req = SearchRequest(
        q=req.question,
        since_date=req.since_date,
        limit=settings.ask_max_docs,
    )
    hits = _run_search(db, search_req)
    top_hits = hits[: settings.ask_max_docs]

    hits_for_llm = [h.model_dump() for h in top_hits]

    answer = generate_answer_with_ollama(
        question=req.question,
        language=req.language,
        hits=hits_for_llm,
        model=settings.ollama_model,
    )

    return AskResponse(
        answer=answer,
        used_documents=top_hits,
    )
