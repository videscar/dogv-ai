from __future__ import annotations

import logging
import time
from typing import Any

from agent.shared import QAState, return_with_profile
from api.answer import build_answer, no_evidence_answer
from api.config import get_settings
from api.db import SessionLocal
from api.dogv_urls import build_html_url, build_pdf_url
from api.models import DogvDocument, DogvIssue

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _map_language(intent_lang: str | None) -> str:
    if intent_lang == "ca":
        return "va_va"
    return "es_es"


def _parse_citation_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for key in ("doc_id", "document_id", "id"):
            if key in value:
                return _parse_citation_id(value.get(key))
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("doc_"):
            text = text[4:]
        if text.isdigit():
            return int(text)
    return None


def answer_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    intent = state.get("intent") or {}
    language = state.get("language") or _map_language(intent.get("language"))
    evidence = state.get("evidence") or []
    full_docs = state.get("full_docs") or []
    try:
        if not evidence and not full_docs:
            elapsed = time.monotonic() - start
            logger.info("answer.skip req=%s reason=no_evidence elapsed=%.2fs", request_id, elapsed)
            return return_with_profile(
                state,
                "answer",
                no_evidence_answer(language),
                elapsed_seconds=round(elapsed, 3),
                skipped=True,
                reason="no_evidence",
                citations=0,
                evidence=0,
                full_docs=0,
                repair_attempts=0,
                repair_success=False,
                fallback_reason="no_evidence",
            )
        result = build_answer(state["question"], language, evidence, full_docs=full_docs)
        answer = result.get("answer") or ""
        diagnostics = result.get("diagnostics") or {}
        cited_ids: set[int] = set()
        for raw in result.get("citations") or []:
            parsed = _parse_citation_id(raw)
            if parsed is not None:
                cited_ids.add(parsed)

        citations = []
        if not cited_ids and evidence:
            for item in evidence:
                parsed = _parse_citation_id(item)
                if parsed is not None:
                    cited_ids.add(parsed)
        if cited_ids:
            with SessionLocal() as db:
                docs = (
                    db.query(DogvDocument, DogvIssue)
                    .join(DogvIssue)
                    .filter(DogvDocument.id.in_(cited_ids))
                    .all()
                )
            for doc, issue in docs:
                citations.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "pdf_url": build_pdf_url(doc.pdf_url),
                        "html_url": build_html_url(doc.html_url),
                    }
                )

        elapsed = time.monotonic() - start
        logger.info(
            "answer.done req=%s citations=%s evidence=%s full_docs=%s elapsed=%.2fs",
            request_id,
            len(citations),
            len(evidence),
            len(full_docs),
            elapsed,
        )
        return return_with_profile(
            state,
            "answer",
            {"answer": answer, "citations": citations},
            elapsed_seconds=round(elapsed, 3),
            skipped=False,
            citations=len(citations),
            evidence=len(evidence),
            full_docs=len(full_docs),
            validator_triggered=bool(diagnostics.get("validation_errors_initial")),
            repair_attempts=int(diagnostics.get("repair_attempts") or 0),
            repair_success=bool(diagnostics.get("repair_success")),
            fallback_reason=diagnostics.get("fallback_reason"),
            validation_errors_initial=diagnostics.get("validation_errors_initial") or [],
            validation_errors_final=diagnostics.get("validation_errors_final") or [],
            deterministic_fix_applied=bool(diagnostics.get("deterministic_fix_applied")),
            deterministic_fix_types=diagnostics.get("deterministic_fix_types") or [],
            repair_skipped_reason=diagnostics.get("repair_skipped_reason"),
            rejected_answer=diagnostics.get("rejected_answer"),
        )
    except Exception:
        logger.exception("answer.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
