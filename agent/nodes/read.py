from __future__ import annotations

import logging
import re
import time
from typing import Any

from sqlalchemy import text as sa_text

from agent.shared import (
    QAState,
    best_snippet,
    coverage_score,
    estimate_tokens,
    extract_keywords_simple,
    return_with_profile,
    rrf_margin_ratio,
)
from api.config import get_settings
from api.db import SessionLocal
from api.intent import needs_amounts, needs_eligibility
from api.models import DogvDocument, DogvIssue
from api.reader import extract_evidence
from api.retrieval import top_chunks_for_docs
from api.taxonomy import is_base_like

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def read_docs_node(state: QAState) -> QAState:
    start = time.monotonic()
    request_id = state.get("request_id")
    doc_ids = state.get("selected_doc_ids") or []
    if not doc_ids:
        elapsed = time.monotonic() - start
        logger.info("read.skip req=%s reason=no_docs elapsed=%.2fs", request_id, elapsed)
        return return_with_profile(
            state,
            "read",
            {"evidence": []},
            elapsed_seconds=round(elapsed, 3),
            skipped=True,
            reason="no_docs",
            docs=0,
            chunks=0,
            tokens_est=0,
            full_docs=0,
            full_tokens_est=0,
        )
    question = state["question"]
    keywords = extract_keywords_simple(question)
    eligibility_required = needs_eligibility(question)
    amount_extra = getattr(settings, "ask_read_amount_docs", 1)
    read_max_docs = getattr(settings, "ask_read_max_docs", 3)
    rerank_cap = getattr(settings, "ask_rerank_max_candidates", read_max_docs)
    read_max_docs = max(read_max_docs, min(len(doc_ids), rerank_cap))
    expand_ratio = getattr(settings, "ask_rrf_expand_margin_ratio", 0.12)
    expand_probe = getattr(settings, "ask_rrf_margin_probe", 5)
    expand_docs = getattr(settings, "ask_read_expand_docs", 2)
    candidate_docs = state.get("candidate_docs") or []
    if candidate_docs and rrf_margin_ratio(candidate_docs, probe=expand_probe) < expand_ratio:
        read_max_docs += expand_docs
    if read_max_docs > len(doc_ids):
        read_max_docs = len(doc_ids)
    doc_ids = doc_ids[:read_max_docs]
    top_chunks = state.get("top_chunks") or {}
    coverage_extra = getattr(settings, "ask_read_coverage_docs", 2)
    if coverage_extra > 0 and candidate_docs and keywords:
        scored: list[tuple[int, int]] = []
        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            text = (item.get("title") or "")
            chunk_list = top_chunks.get(doc_id) or []
            if chunk_list:
                text = f"{text} {best_snippet(question, chunk_list)}"
            score = coverage_score(text, keywords)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        extras = [doc_id for _, doc_id in scored[:coverage_extra]]
        if extras:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + extras:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged
            max_with_extra = min(len(doc_ids), read_max_docs + coverage_extra)
            doc_ids = doc_ids[:max_with_extra]
    eligibility_extra = getattr(settings, "ask_read_eligibility_docs", 1)
    if eligibility_required and needs_amounts(question):
        eligibility_extra = max(eligibility_extra, 3)
    if eligibility_required and eligibility_extra > 0 and candidate_docs:
        base_scored: list[tuple[int, int]] = []
        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            if is_base_like(item):
                text = (item.get("title") or "")
                chunk_list = top_chunks.get(doc_id) or []
                if chunk_list:
                    text = f"{text} {best_snippet(question, chunk_list)}"
                score = coverage_score(text, keywords) if keywords else 0
                base_scored.append((score, doc_id))
        base_scored.sort(key=lambda item: item[0], reverse=True)
        base_candidates = [doc_id for _, doc_id in base_scored[:eligibility_extra]]
        if base_candidates:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + base_candidates:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged

    if amount_extra > 0 and candidate_docs and needs_amounts(question):
        scored: list[tuple[int, int]] = []

        def _amount_score(text: str) -> int:
            return len(re.findall(r"\b\d[\d.,]*\b", text))

        for item in candidate_docs:
            doc_id = int(item["document_id"])
            if doc_id in doc_ids:
                continue
            chunk_list = top_chunks.get(doc_id) or []
            text = " ".join((c.get("text") or "") for c in chunk_list)
            if not text:
                text = (item.get("summary") or item.get("text") or item.get("title") or "")
            score = _amount_score(text)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        extras = [doc_id for _, doc_id in scored[:amount_extra]]
        if extras:
            seen: set[int] = set()
            merged: list[int] = []
            for doc_id in doc_ids + extras:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                merged.append(doc_id)
            doc_ids = merged

    try:
        top_chunks = state.get("top_chunks") or {}
        chunk_candidates = state.get("chunk_candidates") or []
        query_embedding = state.get("query_embedding")

        with SessionLocal() as db:
            docs = (
                db.query(DogvDocument, DogvIssue)
                .join(DogvIssue)
                .filter(DogvDocument.id.in_(doc_ids))
                .all()
            )
            fallback_chunks = {}
            extra_chunks: dict[int, list[dict[str, Any]]] = {}
            if query_embedding:
                fallback_chunks = top_chunks_for_docs(
                    db,
                    query_embedding,
                    doc_ids,
                    per_doc=getattr(settings, "ask_chunks_per_doc", 4),
                )
            extra_limit = getattr(settings, "ask_chunks_per_doc", 4)
            if extra_limit > 0:
                rows = db.execute(
                    sa_text(
                        """
                        SELECT document_id, chunk_index, text
                        FROM (
                            SELECT document_id, chunk_index, text,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY document_id
                                       ORDER BY chunk_index
                                   ) AS rn
                            FROM rag_chunk
                            WHERE document_id = ANY(:doc_ids)
                        ) ranked
                        WHERE rn <= :extra_limit
                        ORDER BY document_id, chunk_index
                        """
                    ),
                    {"doc_ids": doc_ids, "extra_limit": extra_limit},
                ).mappings().all()
                for row in rows:
                    extra_chunks.setdefault(int(row["document_id"]), []).append(
                        {
                            "chunk_index": int(row["chunk_index"]),
                            "text": row["text"],
                        }
                    )

        docs_by_id = {doc.id: (doc, issue) for doc, issue in docs}

        def _build_payload(
            target_ids: list[int],
            chunks_map: dict[int, list[dict[str, Any]]],
            extra_chunks: dict[int, list[dict[str, Any]]],
            ordered_chunks: dict[int, list[dict[str, Any]]],
        ):
            payload: list[dict[str, Any]] = []
            target_chunks = getattr(settings, "ask_chunks_per_doc", 4)
            chunk_max_chars = getattr(settings, "ask_chunk_max_chars", 1200)
            fallback_chars = getattr(settings, "ask_doc_fallback_chars", 12000)
            for doc_id in target_ids:
                row = docs_by_id.get(doc_id)
                if not row:
                    continue
                doc, issue = row
                primary = chunks_map.get(doc.id, [])
                fallback = extra_chunks.get(doc.id, [])
                merged = primary if primary else fallback
                if primary and fallback and len(primary) < target_chunks:
                    seen = {c.get("chunk_id") for c in primary}
                    merged = primary + [c for c in fallback if c.get("chunk_id") not in seen]
                if len(merged) < target_chunks:
                    extra = ordered_chunks.get(doc.id, [])
                    existing_text = {c.get("text") for c in merged if c.get("text")}
                    merged = merged + [c for c in extra if c.get("text") not in existing_text]
                chunks = [c["text"][:chunk_max_chars] for c in merged[:target_chunks]]
                if not chunks and doc.text:
                    chunks = [doc.text[:fallback_chars]]
                payload.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "chunks": chunks,
                    }
                )
            return payload

        def _build_full_docs(target_ids: list[int]) -> list[dict[str, Any]]:
            max_full_docs = int(getattr(settings, "full_doc_max_docs", 0) or 0)
            if max_full_docs <= 0:
                return []
            full_docs_local = []
            total_chars = 0
            for doc_id in target_ids:
                row = docs_by_id.get(doc_id)
                if not row:
                    continue
                doc, issue = row
                if not doc.text:
                    continue
                text = doc.text
                if len(text) > settings.full_doc_max_chars:
                    continue
                if total_chars + len(text) > settings.full_doc_total_chars:
                    break
                full_docs_local.append(
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "ref": doc.ref,
                        "issue_date": issue.date.isoformat() if issue.date else None,
                        "text": text,
                    }
                )
                total_chars += len(text)
                if len(full_docs_local) >= max_full_docs:
                    break
            return full_docs_local

        confidence_min = getattr(settings, "ask_doc_confidence_min", 0.06)
        top_score = float(candidate_docs[0].get("rrf_score", 0.0)) if candidate_docs else 0.0
        high_confidence = top_score >= confidence_min

        docs_payload = _build_payload(doc_ids, top_chunks, fallback_chunks, extra_chunks)
        evidence = extract_evidence(state["question"], docs_payload, full_docs=None)
        full_docs = []
        if evidence or high_confidence:
            full_docs = _build_full_docs(doc_ids)
            if not evidence and full_docs:
                evidence = extract_evidence(state["question"], docs_payload, full_docs=full_docs)
                if not evidence:
                    full_docs = []
        chunk_count = sum(len(doc.get("chunks") or []) for doc in docs_payload)
        char_count = sum(len(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
        token_est = sum(estimate_tokens(chunk) for doc in docs_payload for chunk in (doc.get("chunks") or []))
        full_token_est = sum(estimate_tokens(doc.get("text") or "") for doc in full_docs)
        elapsed = time.monotonic() - start
        logger.info(
            "read.done req=%s docs=%s chunks=%s chars=%s tokens~=%s evidence=%s full_docs=%s full_chars=%s full_tokens~=%s elapsed=%.2fs",
            request_id,
            len(doc_ids),
            chunk_count,
            char_count,
            token_est,
            len(evidence),
            len(full_docs),
            sum(len(doc.get("text") or "") for doc in full_docs),
            full_token_est,
            elapsed,
        )
        return return_with_profile(
            state,
            "read",
            {"evidence": evidence, "full_docs": full_docs},
            elapsed_seconds=round(elapsed, 3),
            skipped=False,
            docs=len(doc_ids),
            chunks=chunk_count,
            chars=char_count,
            tokens_est=token_est,
            evidence=len(evidence),
            full_docs=len(full_docs),
            full_chars=sum(len(doc.get("text") or "") for doc in full_docs),
            full_tokens_est=full_token_est,
        )
    except Exception:
        logger.exception("read.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        raise
