from __future__ import annotations

import logging
import time

from sqlalchemy import text as sa_text

from agent.shared import QAState, return_with_profile
from api.auto_ingest import ensure_range_ingested
from api.config import get_settings
from api.db import SessionLocal
from api.dogv_resolver import (
    Reference,
    corpus_like_patterns,
    parse_reference,
    resolve,
    _query_lang,
)

settings = get_settings()
logger = logging.getLogger("dogv.graph")

DEFAULT_LANGS = ["es_es", "va_va"]


def _reference_in_corpus(db, ref: Reference) -> bool:
    """True when the referenced disposition is already an originating doc in the DB."""
    patterns = corpus_like_patterns(ref)
    clause = " OR ".join(f"title ILIKE :p{i}" for i in range(len(patterns)))
    params = {f"p{i}": pat for i, pat in enumerate(patterns)}
    row = db.execute(
        sa_text(f"SELECT 1 FROM dogv_documents WHERE {clause} LIMIT 1"), params
    ).first()
    return row is not None


def _find_ondemand_doc_id(db, ref: Reference, fecha_pub) -> int | None:
    """The dogv_documents.id of the freshly-ingested principal disposition."""
    patterns = corpus_like_patterns(ref)
    clause = " OR ".join(f"d.title ILIKE :p{i}" for i in range(len(patterns)))
    params = {f"p{i}": pat for i, pat in enumerate(patterns)}
    params["fp"] = fecha_pub
    row = db.execute(
        sa_text(
            f"""
            SELECT d.id FROM dogv_documents d
            JOIN dogv_issues i ON d.issue_id = i.id
            WHERE i.date = :fp AND ({clause})
            ORDER BY d.id LIMIT 1
            """
        ),
        params,
    ).first()
    return int(row[0]) if row else None


def _pin_doc(db, doc_id: int) -> None:
    """Mark an on-demand-fetched historical doc so a future rolling-window
    eviction never drops it. Uses the existing doc_tags JSONB (no migration)."""
    db.execute(
        sa_text(
            "UPDATE dogv_documents "
            "SET doc_tags = COALESCE(doc_tags, '{}'::jsonb) || '{\"pinned\": true, \"ondemand\": true}'::jsonb "
            "WHERE id = :id"
        ),
        {"id": doc_id},
    )
    db.commit()


def backfill_node(state: QAState) -> QAState:
    """On-demand historical fetch.

    When the question names a specific disposition (e.g. "Decreto 185/2018") that
    is not in the corpus window, resolve it against the DOGV portal, ingest its
    publication-day issue via the normal pipeline, and loop back to retrieval so
    the now-present document can be cited. Targeted replacement for the old blunt
    month-walk backfill.
    """
    start = time.monotonic()
    request_id = state.get("request_id")

    def _skip(reason: str) -> QAState:
        elapsed = time.monotonic() - start
        logger.info("backfill.skip req=%s reason=%s elapsed=%.2fs", request_id, reason, elapsed)
        return return_with_profile(
            state, "backfill", {"backfill_attempted": True},
            elapsed_seconds=round(elapsed, 3), status="skipped", reason=reason,
        )

    try:
        if not settings.backfill_enabled:
            return _skip("disabled")
        if state.get("backfill_attempted"):
            return _skip("already_attempted")

        question = state.get("question") or ""
        ref = parse_reference(question)
        if ref is None:
            return _skip("no_reference")

        with SessionLocal() as db:
            if _reference_in_corpus(db, ref):
                return _skip("already_in_corpus")

        resolved = resolve(ref, _query_lang(question))
        if resolved is None:
            return _skip("unresolved")

        # Ingest the disposition's publication-day issue (both languages) through
        # the normal pipeline: insert -> extract HTML body -> classify -> chunk+embed.
        ensure_range_ingested(resolved.fecha_publicacion, resolved.fecha_publicacion, DEFAULT_LANGS)

        ondemand_doc_id = None
        with SessionLocal() as db:
            ondemand_doc_id = _find_ondemand_doc_id(db, ref, resolved.fecha_publicacion)
            if ondemand_doc_id is not None:
                _pin_doc(db, ondemand_doc_id)

        elapsed = time.monotonic() - start
        logger.info(
            "backfill.fetched req=%s ref=%s disp_id=%s pub=%s doc_id=%s elapsed=%.2fs",
            request_id, ref.num_year, resolved.disposicion_id,
            resolved.fecha_publicacion, ondemand_doc_id, elapsed,
        )
        return return_with_profile(
            state,
            "backfill",
            {
                "backfill_attempted": True,
                "ondemand_doc_id": ondemand_doc_id,
                "ondemand_ref": ref.num_year,
            },
            elapsed_seconds=round(elapsed, 3),
            status="fetched",
            ref=ref.num_year,
            disposicion_id=resolved.disposicion_id,
            fecha_publicacion=str(resolved.fecha_publicacion),
            doc_id=ondemand_doc_id,
        )
    except Exception:
        logger.exception("backfill.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start)
        # Never fail the whole answer because the on-demand fetch broke.
        return _skip("error")
