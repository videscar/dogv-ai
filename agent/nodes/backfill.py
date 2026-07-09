from __future__ import annotations

import logging
import time

from sqlalchemy import text as sa_text

from agent.shared import QAState, return_with_profile
from api.auto_ingest import ingest_one_disposicion
from api.config import get_settings
from api.db import SessionLocal
from api.dogv_resolver import (
    Reference,
    _query_lang,
    _strip_accents,
    corpus_like_patterns,
    infer_reference_from_corpus,
    parse_reference,
    resolve,
)

settings = get_settings()
logger = logging.getLogger("dogv.graph")


# es<->va month forms, so a date stated in one language still matches the title's
# other-language twin in the corpus (titles are stored per language).
_MONTH_FORMS: dict[str, tuple[str, ...]] = {
    "enero": ("enero", "gener"),
    "gener": ("enero", "gener"),
    "febrero": ("febrero", "febrer"),
    "febrer": ("febrero", "febrer"),
    "marzo": ("marzo", "març", "marc"),
    "març": ("marzo", "març", "marc"),
    "marc": ("marzo", "març", "marc"),
    "abril": ("abril",),
    "mayo": ("mayo", "maig"),
    "maig": ("mayo", "maig"),
    "junio": ("junio", "juny"),
    "juny": ("junio", "juny"),
    "julio": ("julio", "juliol"),
    "juliol": ("julio", "juliol"),
    "agosto": ("agosto", "agost"),
    "agost": ("agosto", "agost"),
    "septiembre": ("septiembre", "setiembre", "setembre"),
    "setiembre": ("septiembre", "setiembre", "setembre"),
    "setembre": ("septiembre", "setiembre", "setembre"),
    "octubre": ("octubre",),
    "noviembre": ("noviembre", "novembre"),
    "novembre": ("noviembre", "novembre"),
    "diciembre": ("diciembre", "desembre"),
    "desembre": ("diciembre", "desembre"),
}


def _date_patterns(ref: Reference) -> list[str]:
    """ILIKE patterns for the disposition date stated in the question (e.g. both
    '%30 de octubre%' and '%30 d'octubre%'), or [] when no date was parsed."""
    if not ref.date_day or not ref.date_month:
        return []
    months = _MONTH_FORMS.get(ref.date_month, (ref.date_month,))
    pats: list[str] = []
    for mo in months:
        pats.append(f"% {ref.date_day} de {mo}%")
        pats.append(f"% {ref.date_day} d'{mo}%")
    return pats


def _reference_corpus_doc_ids(db, ref: Reference, limit: int = 4) -> list[int]:
    """Ids of originating docs whose title carries this disposition's tipo+N/YYYY.

    "Orden N/YYYY" is NOT a unique key — each conselleria numbers independently —
    so when the question also states the disposition date, require the corpus
    title to carry that date too; otherwise a same-numbered order from another
    body masks a genuinely-missing one and suppresses the on-demand fetch.
    A false negative here is harmless: it only triggers an idempotent re-fetch.

    Lowest id first so es/va twins resolve to a stable principal."""
    patterns = corpus_like_patterns(ref)
    clause = "(" + " OR ".join(f"title ILIKE :p{i}" for i in range(len(patterns))) + ")"
    params: dict = {f"p{i}": pat for i, pat in enumerate(patterns)}
    date_pats = _date_patterns(ref)
    if date_pats:
        clause += " AND (" + " OR ".join(f"title ILIKE :d{i}" for i in range(len(date_pats))) + ")"
        params.update({f"d{i}": pat for i, pat in enumerate(date_pats)})
    params["lim"] = limit
    rows = db.execute(
        sa_text(f"SELECT id FROM dogv_documents WHERE {clause} ORDER BY id LIMIT :lim"),
        params,
    ).all()
    return [int(r[0]) for r in rows]


def _resolved_matches_topic(titulo: str, topic_terms: list[str]) -> bool:
    """True when the resolved disposition's title carries at least one of the
    query's topic terms — a guard so an inferred principal that resolves to an
    unrelated law is never fetched/cited."""
    hay = _strip_accents((titulo or "").lower())
    return any(_strip_accents(t) in hay for t in topic_terms if len(t) >= 4)


def _reference_in_corpus(db, ref: Reference) -> bool:
    """True when the referenced disposition is already an originating doc in the DB."""
    return bool(_reference_corpus_doc_ids(db, ref, limit=1))


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
            state,
            "backfill",
            {"backfill_attempted": True},
            elapsed_seconds=round(elapsed, 3),
            status="skipped",
            reason=reason,
        )

    try:
        if not settings.backfill_enabled:
            return _skip("disabled")
        if state.get("backfill_attempted"):
            return _skip("already_attempted")

        question = state.get("question") or ""
        ref = parse_reference(question)
        # When the question names a norm by type+topic but no number ("la Ley de
        # Transparencia"), infer its N/YYYY from how the in-window corpus names it,
        # so the fetch path can recover a foundational law that predates the window.
        inferred = False
        with SessionLocal() as db:
            if ref is None:
                if settings.infer_named_norm_from_corpus_enabled:
                    ref = infer_reference_from_corpus(db, question)
                    inferred = ref is not None
                if ref is None:
                    return _skip("no_reference")
            corpus_ids = _reference_corpus_doc_ids(db, ref, limit=4)
        # An inferred principal carries no number in the question for the answer node
        # to match against; hand it the resolved ref so it can still force-cite it.
        norm_target_ref = {"tipo": ref.tipo, "num_year": ref.num_year} if inferred else None
        if corpus_ids:
            # The norm is already in the corpus. Retrieval+rerank are LLM-driven and
            # non-deterministic, so the exact norm sometimes loses its candidate slot
            # to topical same-number/same-subject siblings — the same question then
            # cites the gold on one run and misses it on the next. Pin it into the
            # read set so the norm-target citation guarantee fires deterministically.
            # Only pin an UNAMBIGUOUS principal: a lone corpus match, or — for Orden
            # numbers that repeat across consellerias — one the question's date already
            # narrowed to (its es/va twins share that date). Otherwise stay hands-off.
            pin_ids = corpus_ids if (len(corpus_ids) == 1 or _date_patterns(ref)) else []
            elapsed = time.monotonic() - start
            logger.info(
                "backfill.skip req=%s reason=already_in_corpus pin=%s elapsed=%.2fs",
                request_id,
                pin_ids,
                elapsed,
            )
            return return_with_profile(
                state,
                "backfill",
                {
                    "backfill_attempted": True,
                    "norm_pin_doc_ids": pin_ids,
                    "norm_target_ref": norm_target_ref,
                },
                elapsed_seconds=round(elapsed, 3),
                status="skipped",
                reason="already_in_corpus",
                pin=pin_ids,
            )

        lang = _query_lang(question)
        resolved = resolve(ref, lang)
        if resolved is None:
            return _skip("unresolved")
        # An inferred principal is a guess from corpus titles — confirm the resolved
        # disposition actually carries the query's topic before fetching/citing it,
        # so a mis-inference never force-cites the wrong law.
        if inferred and not _resolved_matches_topic(resolved.titulo, ref.topic_terms):
            return _skip("inferred_topic_mismatch")

        # Ingest just this one disposition (issue-day ingest would fetch ~55 doc
        # bodies at ~5s each): create the row, extract its body, classify, embed.
        ondemand_doc_id = ingest_one_disposicion(resolved.disposicion_id, lang)
        if ondemand_doc_id is not None:
            with SessionLocal() as db:
                _pin_doc(db, ondemand_doc_id)

        elapsed = time.monotonic() - start
        logger.info(
            "backfill.fetched req=%s ref=%s disp_id=%s pub=%s doc_id=%s elapsed=%.2fs",
            request_id,
            ref.num_year,
            resolved.disposicion_id,
            resolved.fecha_publicacion,
            ondemand_doc_id,
            elapsed,
        )
        return return_with_profile(
            state,
            "backfill",
            {
                "backfill_attempted": True,
                "ondemand_doc_id": ondemand_doc_id,
                "ondemand_ref": ref.num_year,
                # Pin the freshly-fetched norm into the read set: after the loop back
                # to retrieval it must not be re-evicted by the same non-deterministic
                # ranking that made it absent in the first place.
                "norm_pin_doc_ids": [ondemand_doc_id] if ondemand_doc_id is not None else [],
                "norm_target_ref": norm_target_ref,
            },
            elapsed_seconds=round(elapsed, 3),
            status="fetched",
            ref=ref.num_year,
            disposicion_id=resolved.disposicion_id,
            fecha_publicacion=str(resolved.fecha_publicacion),
            doc_id=ondemand_doc_id,
        )
    except Exception:
        logger.exception(
            "backfill.error req=%s elapsed=%.2fs", request_id, time.monotonic() - start
        )
        # Never fail the whole answer because the on-demand fetch broke.
        return _skip("error")
