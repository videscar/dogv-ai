"""Recover dispositions dropped on DOGV "bis"-collision dates.

When a date publishes both an ordinary and a *bis* (extraordinary) edition, the
portal's date-keyed sumario JSON (`GET /dogv-portal/dogv?date=`) returns ONLY the
bis edition, so the ordinary edition's ~50 dispositions are never ingested and no
gap is recorded (the pipeline thinks the small bis issue is complete).

This script recovers the missing edition:
  1. Enumerate every disposition signatura from BOTH edition sumario PDFs
     (the PDFs reliably carry one `resultat-dogv?signatura=` link per item).
  2. For signaturas absent from the corpus, resolve signatura -> disposition id
     via the full-text search (matching `codigoInsercion`), then ingest each via
     the normal on-demand path (ingest_one_disposicion: create row, extract body,
     classify, chunk+embed). Idempotent and resumable.

Usage:
    python scripts/recover_bis_editions.py --date 2025-10-31           # one date
    python scripts/recover_bis_editions.py --date 2025-10-31 --dry-run # report only
    python scripts/recover_bis_editions.py --all                       # full backfill
    python scripts/recover_bis_editions.py --all --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from sqlalchemy import text as sa_text

sys.path.insert(0, ".")

from api.auto_ingest import ingest_one_disposicion
from api.bis_editions import (
    edition_pdf_urls,
    existing_refs,
    fetch_pdf,
    parse_pdf_titles,
    resolve_signatura_ids,
)
from api.db import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recover_bis")


def recover_date(date_iso: str, numero: str, lang: str, dry_run: bool) -> dict:
    titles: dict[str, str] = {}
    for url in edition_pdf_urls(date_iso, numero, lang):
        content = fetch_pdf(url)
        if content is None:
            continue
        titles.update(parse_pdf_titles(content))
    target_sigs = set(titles)

    have = existing_refs(target_sigs, lang)
    missing = target_sigs - have
    stats = {
        "date": date_iso, "numero": numero, "lang": lang,
        "in_pdfs": len(target_sigs), "already": len(have), "missing": len(missing),
        "resolved": 0, "ingested": 0, "unresolved": [], "failed": [],
    }
    if not missing:
        logger.info("recover %s/%s %s: complete (pdf=%d already=%d)",
                    date_iso, numero, lang, len(target_sigs), len(have))
        return stats

    resolved = resolve_signatura_ids(missing, titles, lang)
    stats["resolved"] = len(resolved)
    stats["unresolved"] = sorted(missing - set(resolved))
    logger.info("recover %s/%s %s: pdf=%d already=%d missing=%d resolved=%d unresolved=%d",
                date_iso, numero, lang, len(target_sigs), len(have),
                len(missing), len(resolved), len(stats["unresolved"]))
    if dry_run:
        return stats

    for sig, disp_id in resolved.items():
        try:
            doc_id = ingest_one_disposicion(disp_id, lang)
            if doc_id:
                stats["ingested"] += 1
            else:
                stats["failed"].append(sig)
        except Exception:
            logger.exception("ingest_fail sig=%s disp_id=%s lang=%s", sig, disp_id, lang)
            stats["failed"].append(sig)
    logger.info("recover %s/%s %s: ingested=%d failed=%d",
                date_iso, numero, lang, stats["ingested"], len(stats["failed"]))
    return stats


def _bis_issues() -> list[tuple[str, str, str]]:
    """(date_iso, numero, language) for every bis-truncated issue in the window."""
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                "SELECT i.date::text, i.numero, i.language FROM dogv_issues i "
                "WHERE (i.raw_json->>'esBis')='true' AND i.numero IS NOT NULL "
                "AND i.date >= current_date - interval '24 months' "
                "ORDER BY i.date, i.language"
            )
        ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="single issue date YYYY-MM-DD")
    ap.add_argument("--lang", default=None, help="es_es / va_va (default: both)")
    ap.add_argument("--all", action="store_true", help="all bis-truncated issues in window")
    ap.add_argument("--workers", type=int, default=1, help="parallel issues (--all path)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        targets = _bis_issues()
    elif args.date:
        langs = [args.lang] if args.lang else ["es_es", "va_va"]
        with SessionLocal() as db:
            targets = []
            for lang in langs:
                row = db.execute(
                    sa_text("SELECT numero FROM dogv_issues WHERE date=:d AND language=:l "
                            "AND numero IS NOT NULL"),
                    {"d": args.date, "l": lang},
                ).first()
                if row:
                    targets.append((args.date, row[0], lang))
    else:
        ap.error("pass --date or --all")

    logger.info("recover start: %d issue(s) workers=%d dry_run=%s",
                len(targets), args.workers, args.dry_run)
    totals = {"in_pdfs": 0, "already": 0, "missing": 0, "resolved": 0, "ingested": 0,
              "unresolved": 0, "failed": 0}
    t0 = time.time()

    def _one(target):
        date_iso, numero, lang = target
        try:
            return recover_date(date_iso, numero, lang, args.dry_run)
        except Exception:
            logger.exception("recover_issue_error %s/%s %s", date_iso, numero, lang)
            return None

    def _acc(s):
        if not s:
            return
        for k in ("in_pdfs", "already", "missing", "resolved", "ingested"):
            totals[k] += s[k]
        totals["unresolved"] += len(s["unresolved"])
        totals["failed"] += len(s["failed"])

    if args.workers > 1:
        # Issue-level parallelism: distinct dates don't collide (extract/classify/
        # chunk are scoped per date / per doc-id). Bounded to spare the embed+LLM.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for s in ex.map(_one, targets):
                _acc(s)
    else:
        for target in targets:
            _acc(_one(target))
    logger.info("recover DONE in %.0fs totals=%s", time.time() - t0, totals)


if __name__ == "__main__":
    main()
