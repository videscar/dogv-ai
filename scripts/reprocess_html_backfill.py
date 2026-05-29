"""
One-time backfill: re-extract every document with the HTML-first method, then
re-classify and re-chunk/re-embed.

All 28.9k documents currently have PDF-sourced text. This switches ~90% of them
to the cleaner HTML body (and keeps PDF for the annex minority), which changes
their chunks and therefore requires a full re-chunk + re-embed.

The dominant cost is the portal HTML fetch (~5 s/doc), so fetching is done
concurrently per issue-date; the DB writes and embedding stay sequential. Work
is checkpointed by issue-date and is resumable: an interrupted run re-does only
the in-progress date.

Pipeline servers (chat for classify, embed for chunks) must be up for a real
run. --dry-run needs neither (it only fetches + reports the source decision).

Usage:
    # pilot: one month, see the html/pdf split without writing anything
    python scripts/reprocess_html_backfill.py --start 2026-05-01 --end 2026-05-29 --dry-run

    # real run over a range (writes to DB; needs servers up)
    python scripts/reprocess_html_backfill.py --start 2026-05-01 --end 2026-05-29

    # full corpus (resumable; safe to re-run after interruption)
    python scripts/reprocess_html_backfill.py --all
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session, load_only

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.dogv_urls import build_pdf_url
from api.models import DogvDocument, DogvIssue
from scripts.build_chunks import build_chunks_for_range
from scripts.classify_documents import classify_range
from scripts.download_assets import build_pdf_path, download_pdf
from scripts.extract_documents import _delete_rag_rows, _invalidate_doc_for_reindex
from scripts.extract_text import (
    _needs_pdf,
    fetch_disposicion_body,
    resolve_document_text,
)

CHECKPOINT = Path("data/reprocess_html_checkpoint.json")
_DOC_COLUMNS = (
    DogvDocument.id,
    DogvDocument.pdf_url,
    DogvDocument.raw_json,
    DogvDocument.title,
    DogvDocument.conselleria,
    DogvDocument.section,
    DogvDocument.text,
    DogvDocument.text_source,
    DogvDocument.text_updated_at,
    DogvDocument.doc_kind,
    DogvDocument.doc_subkind,
    DogvDocument.doc_kind_confidence,
    DogvDocument.doc_tags,
)


def load_checkpoint() -> Optional[date]:
    if not CHECKPOINT.exists():
        return None
    try:
        raw = json.loads(CHECKPOINT.read_text())
        return date.fromisoformat(raw["last_done"])
    except Exception:
        return None


def save_checkpoint(d: date) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps({"last_done": d.isoformat()}))


def target_dates(db: Session, start: Optional[date], end: Optional[date]) -> list[date]:
    q = db.query(DogvIssue.date).distinct()
    if start:
        q = q.filter(DogvIssue.date >= start)
    if end:
        q = q.filter(DogvIssue.date <= end)
    return [row[0] for row in q.order_by(DogvIssue.date.asc()).all()]


def _prefetch_worker(task: tuple) -> tuple:
    """Network-only (runs in a thread): fetch the HTML body and warm the PDF
    cache for annex/unusable-html docs. Returns (doc_id, html_text, has_annex)."""
    doc_id, disp_id, lang, pdf_url = task
    html_text, has_annex = (None, False)
    if disp_id and lang:
        html_text, has_annex = fetch_disposicion_body(disp_id, lang)
    if pdf_url and _needs_pdf(html_text, has_annex):
        pdf_path = build_pdf_path(pdf_url)
        if not pdf_path.exists():
            full_url = build_pdf_url(pdf_url)
            if full_url:
                download_pdf(full_url, pdf_path)
    return doc_id, html_text, has_annex


def reprocess_date(
    db: Session, d: date, workers: int, dry_run: bool, skip_classify: bool = False
) -> tuple[int, int, int]:
    rows = (
        db.query(DogvDocument, DogvIssue.language)
        .join(DogvIssue)
        .options(load_only(*_DOC_COLUMNS))
        .filter(DogvIssue.date == d)
        .all()
    )
    if not rows:
        return 0, 0, 0

    # Phase A — concurrent network: fetch HTML bodies, warm annex PDF cache.
    tasks = [
        (doc.id, (doc.raw_json or {}).get("id"), lang, doc.pdf_url)
        for doc, lang in rows
    ]
    prefetched: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for doc_id, html_text, has_annex in ex.map(_prefetch_worker, tasks):
            prefetched[doc_id] = (html_text, has_annex)

    html_n = pdf_n = none_n = 0
    now = datetime.now(timezone.utc)
    for doc, lang in rows:
        pre = prefetched.get(doc.id, (None, False))
        text, source = resolve_document_text(doc, lang, prefetched=pre)
        if not dry_run:
            if skip_classify:
                # Drop stale chunks + reset text, but KEEP doc_kind: classification
                # is title-driven and unchanged by PDF->HTML, so re-classifying the
                # whole corpus (the dominant LLM cost) is unnecessary.
                _delete_rag_rows(db, doc.id)
                doc.text = None
                doc.text_source = None
                doc.text_updated_at = None
            else:
                # Reset prior text/classification and drop stale chunks so classify
                # and build_chunks pick the doc up again.
                _invalidate_doc_for_reindex(db, doc, reset_text=True)
            if text is not None:
                doc.text = text
                doc.text_source = source
                doc.text_updated_at = now
        if text is None:
            none_n += 1
        elif source == "html":
            html_n += 1
        else:
            pdf_n += 1

    if dry_run:
        print(f"  {d}  docs={len(rows)}  html={html_n} pdf={pdf_n} none={none_n}  (dry-run, no writes)")
        return html_n, pdf_n, none_n

    db.commit()
    # Phase C — needs servers: (optionally) re-classify (chat) then re-chunk/re-embed (embed).
    if not skip_classify:
        classify_range(db, d, d)
    build_chunks_for_range(db, d, d, force=True)
    action = "rechunked (kept doc_kind)" if skip_classify else "reclassified+rechunked"
    print(f"  {d}  docs={len(rows)}  html={html_n} pdf={pdf_n} none={none_n}  {action}")
    return html_n, pdf_n, none_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--all", action="store_true", help="process the entire corpus")
    parser.add_argument("--workers", type=int, default=8, help="concurrent HTML fetchers")
    parser.add_argument("--dry-run", action="store_true", help="fetch + report only; no DB writes, no servers")
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="keep existing doc_kind (title-driven); only re-extract + re-chunk/re-embed",
    )
    parser.add_argument("--limit-dates", type=int, help="process at most N dates (pilot)")
    parser.add_argument("--resume", action="store_true", help="skip dates up to the saved checkpoint")
    args = parser.parse_args()

    if not args.all and not (args.start or args.end):
        parser.error("refusing to run without scope: pass --all or --start/--end")

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    db = SessionLocal()
    try:
        dates = target_dates(db, start, end)
        ckpt = load_checkpoint() if args.resume and not args.dry_run else None
        if ckpt:
            dates = [d for d in dates if d > ckpt]
            print(f"resuming after checkpoint {ckpt}")
        if args.limit_dates:
            dates = dates[: args.limit_dates]

        mode = "DRY-RUN" if args.dry_run else "WRITE"
        classify_note = " skip-classify" if args.skip_classify else ""
        print(f"[{mode}{classify_note}] {len(dates)} issue-dates to process, workers={args.workers}")

        t_html = t_pdf = t_none = 0
        for d in dates:
            h, p, n = reprocess_date(db, d, args.workers, args.dry_run, args.skip_classify)
            t_html += h
            t_pdf += p
            t_none += n
            if not args.dry_run:
                save_checkpoint(d)

        total = t_html + t_pdf + t_none
        print(
            f"\nDone [{mode}]. docs={total}  html={t_html} "
            f"({100*t_html/max(1,total):.1f}%)  pdf={t_pdf}  none={t_none}"
        )
        if not args.dry_run:
            print(f"checkpoint at {dates[-1] if dates else '(none)'}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
