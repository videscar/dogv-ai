"""
Extract document body text into the database.

For each document we fetch the portal's HTML body (the `texto` field of the
disposicion detail endpoint) and prefer it over the cached-PDF extraction,
falling back to the PDF only when the HTML is dramatically shorter (an annex
the portal publishes only as PDF). HTML is cleaner for ~90% of documents (no
per-page header/footer/CVE noise); the PDF is kept for the annex minority.

Usage:
    python scripts/extract_text.py [YYYY-MM-DD [YYYY-MM-DD]]
If dates are provided, filter documents by issue date range (inclusive).
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session, load_only
from pypdf import PdfReader

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.dogv_urls import build_pdf_url
from api.models import DogvDocument, DogvIssue
from api.text_cleaning import clean_html
from scripts.download_assets import build_pdf_path, download_pdf
from scripts.sumario_ingest import fetch_disposicion_json


def extract_text_from_pdf(path: Path) -> Optional[str]:
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
        text = "\n".join(parts).strip()
        return text
    except Exception as e:
        print(f"[error] reading {path}: {e}")
        return None


def fetch_disposicion_body(disp_id, lang: str) -> tuple[Optional[str], bool]:
    """Fetch the portal detail JSON and return (cleaned HTML body, has_annex).

    `texto` is the body; `contieneAnexo` tells us whether an annex exists that
    the portal publishes only as PDF — the signal for whether we need the PDF
    at all. On any failure returns (None, False) so the caller falls back to
    the PDF anyway.
    """
    try:
        data = fetch_disposicion_json(disp_id, lang)
    except Exception as e:
        print(f"[warn] disposicion fetch failed id={disp_id} lang={lang}: {e}")
        return None, False
    has_annex = bool(data.get("contieneAnexo"))
    texto = data.get("texto")
    cleaned = clean_html(texto).strip() if texto else ""
    return (cleaned or None), has_annex


def _norm_len(text: Optional[str]) -> int:
    """Whitespace-normalized character count — fair content comparison across
    HTML (compact) and PDF (whitespace-heavy table layouts) extraction."""
    if not text:
        return 0
    return len(re.sub(r"\s+", " ", text).strip())


# HTML is cleaner but shorter than the PDF (it omits per-page header/footer/CVE
# noise), so the HTML body normally runs ~60-98% of the PDF's normalized length.
# We prefer HTML and only fall back to PDF when the HTML is *dramatically*
# shorter, which signals an annex the portal publishes only in the PDF. The
# corpus distribution has a clean gap at ~0.6 (≈91% of docs sit above it; the
# content-loss tail sits below), so that is the fallback threshold.
HTML_MIN_RATIO = 0.6


def _html_usable(html_text: Optional[str]) -> bool:
    return bool(html_text) and _norm_len(html_text) > 200


def _needs_pdf(html_text: Optional[str], has_annex: bool) -> bool:
    """Whether we must fetch/parse the PDF for this document.

    Only annex documents (annex published only as PDF) or documents whose HTML
    body is unusable need the PDF; the ~90% with a complete HTML body skip the
    PDF download + parse entirely.
    """
    return bool(has_annex) or not _html_usable(html_text)


def _choose_source(html_text: Optional[str], pdf_text: Optional[str]) -> Optional[str]:
    """Decide which source to keep: 'html', 'pdf', or None when neither is usable."""
    html_usable = _html_usable(html_text)
    pdf_usable = bool(pdf_text)
    if html_usable and (
        not pdf_usable or _norm_len(html_text) >= HTML_MIN_RATIO * _norm_len(pdf_text)
    ):
        return "html"
    if pdf_usable:
        return "pdf"
    return "html" if html_usable else None


def _build_text_header(title, conselleria, section) -> str:
    """Title/organismo/section live in separate JSON fields, not inside the HTML
    `texto`. Prepend them so HTML-sourced body text keeps the context the PDF
    body naturally carried."""
    parts = [str(p).strip() for p in (title, conselleria, section) if p and str(p).strip()]
    return "\n".join(parts)


def select_document_text(
    *,
    html_text: Optional[str],
    pdf_text: Optional[str],
    title=None,
    conselleria=None,
    section=None,
) -> tuple[Optional[str], Optional[str]]:
    """Pick the better of an already-fetched HTML body vs PDF text.

    Returns (text, source) where source is 'html' or 'pdf', or (None, None) when
    neither yields usable text. See _choose_source for the prefer-HTML rule.
    HTML-sourced text gets the title/organismo/section header prepended.
    """
    source = _choose_source(html_text, pdf_text)
    if source == "html":
        header = _build_text_header(title, conselleria, section)
        final = f"{header}\n\n{html_text}".strip() if header else html_text
        return final, "html"
    if source == "pdf":
        return pdf_text, "pdf"
    return None, None


def _count_target_documents(db: Session, start_date=None, end_date=None) -> int:
    q = (
        db.query(DogvDocument.id)
        .join(DogvIssue)
        .filter(DogvDocument.text.is_(None))
    )
    if start_date:
        q = q.filter(DogvIssue.date >= start_date)
    if end_date:
        q = q.filter(DogvIssue.date <= end_date)
    return q.count()


def iter_target_document_ids(db: Session, start_date=None, end_date=None, batch_size: int = 200):
    last_id = 0
    while True:
        q = (
            db.query(DogvDocument.id)
            .join(DogvIssue)
            .filter(DogvDocument.text.is_(None))
            .filter(DogvDocument.id > last_id)
        )
        if start_date:
            q = q.filter(DogvIssue.date >= start_date)
        if end_date:
            q = q.filter(DogvIssue.date <= end_date)
        q = q.order_by(DogvDocument.id.asc()).limit(batch_size)
        ids = [row[0] for row in q.all()]
        if not ids:
            break
        yield ids
        last_id = ids[-1]


def extract_range(
    db: Session,
    start_date=None,
    end_date=None,
    batch_size: int = 200,
    commit_every: int = 200,
    fetch_html: bool = True,
) -> int:
    total = _count_target_documents(db, start_date, end_date)
    print(f"Found {total} documents needing text")

    updated = 0
    html_count = 0
    pdf_count = 0
    for ids in iter_target_document_ids(db, start_date, end_date, batch_size=batch_size):
        rows = (
            db.query(DogvDocument, DogvIssue.language)
            .join(DogvIssue)
            .options(
                load_only(
                    DogvDocument.id,
                    DogvDocument.pdf_url,
                    DogvDocument.raw_json,
                    DogvDocument.title,
                    DogvDocument.conselleria,
                    DogvDocument.section,
                    DogvDocument.text,
                    DogvDocument.text_source,
                    DogvDocument.text_updated_at,
                )
            )
            .filter(DogvDocument.id.in_(ids))
            .all()
        )
        for doc, lang in rows:
            disp_id = (doc.raw_json or {}).get("id")
            html_text = None
            has_annex = False
            if fetch_html and disp_id and lang:
                html_text, has_annex = fetch_disposicion_body(disp_id, lang)

            # Only fetch/parse the PDF when the HTML can't stand alone: an annex
            # the portal keeps PDF-only, or an unusable HTML body. The ~90% with
            # a complete HTML body skip the PDF download + parse entirely.
            pdf_text = None
            if doc.pdf_url and _needs_pdf(html_text, has_annex):
                pdf_path = build_pdf_path(doc.pdf_url)
                if not pdf_path.exists():
                    full_url = build_pdf_url(doc.pdf_url)
                    if full_url:
                        download_pdf(full_url, pdf_path)
                if pdf_path.exists():
                    pdf_text = extract_text_from_pdf(pdf_path)

            text, source = select_document_text(
                html_text=html_text,
                pdf_text=pdf_text,
                title=doc.title,
                conselleria=doc.conselleria,
                section=doc.section,
            )

            if text is None:
                print(f"[skip] no text extracted for doc id={doc.id}")
                continue

            doc.text = text
            doc.text_source = source
            doc.text_updated_at = datetime.now(timezone.utc)
            updated += 1
            if source == "html":
                html_count += 1
            else:
                pdf_count += 1

            if updated % commit_every == 0:
                db.commit()
                print(f"[progress] committed {updated} docs (html={html_count} pdf={pdf_count})")

    db.commit()
    print(f"Done. Updated text for {updated} documents (html={html_count} pdf={pdf_count}).")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("end_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--commit-every", type=int, default=200)
    parser.add_argument("--no-fetch-html", action="store_true", help="PDF only; skip the HTML body fetch")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    db = SessionLocal()
    try:
        extract_range(
            db,
            start_date,
            end_date,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
            fetch_html=not args.no_fetch_html,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
