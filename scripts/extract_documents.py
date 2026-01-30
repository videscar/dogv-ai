"""
Extract 'disposicion' entries from DogvIssue.raw_json into DogvDocument rows.

Usage:
    python scripts/extract_documents.py [YYYY-MM-DD [lang]]

If no args: process ALL issues in the DB.
"""

import sys
from datetime import datetime
from typing import Any, Dict, Iterable

from sqlalchemy.orm import Session

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401
from sqlalchemy import text as sa_text
from api.db import SessionLocal
from api.models import DogvIssue, DogvDocument


def _iter_target_issues(db: Session, date_str: str | None, lang: str | None) -> Iterable[DogvIssue]:
    q = db.query(DogvIssue)
    if date_str:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        q = q.filter(DogvIssue.date == date)
    if lang:
        q = q.filter(DogvIssue.language == lang)
    return q.all()


def _extract_document_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map one raw 'disposicion' JSON object to DogvDocument fields.
    Adjust keys if you see different names in debug_issue_json.
    """

    def _normalize(value: Any, nested_keys: tuple[str, ...] = ()) -> str | None:
        """
        Ensure we return simple text for DB columns.
        - Dicts: try nested_keys, else str(dict)
        - Lists: join as comma-separated string
        - Other: cast to str
        """
        if value is None:
            return None
        if isinstance(value, dict):
            for k in nested_keys:
                if value.get(k):
                    return str(value[k])
            return str(value)
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    # Section / area of the gazette
    section_raw = (
        doc.get("seccion")
        or doc.get("section")
        or doc.get("apartado")
    )
    section = _normalize(section_raw, ("descripcion", "description", "desc", "name"))

    # Reference / code
    ref_raw = (
        doc.get("numeroDisposicion")
        or doc.get("ref")
        or doc.get("codigo")
        or doc.get("codigoInsercion")
        or doc.get("codigo_insercion")
    )
    ref = _normalize(ref_raw)

    conselleria_raw = (
        doc.get("conselleria")
        or doc.get("organismo")
        or doc.get("organo")
    )
    conselleria = _normalize(conselleria_raw, ("descripcion", "description", "name"))

    # Title & type
    title_raw = (
        doc.get("titulo")
        or doc.get("titol")
        or doc.get("tituloDisposicion")
        or doc.get("titolDisposicion")
    )
    title = _normalize(title_raw)

    doc_type_raw = (
        doc.get("tipoDisposicion")
        or doc.get("tipo")
        or doc.get("tipus")
    )
    doc_type = _normalize(doc_type_raw)

    # URLs
    pdf_url_raw = (
        doc.get("urlPdfC")
        or doc.get("urlPdf")
        or doc.get("url_pdf")
    )
    pdf_url = _normalize(pdf_url_raw)

    html_url_raw = (
        doc.get("urlHtmlC")
        or doc.get("urlHtml")
        or doc.get("url_html")
    )
    html_url = _normalize(html_url_raw)

    return {
        "section": section,
        "ref": ref,
        "conselleria": conselleria,
        "title": title,
        "type": doc_type,
        "pdf_url": pdf_url,
        "html_url": html_url,
    }


def process_issue(db: Session, issue: DogvIssue) -> int:
    raw = issue.raw_json or {}
    disposiciones = raw.get("disposicion") or raw.get("disposiciones") or []

    if not isinstance(disposiciones, list):
        print(f"[WARN] issue id={issue.id}: 'disposicion' is not a list")
        return 0

    # Clear existing docs (and their chunks) for idempotency
    existing_docs = db.query(DogvDocument.id).filter(DogvDocument.issue_id == issue.id).all()
    if existing_docs:
        for (doc_id,) in existing_docs:
            try:
                db.execute(sa_text("DELETE FROM rag_chunk WHERE document_id = :doc_id"), {"doc_id": doc_id})
                db.execute(sa_text("DELETE FROM rag_title WHERE document_id = :doc_id"), {"doc_id": doc_id})
                db.execute(sa_text("DELETE FROM rag_doc WHERE document_id = :doc_id"), {"doc_id": doc_id})
            except Exception:
                pass
        db.query(DogvDocument).filter(DogvDocument.issue_id == issue.id).delete()

    count = 0
    for idx, d in enumerate(disposiciones):
        if not isinstance(d, dict):
            print(f"[WARN] issue id={issue.id}: disposicion[{idx}] is not an object")
            continue

        fields = _extract_document_fields(d)
        doc = DogvDocument(
            issue_id=issue.id,
            section=fields["section"],
            ref=fields["ref"],
            conselleria=fields["conselleria"],
            title=fields["title"],
            type=fields["type"],
            pdf_url=fields["pdf_url"],
            html_url=fields["html_url"],
            raw_json=d,
        )
        db.add(doc)
        count += 1

    db.commit()
    print(f"[process_issue] issue id={issue.id}, numero={issue.numero}: stored {count} documents")
    return count


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else None
    lang = sys.argv[2] if len(sys.argv) > 2 else None

    db = SessionLocal()
    try:
        issues = _iter_target_issues(db, date_str, lang)
        if not issues:
            print("No issues found for given filters.")
            return

        total_docs = 0
        for issue in issues:
            total_docs += process_issue(db, issue)

        print(f"Total documents stored: {total_docs}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
