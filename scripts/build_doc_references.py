"""
Backfill the doc_reference cross-reference link table from existing corpus
titles/bodies (see sql/2026-07-doc-references.sql, api/doc_references.py).

Idempotent (ON CONFLICT DO NOTHING) and safe to re-run over the full corpus
after retagging the extraction patterns.

Usage:
    python scripts/build_doc_references.py [--batch-size 500] [--limit N]
"""

import argparse
import logging
import time

from sqlalchemy import text as sa_text
from sqlalchemy.orm import load_only

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.doc_references import extract_references, resolve_target_document_id
from api.models import DogvDocument, DogvIssue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dogv.build_doc_references")

_INSERT_SQL = sa_text(
    """
    INSERT INTO doc_reference
        (source_document_id, target_document_id, ref_tipo, ref_numero, ref_anyo, ref_kind, ref_key, raw_text)
    VALUES
        (:source_document_id, :target_document_id, :ref_tipo, :ref_numero, :ref_anyo, :ref_kind, :ref_key, :raw_text)
    ON CONFLICT (source_document_id, ref_key, ref_kind) DO NOTHING
    """
)


def build_references_for_doc(db, doc: DogvDocument) -> tuple[int, int]:
    """Extract + insert references for one document. Returns (extracted, resolved)."""
    refs = extract_references(doc.title or "", doc.text)
    resolved = 0
    for ref in refs:
        target_id = resolve_target_document_id(db, doc.id, ref, source_title=doc.title or "")
        if target_id is not None:
            resolved += 1
        db.execute(
            _INSERT_SQL,
            {
                "source_document_id": doc.id,
                "target_document_id": target_id,
                "ref_tipo": ref.ref_tipo,
                "ref_numero": ref.ref_numero,
                "ref_anyo": ref.ref_anyo,
                "ref_kind": ref.ref_kind,
                "ref_key": ref.ref_key,
                "raw_text": ref.raw_text,
            },
        )
    return len(refs), resolved


def build_references_for_range(db, start_date=None, end_date=None) -> int:
    """Extract+insert references for documents ingested in [start_date,
    end_date] (by issue date). Called from the ingest pipeline right after
    text extraction finishes; failure-isolated per document so a bad
    extraction never fails ingest.

    Returns the number of documents processed.
    """
    query = db.query(DogvDocument).options(
        load_only(DogvDocument.id, DogvDocument.title, DogvDocument.text)
    )
    if start_date is not None or end_date is not None:
        query = query.join(DogvIssue, DogvDocument.issue_id == DogvIssue.id)
        if start_date is not None:
            query = query.filter(DogvIssue.date >= start_date)
        if end_date is not None:
            query = query.filter(DogvIssue.date <= end_date)

    processed = 0
    for doc in query.all():
        try:
            build_references_for_doc(db, doc)
        except Exception:
            logger.exception("doc_reference.extract_failed document_id=%s", doc.id)
            db.rollback()
            continue
        processed += 1
    db.commit()
    logger.info("doc_reference.range_processed docs=%d", processed)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="cap docs processed (debug)")
    args = parser.parse_args()

    db = SessionLocal()
    start = time.monotonic()
    docs_processed = 0
    total_extracted = 0
    total_resolved = 0
    kind_counts: dict[str, int] = {}

    try:
        id_query = db.query(DogvDocument.id).order_by(DogvDocument.id)
        if args.limit:
            id_query = id_query.limit(args.limit)
        all_ids = [row[0] for row in id_query.all()]

        for i in range(0, len(all_ids), args.batch_size):
            batch_ids = all_ids[i : i + args.batch_size]
            docs = (
                db.query(DogvDocument)
                .options(load_only(DogvDocument.id, DogvDocument.title, DogvDocument.text))
                .filter(DogvDocument.id.in_(batch_ids))
                .all()
            )
            for d in docs:
                n_extracted, n_resolved = build_references_for_doc(db, d)
                total_extracted += n_extracted
                total_resolved += n_resolved
                docs_processed += 1
            db.commit()
            elapsed = time.monotonic() - start
            logger.info(
                "progress docs=%d refs=%d resolved=%d elapsed=%.1fs",
                docs_processed,
                total_extracted,
                total_resolved,
                elapsed,
            )

        kind_rows = db.execute(
            sa_text("SELECT ref_kind, count(*) FROM doc_reference GROUP BY ref_kind")
        ).all()
        kind_counts = {row[0]: row[1] for row in kind_rows}
        total_rows = sum(kind_counts.values())
        resolved_rows = db.execute(
            sa_text("SELECT count(*) FROM doc_reference WHERE target_document_id IS NOT NULL")
        ).scalar()
    finally:
        db.close()

    elapsed = time.monotonic() - start
    pct_resolved = (resolved_rows / total_rows * 100) if total_rows else 0.0
    logger.info("=== build_doc_references summary ===")
    logger.info("docs_processed=%d elapsed=%.1fs", docs_processed, elapsed)
    logger.info("references_extracted_this_run=%d", total_extracted)
    logger.info(
        "doc_reference_rows_total=%d resolved_to_corpus=%d (%.1f%%)",
        total_rows,
        resolved_rows,
        pct_resolved,
    )
    logger.info("breakdown_by_ref_kind=%s", kind_counts)


if __name__ == "__main__":
    main()
