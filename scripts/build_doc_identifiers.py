"""
Backfill the doc_identifier structured-identifier table from existing corpus
titles/bodies (see sql/2026-07-doc-identifier.sql, api/identifiers.py).

Idempotent (ON CONFLICT DO NOTHING) and safe to re-run over the full corpus.

Usage:
    python scripts/build_doc_identifiers.py [--batch-size 500] [--limit N]
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
from api.identifiers import extract_doc_identifiers
from api.models import DogvDocument, DogvIssue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dogv.build_doc_identifiers")

_INSERT_SQL = sa_text(
    """
    INSERT INTO doc_identifier (document_id, id_kind, id_key, raw, source)
    VALUES (:document_id, :id_kind, :id_key, :raw, :source)
    ON CONFLICT (document_id, id_kind, id_key) DO NOTHING
    """
)


def build_identifiers_for_doc(db, doc: DogvDocument) -> int:
    """Extract + insert identifiers for one document. Returns count extracted."""
    idents = extract_doc_identifiers(doc.title or "", doc.text)
    for ident in idents:
        db.execute(
            _INSERT_SQL,
            {
                "document_id": doc.id,
                "id_kind": ident.id_kind,
                "id_key": ident.id_key,
                "raw": ident.raw,
                "source": ident.source,
            },
        )
    return len(idents)


def build_identifiers_for_range(db, start_date=None, end_date=None) -> int:
    """Extract+insert identifiers for documents ingested in [start_date,
    end_date] (by issue date). Called from the ingest pipeline right after text
    extraction; failure-isolated per document so a bad extraction never fails
    ingest. Returns the number of documents processed."""
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
            build_identifiers_for_doc(db, doc)
        except Exception:
            logger.exception("doc_identifier.extract_failed document_id=%s", doc.id)
            db.rollback()
            continue
        processed += 1
    db.commit()
    logger.info("doc_identifier.range_processed docs=%d", processed)
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
                total_extracted += build_identifiers_for_doc(db, d)
                docs_processed += 1
            db.commit()
            elapsed = time.monotonic() - start
            logger.info(
                "progress docs=%d identifiers=%d elapsed=%.1fs",
                docs_processed,
                total_extracted,
                elapsed,
            )

        kind_rows = db.execute(
            sa_text("SELECT id_kind, count(*) FROM doc_identifier GROUP BY id_kind ORDER BY id_kind")
        ).all()
        kind_counts = {row[0]: row[1] for row in kind_rows}
        distinct_keys = db.execute(
            sa_text("SELECT id_kind, count(DISTINCT id_key) FROM doc_identifier GROUP BY id_kind")
        ).all()
    finally:
        db.close()

    elapsed = time.monotonic() - start
    logger.info("=== build_doc_identifiers summary ===")
    logger.info("docs_processed=%d elapsed=%.1fs", docs_processed, elapsed)
    logger.info("identifiers_extracted_this_run=%d", total_extracted)
    logger.info("rows_by_id_kind=%s", kind_counts)
    logger.info("distinct_keys_by_id_kind=%s", dict(distinct_keys))


if __name__ == "__main__":
    main()
