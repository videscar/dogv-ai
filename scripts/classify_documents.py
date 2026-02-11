"""
Classify DOGV documents by topic/subtopic using gpt-oss-20b.

Usage:
    python scripts/classify_documents.py [YYYY-MM-DD [YYYY-MM-DD]]
    python scripts/classify_documents.py [YYYY-MM-DD [YYYY-MM-DD]] --workers 2
    python scripts/classify_documents.py [YYYY-MM-DD [YYYY-MM-DD]] --workers 2 --ollama-urls http://127.0.0.1:11434,http://127.0.0.1:11435
"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from typing import Any

from sqlalchemy.orm import Session, load_only

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.classify import classify_document
from api.db import SessionLocal
from api.models import DogvDocument, DogvIssue


def _default_workers() -> int:
    raw = os.getenv("DOGV_CLASSIFY_WORKERS", "2").strip()
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


def _parse_ollama_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    urls = [url.strip().rstrip("/") for url in raw.split(",") if url.strip()]
    return urls


def _count_target_documents(db: Session, start_date=None, end_date=None) -> int:
    q = (
        db.query(DogvDocument.id)
        .join(DogvIssue)
        .filter(DogvDocument.text.isnot(None))
        .filter(DogvDocument.doc_kind.is_(None))
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
            .filter(DogvDocument.text.isnot(None))
            .filter(DogvDocument.doc_kind.is_(None))
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


def _classify_task(
    doc_id: int,
    title: str | None,
    text: str | None,
    base_url: str | None = None,
) -> tuple[int, dict[str, Any] | None, str | None]:
    try:
        return doc_id, classify_document(title, text, base_url=base_url), None
    except Exception as exc:
        return doc_id, None, str(exc)


def classify_range(
    db: Session,
    start_date=None,
    end_date=None,
    batch_size: int = 200,
    commit_every: int = 200,
    workers: int | None = None,
    ollama_urls: list[str] | None = None,
) -> int:
    workers = max(1, workers or _default_workers())
    ollama_urls = ollama_urls or _parse_ollama_urls(os.getenv("DOGV_CLASSIFY_OLLAMA_URLS"))
    endpoint_count = len(ollama_urls)
    total = _count_target_documents(db, start_date, end_date)
    if endpoint_count > 0:
        print(f"Found {total} documents to classify (workers={workers}, endpoints={endpoint_count})")
    else:
        print(f"Found {total} documents to classify (workers={workers})")
    updated = 0

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for ids in iter_target_document_ids(db, start_date, end_date, batch_size=batch_size):
            docs = (
                db.query(DogvDocument)
                .options(load_only(DogvDocument.id, DogvDocument.title, DogvDocument.text))
                .filter(DogvDocument.id.in_(ids))
                .all()
            )
            if not docs:
                continue

            results: dict[int, tuple[dict[str, Any] | None, str | None]] = {}
            if executor is None:
                for index, doc in enumerate(docs):
                    base_url = ollama_urls[index % endpoint_count] if endpoint_count else None
                    _, result, error = _classify_task(doc.id, doc.title, doc.text, base_url=base_url)
                    results[doc.id] = (result, error)
            else:
                futures: list[Future[tuple[int, dict[str, Any] | None, str | None]]] = [
                    executor.submit(
                        _classify_task,
                        doc.id,
                        doc.title,
                        doc.text,
                        ollama_urls[index % endpoint_count] if endpoint_count else None,
                    )
                    for index, doc in enumerate(docs)
                ]
                for future in as_completed(futures):
                    doc_id, result, error = future.result()
                    results[doc_id] = (result, error)

            for doc in docs:
                result, error = results.get(doc.id, (None, "missing classification result"))
                if error is not None:
                    print(f"[warn] classify failed for doc {doc.id}: {error}")
                    doc.doc_kind = "Otros"
                    doc.doc_subkind = None
                    doc.doc_kind_confidence = None
                    doc.doc_tags = {"tags": []}
                else:
                    doc.doc_kind = (result or {}).get("doc_kind") or "Otros"
                    doc.doc_subkind = (result or {}).get("doc_subkind")
                    doc.doc_kind_confidence = (result or {}).get("confidence")
                    doc.doc_tags = {"tags": (result or {}).get("tags") or []}

                updated += 1
                if updated % commit_every == 0:
                    db.commit()
                    print(f"[progress] classified {updated} docs")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    db.commit()
    print(f"Done. Classified {updated} documents.")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("end_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--commit-every", type=int, default=200)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--ollama-urls",
        type=str,
        default=os.getenv("DOGV_CLASSIFY_OLLAMA_URLS", ""),
        help="Comma-separated Ollama base URLs for classification load split.",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    db = SessionLocal()
    try:
        classify_range(
            db,
            start_date,
            end_date,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
            workers=args.workers,
            ollama_urls=_parse_ollama_urls(args.ollama_urls),
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
