"""
Chunk documents and store embeddings + BM25 vectors in Postgres.

Usage:
    python scripts/build_chunks.py [YYYY-MM-DD [YYYY-MM-DD]] [--force]
"""

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session, load_only

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import get_settings
from api.db import SessionLocal
from api.ollama import OllamaClient
from api.models import DogvDocument, DogvIssue
from api.retrieval import ts_config_for_language

settings = get_settings()
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None

_TOKENIZER_MAP = {
    "bge-m3": "BAAI/bge-m3",
}


def _embed_with_retry(
    client: OllamaClient,
    text: str,
    attempts: int = 4,
    raise_on_fail: bool = True,
    fallback_strip_diacritics: bool = False,
    fallback_ascii: bool = False,
) -> list[float] | None:
    def _try_embed(payload: str) -> list[float]:
        return client.embed(payload)

    for attempt in range(1, attempts + 1):
        try:
            return _try_embed(text)
        except Exception as exc:
            wait = min(30, 2 ** attempt)
            print(f"[warn] embed failed (attempt {attempt}/{attempts}): {exc}; retry in {wait}s")
            time.sleep(wait)
    if fallback_strip_diacritics:
        stripped = _strip_diacritics(text)
        if stripped and stripped != text:
            print("[warn] embed retry with diacritics-stripped text")
            for attempt in range(1, attempts + 1):
                try:
                    return _try_embed(stripped)
                except Exception as exc:
                    wait = min(30, 2 ** attempt)
                    print(f"[warn] embed failed (attempt {attempt}/{attempts}): {exc}; retry in {wait}s")
                    time.sleep(wait)
    if fallback_ascii:
        ascii_text = _strip_to_ascii(text)
        if ascii_text and ascii_text != text:
            print("[warn] embed retry with ascii-stripped text")
            for attempt in range(1, attempts + 1):
                try:
                    return _try_embed(ascii_text)
                except Exception as exc:
                    wait = min(30, 2 ** attempt)
                    print(f"[warn] embed failed (attempt {attempt}/{attempts}): {exc}; retry in {wait}s")
                    time.sleep(wait)
    if raise_on_fail:
        raise RuntimeError("embed failed after retries")
    return None


def _strip_diacritics(text: str) -> str:
    if not text:
        return text
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _strip_to_ascii(text: str) -> str:
    if not text:
        return text
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = []
    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in ("\n", "\r", "\t"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _prepare_embed_text(text: str) -> str:
    if not text:
        return ""
    text = _sanitize_text(text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text


def _embed_batch_with_retry(
    client: OllamaClient,
    batch: list[str],
    attempts: int = 3,
) -> list[list[float]]:
    for attempt in range(1, attempts + 1):
        try:
            return client.embed_batch(batch)
        except Exception as exc:
            wait = min(30, 2 ** attempt)
            print(f"[warn] embed_batch failed (attempt {attempt}/{attempts}): {exc}; retry in {wait}s")
            time.sleep(wait)
    # Fallback to per-item embedding for better resilience
    embeddings: list[list[float]] = []
    for text in batch:
        item = _embed_with_retry(
            client,
            text,
            raise_on_fail=False,
            fallback_strip_diacritics=True,
            fallback_ascii=True,
        )
        if item is None:
            return []
        embeddings.append(item)
    return embeddings


def _embed_chunks_individual(
    client: OllamaClient,
    chunks: list[str],
    embed_sleep_ms: int = 0,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for chunk in chunks:
        if embed_sleep_ms:
            time.sleep(embed_sleep_ms / 1000)
        item = _embed_with_retry(
            client,
            chunk,
            raise_on_fail=False,
            fallback_strip_diacritics=True,
            fallback_ascii=True,
        )
        if item is None:
            return []
        embeddings.append(item)
    return embeddings


def _normalize_chunk_text(text: str) -> str:
    if not text:
        return ""
    text = _sanitize_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text

@lru_cache(maxsize=1)
def _load_embed_tokenizer() -> Any | None:
    if AutoTokenizer is None:
        return None
    model = (settings.ollama_embed_model or "").strip()
    if not model:
        return None
    repo = _TOKENIZER_MAP.get(model, model)
    try:
        return AutoTokenizer.from_pretrained(repo, use_fast=True)
    except Exception as exc:
        print(f"[warn] tokenizer load failed for {repo}: {exc}")
        return None


def _chunk_text_tokens(
    text: str,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    text = _normalize_chunk_text(text)
    if not text:
        return []

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
    except Exception as exc:
        raise RuntimeError(f"tokenizer encode failed: {exc}") from exc

    offsets = encoded.get("offset_mapping") or []
    total = len(offsets)
    if total == 0:
        return [text]
    if max_tokens and total <= max_tokens:
        return [text]

    overlap = max(0, min(overlap_tokens, max_tokens - 1)) if max_tokens else 0
    chunks: list[str] = []
    idx = 0
    while idx < total:
        end = total if not max_tokens else min(total, idx + max_tokens)
        if end == total and chunks and min_tokens and (end - idx) < min_tokens:
            start_char = offsets[idx][0]
            end_char = offsets[-1][1]
            tail = text[start_char:end_char].strip()
            if tail:
                chunks[-1] = (chunks[-1] + " " + tail).strip()
            break
        start_char = offsets[idx][0]
        end_char = offsets[end - 1][1]
        chunk = text[start_char:end_char].strip()
        if chunk:
            chunks.append(chunk)
        if end >= total:
            break
        idx = max(0, end - overlap)
    return chunks


def chunk_text(
    text: str,
    tokenizer: Any,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for token-based chunking.")
    return _chunk_text_tokens(text, tokenizer, min_tokens, max_tokens, overlap_tokens)


def _build_doc_summary(text: str, max_words: int = 120, max_chars: int = 1200) -> str:
    if not text:
        return ""
    words = text.split()
    if not words:
        return ""
    summary = " ".join(words[:max_words]).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary


def _build_doc_embedding_text(doc: DogvDocument, issue: DogvIssue) -> tuple[str, str]:
    summary = _build_doc_summary(doc.text or "")
    parts = []
    if doc.title:
        parts.append(f"titulo: {doc.title}")
    if summary:
        parts.append(f"resumen: {summary}")
    if doc.ref:
        parts.append(f"ref: {doc.ref}")
    if doc.section:
        parts.append(f"seccion: {doc.section}")
    if doc.doc_kind:
        parts.append(f"doc_kind: {doc.doc_kind}")
    if doc.doc_subkind:
        parts.append(f"doc_subkind: {doc.doc_subkind}")
    if issue.date:
        parts.append(f"fecha: {issue.date.isoformat()}")
    if issue.language:
        parts.append(f"idioma: {issue.language}")
    return "\n".join(parts), summary


def _count_documents(db: Session, start_date=None, end_date=None, force: bool = False) -> int:
    q = db.query(DogvDocument.id).join(DogvIssue).filter(DogvDocument.text.isnot(None))
    if start_date:
        q = q.filter(DogvIssue.date >= start_date)
    if end_date:
        q = q.filter(DogvIssue.date <= end_date)
    if not force:
        q = q.filter(
            ~DogvDocument.id.in_(
                db.execute(sa_text("SELECT DISTINCT document_id FROM rag_chunk")).scalars().all()
            )
        )
    return q.count()


def iter_document_ids(db: Session, start_date=None, end_date=None, force: bool = False, batch_size: int = 100):
    last_id = 0
    while True:
        q = (
            db.query(DogvDocument.id)
            .join(DogvIssue)
            .filter(DogvDocument.text.isnot(None))
            .filter(DogvDocument.id > last_id)
        )
        if start_date:
            q = q.filter(DogvIssue.date >= start_date)
        if end_date:
            q = q.filter(DogvIssue.date <= end_date)
        if not force:
            q = q.filter(
                ~DogvDocument.id.in_(
                    db.execute(sa_text("SELECT DISTINCT document_id FROM rag_chunk")).scalars().all()
                )
            )
        q = q.order_by(DogvDocument.id.asc()).limit(batch_size)
        ids = [row[0] for row in q.all()]
        if not ids:
            break
        yield ids
        last_id = ids[-1]


def build_chunks_for_range(
    db: Session,
    start_date=None,
    end_date=None,
    force: bool = False,
    batch_size: int = 50,
    commit_every: int = 20,
    embed_batch_size: int = 8,
    build_doc_embeddings: bool = True,
    use_embed_batch: bool = True,
    embed_sleep_ms: int = 0,
) -> int:
    client = OllamaClient()
    tokenizer = _load_embed_tokenizer()
    if tokenizer is None:
        raise RuntimeError("Tokenizer unavailable. Install transformers and ensure the model tokenizer loads.")
    total = _count_documents(db, start_date, end_date, force=force)
    print(f"Found {total} documents to chunk/embed")

    processed = 0
    batch_enabled = use_embed_batch and embed_batch_size > 1
    for ids in iter_document_ids(db, start_date, end_date, force=force, batch_size=batch_size):
        rows = (
            db.query(DogvDocument, DogvIssue)
            .join(DogvIssue)
            .options(
                load_only(
                    DogvDocument.id,
                    DogvDocument.text,
                    DogvDocument.title,
                    DogvDocument.ref,
                    DogvDocument.section,
                    DogvDocument.doc_kind,
                    DogvDocument.doc_subkind,
                    DogvDocument.pdf_url,
                    DogvDocument.html_url,
                    DogvDocument.issue_id,
                )
            )
            .filter(DogvDocument.id.in_(ids))
            .all()
        )
        for doc, issue in rows:
            if not doc.text:
                continue
            raw_text = doc.text.strip()
            if not raw_text:
                continue

            if force:
                db.execute(sa_text("DELETE FROM rag_chunk WHERE document_id = :doc_id"), {"doc_id": doc.id})
                db.execute(sa_text("DELETE FROM rag_title WHERE document_id = :doc_id"), {"doc_id": doc.id})
                db.execute(sa_text("DELETE FROM rag_doc WHERE document_id = :doc_id"), {"doc_id": doc.id})

            chunks = chunk_text(
                raw_text,
                tokenizer=tokenizer,
                min_tokens=settings.chunk_min_tokens,
                max_tokens=settings.chunk_max_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
            )
            if not chunks:
                continue
            filtered_chunks: list[str] = []
            embed_texts: list[str] = []
            for chunk in chunks:
                if not chunk.strip() or not _has_alnum(chunk):
                    continue
                embed_text = _prepare_embed_text(chunk)
                if not embed_text:
                    continue
                filtered_chunks.append(chunk)
                embed_texts.append(embed_text)
            chunks = filtered_chunks
            if not chunks:
                print(f"[warn] no alnum content after sanitizing doc {doc.id}")
                continue

            embeddings: list[list[float]] = []
            if batch_enabled:
                embedding_failed = False
                for i in range(0, len(embed_texts), embed_batch_size):
                    batch = embed_texts[i : i + embed_batch_size]
                    batch_embeddings = _embed_batch_with_retry(client, batch)
                    if not batch_embeddings or len(batch_embeddings) != len(batch):
                        print(
                            f"[warn] embedding batch failed for doc {doc.id}; switching to per-item embeds"
                        )
                        batch_enabled = False
                        embedding_failed = True
                        break
                    embeddings.extend(batch_embeddings)
                if embedding_failed:
                    embeddings = _embed_chunks_individual(
                        client,
                        embed_texts,
                        embed_sleep_ms=embed_sleep_ms,
                    )
                    if not embeddings:
                        continue
            else:
                embeddings = _embed_chunks_individual(
                    client,
                    embed_texts,
                    embed_sleep_ms=embed_sleep_ms,
                )
                if not embeddings:
                    continue

            if len(embeddings) != len(chunks):
                print(f"[warn] embedding count mismatch for doc {doc.id}")
                continue

            metadata = {
                "title": doc.title,
                "ref": doc.ref,
                "section": doc.section,
                "doc_kind": doc.doc_kind,
                "doc_subkind": doc.doc_subkind,
                "issue_date": issue.date.isoformat() if issue.date else None,
            }
            ts_config = ts_config_for_language(issue.language)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                db.execute(
                    sa_text(
                        """
                        INSERT INTO rag_chunk (
                            document_id, chunk_index, language, text, embedding, metadata, tsv
                        ) VALUES (
                            :document_id, :chunk_index, :language, :text,
                            CAST(:embedding AS vector), CAST(:metadata AS jsonb),
                            to_tsvector(CAST(:ts_config AS regconfig), :text)
                        )
                        ON CONFLICT (document_id, chunk_index)
                        DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            tsv = EXCLUDED.tsv
                        """
                    ),
                    {
                        "document_id": doc.id,
                        "chunk_index": idx,
                        "language": issue.language,
                        "text": chunk,
                        "embedding": "[" + ",".join(f"{v:.6f}" for v in embedding) + "]",
                        "metadata": json.dumps(metadata),
                        "ts_config": ts_config,
                    },
                )

            if doc.title:
                title_embedding = _embed_with_retry(
                    client,
                    doc.title,
                    raise_on_fail=False,
                    fallback_strip_diacritics=True,
                    fallback_ascii=True,
                )
                if title_embedding:
                    db.execute(
                        sa_text(
                            """
                            INSERT INTO rag_title (document_id, language, title, embedding)
                            VALUES (:document_id, :language, :title, CAST(:embedding AS vector))
                            ON CONFLICT (document_id)
                            DO UPDATE SET
                                title = EXCLUDED.title,
                                embedding = EXCLUDED.embedding
                            """
                        ),
                        {
                            "document_id": doc.id,
                            "language": issue.language,
                            "title": doc.title,
                            "embedding": "[" + ",".join(f"{v:.6f}" for v in title_embedding) + "]",
                        },
                    )
                else:
                    print(f"[warn] title embedding skipped for doc {doc.id}")

            if build_doc_embeddings:
                doc_text, summary = _build_doc_embedding_text(doc, issue)
                if doc_text:
                    doc_embedding = _embed_with_retry(
                        client,
                        doc_text,
                        raise_on_fail=False,
                        fallback_strip_diacritics=True,
                        fallback_ascii=True,
                    )
                    if doc_embedding:
                        metadata = {
                            "ref": doc.ref,
                            "section": doc.section,
                            "doc_kind": doc.doc_kind,
                            "doc_subkind": doc.doc_subkind,
                            "issue_date": issue.date.isoformat() if issue.date else None,
                        }
                        db.execute(
                            sa_text(
                                """
                                INSERT INTO rag_doc (document_id, language, title, summary, metadata, embedding)
                                VALUES (:document_id, :language, :title, :summary, CAST(:metadata AS jsonb),
                                        CAST(:embedding AS vector))
                                ON CONFLICT (document_id)
                                DO UPDATE SET
                                    title = EXCLUDED.title,
                                    summary = EXCLUDED.summary,
                                    metadata = EXCLUDED.metadata,
                                    embedding = EXCLUDED.embedding
                                """
                            ),
                            {
                                "document_id": doc.id,
                                "language": issue.language,
                                "title": doc.title or "",
                                "summary": summary,
                                "metadata": json.dumps(metadata),
                                "embedding": "[" + ",".join(f"{v:.6f}" for v in doc_embedding) + "]",
                            },
                        )
                    else:
                        print(f"[warn] doc embedding skipped for doc {doc.id}")

            processed += 1
            if processed % commit_every == 0:
                db.commit()
                print(f"[progress] processed {processed} docs")

    db.commit()
    print(f"Done. Processed {processed} documents.")
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("end_date", nargs="?", help="YYYY-MM-DD")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--commit-every", type=int, default=20)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--skip-doc-embeddings", action="store_true")
    parser.add_argument("--no-embed-batch", action="store_true")
    parser.add_argument("--embed-sleep-ms", type=int, default=0)
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    db = SessionLocal()
    try:
        build_chunks_for_range(
            db,
            start_date,
            end_date,
            force=args.force,
            batch_size=args.batch_size,
            commit_every=args.commit_every,
            embed_batch_size=args.embed_batch_size,
            build_doc_embeddings=not args.skip_doc_embeddings,
            use_embed_batch=not args.no_embed_batch,
            embed_sleep_ms=args.embed_sleep_ms,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
