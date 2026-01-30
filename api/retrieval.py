from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import date
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session


@dataclass
class RetrievalFilters:
    language: str | None = None
    doc_kind: str | None = None
    doc_subkind: str | None = None
    since_date: date | None = None
    until_date: date | None = None


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


def _run_query(db: Session, sql, params: dict[str, Any], use_seqscan: bool = False):
    if use_seqscan:
        db.execute(sa_text("SET LOCAL enable_indexscan = off"))
        db.execute(sa_text("SET LOCAL enable_bitmapscan = off"))
        db.execute(sa_text("SET LOCAL enable_indexonlyscan = off"))
    rows = db.execute(sql, params).mappings().all()
    if use_seqscan:
        db.execute(sa_text("SET LOCAL enable_indexscan = on"))
        db.execute(sa_text("SET LOCAL enable_bitmapscan = on"))
        db.execute(sa_text("SET LOCAL enable_indexonlyscan = on"))
    return rows


def ts_config_for_language(language: str | None) -> str:
    if not language:
        return "spanish"
    lang = language.lower()
    if lang.startswith(("ca", "va")):
        return "catalan"
    return "spanish"


def _build_filters(filters: RetrievalFilters) -> tuple[str, dict[str, Any]]:
    clauses = []
    params: dict[str, Any] = {}

    if filters.language:
        clauses.append("di.language = :language")
        params["language"] = filters.language
    if filters.doc_kind:
        clauses.append("dd.doc_kind = :doc_kind")
        params["doc_kind"] = filters.doc_kind
    if filters.doc_subkind:
        clauses.append("dd.doc_subkind = :doc_subkind")
        params["doc_subkind"] = filters.doc_subkind
    if filters.since_date:
        clauses.append("di.date >= :since_date")
        params["since_date"] = filters.since_date
    if filters.until_date:
        clauses.append("di.date <= :until_date")
        params["until_date"] = filters.until_date

    if clauses:
        return " AND " + " AND ".join(clauses), params
    return "", params


def vector_search(
    db: Session,
    query_embedding: list[float],
    filters: RetrievalFilters,
    limit: int = 50,
) -> list[dict[str, Any]]:
    filter_sql, params = _build_filters(filters)
    params["query_embedding"] = _vector_literal(query_embedding)
    params["limit"] = limit

    sql = sa_text(
        f"""
        SELECT
            rc.id AS chunk_id,
            rc.document_id,
            rc.chunk_index,
            rc.text,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            (1 - (rc.embedding <=> CAST(:query_embedding AS vector))) AS score
        FROM rag_chunk rc
        JOIN dogv_documents dd ON dd.id = rc.document_id
        JOIN dogv_issues di ON di.id = dd.issue_id
        WHERE rc.embedding IS NOT NULL
        {filter_sql}
        ORDER BY rc.embedding <=> CAST(:query_embedding AS vector)
        LIMIT :limit
        """
    )
    rows = _run_query(db, sql, params)
    if not rows and (filters.doc_kind or filters.doc_subkind):
        rows = _run_query(db, sql, params, use_seqscan=True)
    return [dict(row) for row in rows]


def bm25_search(
    db: Session,
    query_text: str,
    filters: RetrievalFilters,
    limit: int = 50,
) -> list[dict[str, Any]]:
    filter_sql, params = _build_filters(filters)
    params["query_text"] = query_text
    params["limit"] = limit
    ts_config = ts_config_for_language(filters.language)

    sql = sa_text(
        f"""
        WITH q AS (
            SELECT websearch_to_tsquery(:ts_config, :query_text) AS query
        )
        SELECT
            rc.id AS chunk_id,
            rc.document_id,
            rc.chunk_index,
            rc.text,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            ts_rank_cd(rc.tsv, q.query) AS score
        FROM rag_chunk rc
        JOIN dogv_documents dd ON dd.id = rc.document_id
        JOIN dogv_issues di ON di.id = dd.issue_id
        CROSS JOIN q
        WHERE rc.tsv @@ q.query
        {filter_sql}
        ORDER BY score DESC
        LIMIT :limit
        """
    )

    sql_or = sa_text(
        f"""
        WITH q AS (
            SELECT to_tsquery(:ts_config, :query_text) AS query
        )
        SELECT
            rc.id AS chunk_id,
            rc.document_id,
            rc.chunk_index,
            rc.text,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            ts_rank_cd(rc.tsv, q.query) AS score
        FROM rag_chunk rc
        JOIN dogv_documents dd ON dd.id = rc.document_id
        JOIN dogv_issues di ON di.id = dd.issue_id
        CROSS JOIN q
        WHERE rc.tsv @@ q.query
        {filter_sql}
        ORDER BY score DESC
        LIMIT :limit
        """
    )

    def _run(config: str, text: str, use_or: bool = False):
        params_local = dict(params)
        params_local["ts_config"] = config
        params_local["query_text"] = text
        query = sql_or if use_or else sql
        return db.execute(query, params_local).mappings().all()

    rows = _run(ts_config, query_text)
    if not rows and ts_config == "catalan":
        rows = _run("spanish", query_text)

    if not rows:
        tokens = re.findall(r"[\w·'-]+", query_text.lower())
        terms = [t for t in tokens if len(t) >= 4]
        if terms:
            or_query = " | ".join(terms[:8])
            rows = _run(ts_config, or_query, use_or=True)
            if not rows and ts_config == "catalan":
                rows = _run("spanish", or_query, use_or=True)

    return [dict(row) for row in rows]


def title_bm25_search(
    db: Session,
    query_text: str,
    filters: RetrievalFilters,
    limit: int = 50,
) -> list[dict[str, Any]]:
    filter_sql, params = _build_filters(filters)
    params["query_text"] = query_text
    params["limit"] = limit
    ts_config = ts_config_for_language(filters.language)

    sql = sa_text(
        f"""
        WITH q AS (
            SELECT websearch_to_tsquery(:ts_config, :query_text) AS query
        )
        SELECT
            dd.id AS document_id,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            ts_rank_cd(to_tsvector(CAST(:ts_config AS regconfig), coalesce(dd.title, '')), q.query) AS score
        FROM dogv_documents dd
        JOIN dogv_issues di ON di.id = dd.issue_id
        CROSS JOIN q
        WHERE to_tsvector(CAST(:ts_config AS regconfig), coalesce(dd.title, '')) @@ q.query
        {filter_sql}
        ORDER BY score DESC
        LIMIT :limit
        """
    )

    sql_or = sa_text(
        f"""
        WITH q AS (
            SELECT to_tsquery(:ts_config, :query_text) AS query
        )
        SELECT
            dd.id AS document_id,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            ts_rank_cd(to_tsvector(CAST(:ts_config AS regconfig), coalesce(dd.title, '')), q.query) AS score
        FROM dogv_documents dd
        JOIN dogv_issues di ON di.id = dd.issue_id
        CROSS JOIN q
        WHERE to_tsvector(CAST(:ts_config AS regconfig), coalesce(dd.title, '')) @@ q.query
        {filter_sql}
        ORDER BY score DESC
        LIMIT :limit
        """
    )

    def _run(config: str, text: str, use_or: bool = False):
        params_local = dict(params)
        params_local["ts_config"] = config
        params_local["query_text"] = text
        query = sql_or if use_or else sql
        return db.execute(query, params_local).mappings().all()

    rows = _run(ts_config, query_text)
    if not rows and ts_config == "catalan":
        rows = _run("spanish", query_text)

    if not rows:
        tokens = re.findall(r"[\w·'-]+", query_text.lower())
        terms = [t for t in tokens if len(t) >= 2]
        if terms:
            or_query = " | ".join(terms[:8])
            rows = _run(ts_config, or_query, use_or=True)
            if not rows and ts_config == "catalan":
                rows = _run("spanish", or_query, use_or=True)

    return [dict(row) for row in rows]


def title_vector_search(
    db: Session,
    query_embedding: list[float],
    filters: RetrievalFilters,
    limit: int = 30,
) -> list[dict[str, Any]]:
    filter_sql, params = _build_filters(filters)
    params["query_embedding"] = _vector_literal(query_embedding)
    params["limit"] = limit

    sql = sa_text(
        f"""
        SELECT
            rt.document_id,
            dd.title,
            dd.ref,
            dd.type,
            dd.doc_kind,
            dd.doc_subkind,
            dd.pdf_url,
            dd.html_url,
            di.date AS issue_date,
            (1 - (rt.embedding <=> CAST(:query_embedding AS vector))) AS score
        FROM rag_title rt
        JOIN dogv_documents dd ON dd.id = rt.document_id
        JOIN dogv_issues di ON di.id = dd.issue_id
        WHERE rt.embedding IS NOT NULL
        {filter_sql}
        ORDER BY rt.embedding <=> CAST(:query_embedding AS vector)
        LIMIT :limit
        """
    )
    rows = _run_query(db, sql, params)
    if not rows and (filters.doc_kind or filters.doc_subkind):
        rows = _run_query(db, sql, params, use_seqscan=True)
    return [dict(row) for row in rows]


def rrf_fuse(
    sources: list[list[dict[str, Any]]],
    k: int = 60,
    max_docs: int = 50,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    scores: dict[int, float] = {}
    merged: dict[int, dict[str, Any]] = {}

    if weights is None:
        weights = [1.0] * len(sources)
    for source, weight in zip(sources, weights):
        if not source or weight <= 0:
            continue
        for rank, row in enumerate(source, start=1):
            doc_id = row["document_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (weight / (k + rank))
            if doc_id not in merged:
                merged[doc_id] = dict(row)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    results = []
    for doc_id, score in ranked[:max_docs]:
        row = merged[doc_id]
        row["rrf_score"] = score
        results.append(row)
    return results


def group_top_chunks(chunks: list[dict[str, Any]], per_doc: int = 3) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        grouped.setdefault(doc_id, []).append(chunk)

    for doc_id, items in grouped.items():
        items.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        grouped[doc_id] = items[:per_doc]
    return grouped


def top_chunks_for_docs(
    db: Session,
    query_embedding: list[float],
    doc_ids: list[int],
    per_doc: int = 5,
) -> dict[int, list[dict[str, Any]]]:
    if not doc_ids:
        return {}

    params: dict[str, Any] = {
        "query_embedding": _vector_literal(query_embedding),
        "doc_ids": doc_ids,
        "per_doc": per_doc,
    }
    sql = sa_text(
        """
        WITH ranked AS (
            SELECT
                rc.id AS chunk_id,
                rc.document_id,
                rc.chunk_index,
                rc.text,
                (1 - (rc.embedding <=> CAST(:query_embedding AS vector))) AS score,
                ROW_NUMBER() OVER (
                    PARTITION BY rc.document_id
                    ORDER BY rc.embedding <=> CAST(:query_embedding AS vector)
                ) AS rn
            FROM rag_chunk rc
            WHERE rc.document_id = ANY(:doc_ids)
        )
        SELECT chunk_id, document_id, chunk_index, text, score
        FROM ranked
        WHERE rn <= :per_doc
        ORDER BY document_id, rn
        """
    )
    rows = db.execute(sql, params).mappings().all()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["document_id"], []).append(dict(row))
    return grouped
