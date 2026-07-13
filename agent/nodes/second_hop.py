"""Gated second retrieval hop for multi-entity "compare X and Y" questions.

Problem: the primary retrieve pass embeds/BM25s the whole question as one query,
so a second explicit norm (Orden 23/2026 vs Orden 18/2026) or a second topical
facet (alquiler DANA vs Fondo de Cooperacion Municipal) never gets its own
targeted lane and its document falls out of the fused pool entirely. This module
runs a small, surgical extra retrieval pass per uncovered entity and merges its
top docs additively into the pool the main node already computed.

Two triggers, checked in order, capped at `ask_second_hop_max_hops` total hops:
1. Explicit multi-ref: >=2 distinct N/YYYY norm references in the question
   (api.dogv_resolver.parse_references) where a ref's own tipo+num_year is not
   carried by any title already in the pool.
2. Facet: only tried when no ref hop fired (keeps it conservative — an explicit-
   ref question is already handled by trigger 1). Each facet beyond the first
   (api.query_expansion.decompose_question) gets its own hop query; the hop is
   only considered "uncovered" (and merged) when the facet's own top-ranked doc
   is entirely absent from the current fused pool. This is intentionally tight:
   a facet whose top doc already made the pool needs no help, and checking only
   the top-1 (vs "not in top candidates") keeps the trigger from firing on the
   ordinary paraphrase noise every multi-clause question has.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text as sa_text

from agent.nodes.retrieve_pool import PoolQuery, PoolResult, compute_pool
from api.config import enabled_lanes, get_settings
from api.db import SessionLocal
from api.dogv_resolver import (
    Reference,
    _topic_terms_of,
    corpus_like_patterns,
    parse_references,
    reference_matches_title,
)
from api.field_anchor import identity_query, is_multi_field_grant_query
from api.query_expansion import build_bm25_queries, decompose_question
from api.retrieval import RetrievalFilters

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _uncovered_refs(question: str, fused: list[dict[str, Any]]) -> list[Reference]:
    refs = parse_references(question)
    if len(refs) < 2:
        return []
    titles = [d.get("title") or "" for d in fused]
    return [ref for ref in refs if not any(reference_matches_title(ref, t) for t in titles)]


def _ref_hop_question(ref: Reference) -> str:
    """Focused hop query: the reference itself plus the question's topic words,
    so BM25/vector target this entity instead of drifting back to the first one."""
    topic = " ".join(ref.topic_terms[:8])
    return f"{ref.search_text()} {topic}".strip()


def _direct_title_lookup(ref: Reference, limit: int = 3) -> list[dict[str, Any]]:
    """Direct title-pattern SQL fetch (corpus_like_patterns), same mechanism the
    backfill on-demand-fetch path uses to confirm corpus presence. A short numeric
    title like "ORDEN 18/2026..." carries little lexical/semantic signal, so the
    embedding+BM25 hop lane can still miss it even when it's plainly in corpus;
    this is the fallback that recovers it deterministically."""
    patterns = corpus_like_patterns(ref)
    if not patterns:
        return []
    clauses = " OR ".join(f"dd.title ILIKE :p{i}" for i in range(len(patterns)))
    params: dict[str, Any] = {f"p{i}": pat for i, pat in enumerate(patterns)}
    params["limit"] = limit
    sql = sa_text(
        f"""
        SELECT dd.id AS document_id, dd.title, dd.ref, dd.type, dd.doc_kind,
               dd.doc_subkind, dd.pdf_url, dd.html_url, di.date AS issue_date
        FROM dogv_documents dd
        JOIN dogv_issues di ON di.id = dd.issue_id
        WHERE ({clauses})
        ORDER BY di.date DESC
        LIMIT :limit
        """
    )
    with SessionLocal() as db:
        rows = db.execute(sql, params).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        if reference_matches_title(ref, d.get("title") or ""):
            d["rrf_score"] = 0.0
            out.append(d)
    return out


# Interrogative marker opening a second clause after "y"/"i": "i quin és el sou
# base...", "y cuál es...", "y cuánto se concedió...". A conjunction followed by
# one of these is the strongest signal the question carries a second entity.
_CLAUSE_MARKER_RE = re.compile(
    r"^(?:quin(?:a|s|es)?\b|qu[eè]\b|qu[eé]\b|cu[aá]l(?:es)?\b|qui(?:[eé]n(?:es)?)?\b|"
    r"cu[aá]nt[oa]s?\b|quant(?:a|s|es)?\b)",
    re.IGNORECASE,
)

_CONJUNCTION_RE = re.compile(r"\s(?:y|i)\s", re.IGNORECASE)


def _split_compound_clauses(question: str) -> list[str]:
    """Hop-local entity splitter for single-question compound clauses.

    decompose_question only splits on ?/;/. boundaries, so "quina taxa es paga
    (Ordre 23/2026) i quin és el sou base..." — one question mark, two entities —
    never yields a second facet. This splitter handles the conjunction case:
    prefer the "y"/"i" whose right side opens with an interrogative marker (the
    second ask), else fall back to the LAST conjunction (list-final "y el Fondo
    de Cooperación Municipal" names the second entity; earlier conjunctions like
    "la DANA y los municipios" join words inside the first one). Both halves
    must be entity-like (>=2 content words) or the split is rejected. Lives here
    (not in query_expansion) so the global BM25 facet path is untouched and
    ungated questions stay byte-identical.
    """
    q = (question or "").strip()
    if not q:
        return []
    matches = list(_CONJUNCTION_RE.finditer(q))
    if not matches:
        return []
    chosen = None
    for m in matches:
        if _CLAUSE_MARKER_RE.match(q[m.end() :].lstrip(" ¿¡")):
            chosen = m
            break
    if chosen is None:
        chosen = matches[-1]
    halves = [
        q[: chosen.start()].strip(" ¿?¡!.,;:"),
        q[chosen.end() :].strip(" ¿?¡!.,;:"),
    ]
    # A half that runs past a sentence boundary drags the NEXT sentence into the
    # entity query ("el Fondo de Cooperación Municipal? Indica el importe...")
    # and steers the hop to lexically-adjacent non-entities (v2-051 pulled the
    # *turístico* Fondo variant). Keep only the clause up to the boundary.
    halves = [re.split(r"[?;.]", h, maxsplit=1)[0].strip(" ¿?¡!.,;:") for h in halves]
    if any(len(_topic_terms_of(h)) < 2 for h in halves):
        return []
    # The second clause must name an entity of its own (a digit/year or a
    # mid-clause capitalized proper noun: "subgrup A1 (taules 2026)", "Fondo de
    # Cooperación Municipal"). An anaphoric second clause ("y qué subida salarial
    # aplican") asks a second FACT about the same document — hopping on it only
    # pins near-duplicate noise into the read set.
    if not _has_entity_anchor(halves[1]):
        return []
    return halves


def _has_entity_anchor(clause: str) -> bool:
    if re.search(r"\d", clause):
        return True
    words = clause.split()
    return any(w[:1].isupper() for w in words[1:])


def _facet_targets(question: str) -> list[str]:
    """Hop facet candidates: the local compound-clause split when it fires
    (its second half is the entity the primary pass under-weighted), else
    decompose_question's trailing facets. The first half of a local split is
    kept too — coverage checks are cheap relative to a missed entity, and the
    top-1-absent gate no-ops the half the main pass already covered."""
    if getattr(settings, "ask_field_anchor_enabled", True) and is_multi_field_grant_query(question):
        # Field-request clauses must never become hop queries (their pools are
        # per-field noise); the identity hop in apply_second_hop covers these.
        return []
    halves = _split_compound_clauses(question)
    if halves:
        return list(reversed(halves))  # second clause first: it's the likely miss
    facets = decompose_question(question, max_facets=getattr(settings, "ask_max_facets", 3))
    return facets[1:] if len(facets) >= 2 else []


def _run_hop_pool(
    client: Any,
    hop_question: str,
    intent: dict[str, Any],
    lanes: set[str],
    filters: RetrievalFilters,
) -> PoolResult | None:
    try:
        embeddings = client.embed_batch([hop_question])
        if not embeddings or len(embeddings) != 1:
            embeddings = [client.embed(hop_question)]
    except Exception:
        logger.exception("second_hop.embed_failed question=%s", hop_question)
        return None
    query = PoolQuery(
        lanes=lanes,
        bm25_query=hop_question,
        bm25_specs=[build_bm25_queries(hop_question, intent)],
        query_embedding=embeddings[0],
        hyde_embedding=None,
        bm25_limit=max(10, settings.ask_bm25_max_docs),
        min_docs=1,
        expand_ratio=getattr(settings, "ask_rrf_expand_margin_ratio", 0.12),
        expand_probe=getattr(settings, "ask_rrf_margin_probe", 5),
        expand_docs=0,
        anchors_enabled=False,
    )
    return compute_pool(query, filters)


def _merge_additive(
    pool: PoolResult, extra_docs: list[dict[str, Any]], hop_pool: PoolResult
) -> list[int]:
    """Append docs not already in `pool.fused` to the end (never reorder/remove
    existing candidates) and carry their chunk evidence along. Returns the ids
    actually added."""
    existing_ids = {int(d["document_id"]) for d in pool.fused}
    added_ids: list[int] = []
    for doc in extra_docs:
        doc_id = int(doc["document_id"])
        if doc_id in existing_ids:
            continue
        pool.fused.append(doc)
        existing_ids.add(doc_id)
        added_ids.append(doc_id)
        if doc_id in hop_pool.top_chunks and doc_id not in pool.top_chunks:
            pool.top_chunks[doc_id] = hop_pool.top_chunks[doc_id]
    if added_ids:
        added_set = set(added_ids)
        pool.chunk_candidates += [
            c for c in hop_pool.chunk_candidates if int(c.get("document_id", -1)) in added_set
        ]
    return added_ids


def apply_second_hop(
    question: str,
    pool: PoolResult,
    filters: RetrievalFilters,
    intent: dict[str, Any],
    client: Any,
    request_id: str | None,
) -> tuple[PoolResult, list[int], list[int], dict[str, Any] | None]:
    """Run the gated second hop in place on `pool`. Returns (pool, added_doc_ids,
    protect_doc_ids, profile) — profile is None when the gate never fired (zero
    extra work done). protect_doc_ids are each evaluated entity's top-ranked docs
    that were ALREADY in the pool: they need no merge/pin, but they are that
    entity's best evidence, so edition-recency suppression (RC1) must not prune
    them as stale siblings of the OTHER entity's fresher docs (v2-051: the 2025
    DANA-alquiler concesión was in the pool and RC1 dropped it)."""
    if not getattr(settings, "ask_second_hop_enabled", False):
        return pool, [], [], None
    max_hops = max(0, getattr(settings, "ask_second_hop_max_hops", 2))
    if max_hops == 0:
        return pool, [], [], None
    top_docs = max(1, getattr(settings, "ask_second_hop_top_docs", 5))
    lanes = enabled_lanes(settings)

    fired: list[str] = []
    added_doc_ids: list[int] = []
    protect_doc_ids: list[int] = []
    hops_used = 0

    def _collect_protected(hop_pool: PoolResult) -> None:
        pool_ids = {int(d["document_id"]) for d in pool.fused}
        for doc in hop_pool.fused[: min(3, top_docs)]:
            doc_id = int(doc["document_id"])
            if doc_id in pool_ids and doc_id not in protect_doc_ids:
                protect_doc_ids.append(doc_id)

    for ref in _uncovered_refs(question, pool.fused):
        if hops_used >= max_hops:
            break
        hop_pool = _run_hop_pool(client, _ref_hop_question(ref), intent, lanes, filters)
        if hop_pool is None:
            continue
        _collect_protected(hop_pool)
        extra_docs = list(hop_pool.fused[:top_docs])
        extra_ids = {int(d["document_id"]) for d in extra_docs}
        for direct_doc in _direct_title_lookup(ref):
            if int(direct_doc["document_id"]) not in extra_ids:
                extra_docs.append(direct_doc)
        added = _merge_additive(pool, extra_docs, hop_pool)
        if added:
            added_doc_ids += added
            fired.append(f"ref:{ref.raw}")
            hops_used += 1

    # Multi-field grant query: the facet targets would be the FIELD-request
    # clauses ("quant es cobra al mes...", "qui hi pot optar, l'import, el
    # termini..."), whose retrieval pools are per-field noise (amount-dense
    # credit transfers, SS laws) that then gets pinned into the read set. Hop
    # on the identity query instead — the question minus its field clauses —
    # and pin its top docs (typically the es+va editions of the target act)
    # into the read set even when they are already in the pool: the fused pool
    # ranks by the field-diluted full question, so pool membership alone does
    # not get the target read.
    if (
        hops_used < max_hops
        and not fired
        and getattr(settings, "ask_field_anchor_enabled", True)
        and is_multi_field_grant_query(question)
    ):
        identity = identity_query(question)
        if identity:
            hop_pool = _run_hop_pool(client, identity, intent, lanes, filters)
            if hop_pool is not None and hop_pool.fused:
                _collect_protected(hop_pool)
                pin_n = max(1, getattr(settings, "ask_field_anchor_pin_docs", 2))
                anchor_ids = [int(d["document_id"]) for d in hop_pool.fused[:pin_n]]
                _merge_additive(pool, list(hop_pool.fused[:top_docs]), hop_pool)
                for doc_id in anchor_ids:
                    if doc_id not in added_doc_ids:
                        added_doc_ids.append(doc_id)
                fired.append(f"identity:{identity[:80]}")
                hops_used += 1

    if hops_used < max_hops and not fired:
        for facet in _facet_targets(question):
            if hops_used >= max_hops:
                break
            hop_pool = _run_hop_pool(client, facet, intent, lanes, filters)
            if hop_pool is None or not hop_pool.fused:
                continue
            _collect_protected(hop_pool)
            fused_ids = {int(d["document_id"]) for d in pool.fused}
            if int(hop_pool.fused[0]["document_id"]) in fused_ids:
                continue  # facet's top doc already covered: conservative no-op
            added = _merge_additive(pool, list(hop_pool.fused[:top_docs]), hop_pool)
            if added:
                added_doc_ids += added
                fired.append(f"facet:{facet}")
                hops_used += 1

    if not fired:
        return pool, [], [], None
    logger.info(
        "retrieve.second_hop req=%s entities=%s docs_added=%s protected=%s",
        request_id,
        fired,
        added_doc_ids,
        protect_doc_ids,
    )
    return (
        pool,
        added_doc_ids,
        protect_doc_ids,
        {
            "fired": True,
            "entities": fired,
            "docs_added": added_doc_ids,
            "docs_protected": protect_doc_ids,
        },
    )
