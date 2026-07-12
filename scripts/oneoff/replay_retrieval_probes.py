"""Replay the retrieval stage for the grounded-probe misses and report where
each gold document surfaces (per lane, fused pool, post-ladder pool).

Feeds the planner intent recorded in qa_traces through the same helpers the
live pipeline uses (build_bm25_queries -> compute_pool -> relaxation ladder),
so lane-level behaviour matches prod without paying the LLM planner hop.

Usage: python scripts/oneoff/replay_retrieval_probes.py [case ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date

import _path  # noqa: F401

from agent.nodes.retrieve import _build_facet_specs
from agent.nodes.retrieve_pool import (
    PoolQuery,
    apply_relaxation_ladder,
    compute_pool,
    run_all_facets,
)
from api.config import enabled_lanes, get_settings
from api.embed import EmbedClient
from api.query_expansion import build_bm25_queries
from api.retrieval import RetrievalFilters

settings = get_settings()


@dataclass
class Case:
    name: str
    question: str
    intent: dict
    filters: RetrievalFilters
    gold: list[int]


CASES = [
    Case(
        name="P03-director",
        question=(
            "¿Quién fue nombrado director de la Oficina Autonómica de Auditoría e "
            "Inspección Sanitaria de la Comunitat Valenciana?"
        ),
        intent={
            "doc_kind": "Empleo Publico",
            "entities": {
                "cuerpo": "Oficina Autonómica de Auditoría e Inspección Sanitaria",
                "organismo": "Generalitat Valenciana",
            },
            "keywords": [
                "nombramiento",
                "director",
                "Oficina Autonómica de Auditoría e Inspección Sanitaria",
                "Comunitat Valenciana",
            ],
            "language": "es",
        },
        filters=RetrievalFilters(language="es_es"),
        gold=[103980, 104023],
    ),
    Case(
        name="P08-herencia-fecha",
        question=(
            "¿Sobre la herencia intestada de Joaquim Bahilo Ariso, qué acordó la "
            "Dirección General de Patrimonio en julio de 2025?"
        ),
        intent={
            "doc_kind": "Otros",
            "entities": {"organismo": "Dirección General de Patrimonio"},
            "keywords": [
                "herencia intestada",
                "Joaquim Bahilo Ariso",
                "Dirección General de Patrimonio",
            ],
            "language": "es",
        },
        # until_date reflects the intent node's publication-lag allowance
        # (2025-07-31 + ask_intent_date_pub_lag_days).
        filters=RetrievalFilters(
            language="es_es",
            since_date=date(2025, 7, 1),
            until_date=date(2025, 9, 29),
        ),
        gold=[18441, 40342],
    ),
    Case(
        name="P09-toma-posesion",
        question=(
            "¿Qué día tendrá lugar la toma de posesión de quienes obtuvieron destino "
            "en la convocatoria 9/24, escala medicina A1-07-03?"
        ),
        intent={
            "doc_kind": "Empleo Publico",
            "entities": {"convocatoria": "9/24", "cuerpo": "escala medicina A1-07-03"},
            "keywords": [
                "toma de posesión",
                "convocatoria 9/24",
                "escala medicina",
                "A1-07-03",
            ],
            "language": "es",
        },
        filters=RetrievalFilters(language="es_es"),
        gold=[110854, 110918],
    ),
    Case(
        name="P13-codi-import",
        question=(
            "Quin import màxim finançava el programa CODI de capacitació digital per "
            "a l'alumnat convocat a l'agost de 2024?"
        ),
        intent={
            "doc_kind": "Ayudas",
            "entities": {"programa": "CODI"},
            "keywords": [
                "programa CODI",
                "capacitació digital",
                "import màxim",
                "alumnat",
            ],
            "language": "ca",
        },
        # "a l'agost de 2024" now yields a month window via _infer_month_range,
        # plus the intent node's publication-lag allowance on until_date.
        filters=RetrievalFilters(
            language="va_va",
            since_date=date(2024, 8, 1),
            until_date=date(2024, 10, 30),
        ),
        gold=[95359, 95321],
    ),
    Case(
        name="P14-codi-gratificacio",
        question=(
            "Quina gratificació rep el personal docent per cada acció formativa "
            "impartida del programa CODI?"
        ),
        intent={
            "doc_kind": "Ayudas",
            "entities": {"programa": "CODI"},
            "keywords": [
                "gratificació",
                "personal docent",
                "acció formativa",
                "programa CODI",
            ],
            "language": "ca",
        },
        filters=RetrievalFilters(language="va_va"),
        gold=[95359, 95321],
    ),
    Case(
        name="P17-lliria",
        question=(
            "¿A qué ayuntamiento se le redujo la ayuda de 519.260 euros solicitados a "
            "175.000 euros en las subvenciones de cohesión entre destinos turísticos "
            "de la Comunitat Valenciana?"
        ),
        intent={
            "doc_kind": "Subvenciones",
            "entities": {},
            "keywords": [
                "subvenciones",
                "cohesión entre destinos turísticos",
                "519.260 euros",
                "ayuntamiento",
            ],
            "language": "es",
        },
        filters=RetrievalFilters(language="es_es"),
        gold=[88797, 86305],
    ),
]


def _rank_of(rows: list[dict], gold: set[int]) -> str:
    for idx, row in enumerate(rows, start=1):
        doc_id = row.get("document_id")
        if doc_id is not None and int(doc_id) in gold:
            return f"#{idx}/{len(rows)} (doc {int(doc_id)})"
    return f"absent/{len(rows)}"


def replay(case: Case, live_intent: bool = False) -> None:
    print(f"\n=== {case.name} ===")
    client = EmbedClient()
    embedding = client.embed_batch([case.question])[0]
    intent, expansion = case.intent, {}
    if live_intent:
        from api.intent import analyze_intent_and_expand

        intent, expansion = analyze_intent_and_expand(case.question)
        print(f"live intent  = {intent}")
        print(f"expansion    = {expansion}")
    bm25_query, bm25_strict = build_bm25_queries(case.question, intent, expansion=expansion)
    print(f"bm25_query   = {bm25_query!r}")
    print(f"bm25_strict  = {bm25_strict!r}")

    query = PoolQuery(
        lanes=enabled_lanes(settings),
        bm25_query=bm25_query,
        bm25_specs=_build_facet_specs(
            case.question,
            bm25_query,
            bm25_strict,
            case.intent,
            max_facets=max(1, getattr(settings, "ask_max_facets", 3)),
        ),
        query_embedding=embedding,
        hyde_embedding=None,
        bm25_limit=max(10, settings.ask_bm25_max_docs),
        min_docs=max(1, getattr(settings, "ask_min_docs", 3)),
        expand_ratio=getattr(settings, "ask_rrf_expand_margin_ratio", 0.12),
        expand_probe=getattr(settings, "ask_rrf_margin_probe", 5),
        expand_docs=getattr(settings, "ask_max_docs_expand", 20),
        anchors_enabled=getattr(settings, "ask_semantic_anchor_enabled", False),
    )
    gold = set(case.gold)

    (
        vector_hits,
        bm25_hits,
        bm25_strict_hits,
        bm25_title_hits,
        title_hits,
        title_lexical_hits,
        _chunks,
        counts,
    ) = run_all_facets(query, case.filters)
    print(f"lane counts  = {counts}")
    print(f"vector        gold {_rank_of(vector_hits, gold)}")
    print(f"bm25(chunk)   gold {_rank_of(bm25_hits, gold)}")
    print(f"bm25(strict)  gold {_rank_of(bm25_strict_hits, gold)}")
    print(f"bm25(title)   gold {_rank_of(bm25_title_hits, gold)}")
    print(f"title(vector) gold {_rank_of(title_hits, gold)}")
    print(f"title(lex)    gold {_rank_of(title_lexical_hits, gold)}")

    pool = compute_pool(query, case.filters)
    print(f"fused         gold {_rank_of(pool.fused, gold)}")
    pool, final_filters, fallbacks = apply_relaxation_ladder(
        query, pool, case.filters, preserve_temporal_window=False, allow_margin_fallback=False
    )
    print(f"post-ladder   gold {_rank_of(pool.fused, gold)}  fallbacks={fallbacks}")
    print(f"final filters = {final_filters}")

    # Rerank stage minus the LLM: similarity re-sort, max_candidates cut, RC1.
    from agent.nodes.rerank import _apply_edition_recency, _doc_similarity_scores

    ids = [int(c["document_id"]) for c in pool.fused]
    sims = _doc_similarity_scores(embedding, ids)
    resorted = sorted(
        pool.fused,
        key=lambda c: (
            sims.get(int(c["document_id"]), -1.0),
            float(c.get("rrf_score") or 0.0),
        ),
        reverse=True,
    )
    max_candidates = getattr(settings, "ask_rerank_max_candidates", 10)
    cut = resorted[:max_candidates]
    print(f"sim-cut top{max_candidates}  gold {_rank_of(cut, gold)}")
    state = {"filters": final_filters, "norm_pin_doc_ids": [], "second_hop_protect_ids": []}
    doc_ids = [int(c["document_id"]) for c in cut]
    kept, _pool2, dropped = _apply_edition_recency(state, doc_ids, pool.fused, None, "replay")
    gold_kept = [d for d in kept if d in gold]
    print(f"post-RC1      kept={kept} dropped={sorted(dropped)} gold_kept={gold_kept}")


def main() -> None:
    wanted = set(sys.argv[1:])
    for case in CASES:
        if wanted and case.name not in wanted:
            continue
        replay(case)


if __name__ == "__main__":
    main()
