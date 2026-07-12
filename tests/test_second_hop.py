"""Offline unit tests for the second-hop merge logic and gate helpers
(no DB/network — apply_second_hop's DB-touching branches are exercised in the
targeted eval run, not here)."""

from __future__ import annotations

from agent.nodes.retrieve_pool import PoolResult
from agent.nodes.second_hop import (
    _facet_targets,
    _merge_additive,
    _split_compound_clauses,
    _uncovered_refs,
)


def _doc(doc_id: int, title: str = "") -> dict:
    return {"document_id": doc_id, "title": title, "rrf_score": 0.01}


def _pool(*doc_ids: int) -> PoolResult:
    fused = [_doc(d) for d in doc_ids]
    return PoolResult(
        fused=fused,
        top_chunks={},
        chunk_candidates=[],
        counts={},
        rrf_expanded=False,
        soft_language=False,
    )


def test_uncovered_refs_both_missing():
    q = "Compara la Orden 23/2026 y la Orden 18/2026 sobre tasas"
    fused = [_doc(1, "ORDEN 5/2025, de tasas")]
    uncovered = _uncovered_refs(q, fused)
    assert {r.num_year for r in uncovered} == {"23/2026", "18/2026"}


def test_uncovered_refs_one_covered():
    q = "Compara la Orden 23/2026 y la Orden 18/2026 sobre tasas"
    fused = [_doc(1, "ORDEN 23/2026, de tasas de examen")]
    uncovered = _uncovered_refs(q, fused)
    assert {r.num_year for r in uncovered} == {"18/2026"}


def test_uncovered_refs_single_ref_question_never_fires():
    q = "¿Qué establece la Orden 23/2026?"
    assert _uncovered_refs(q, []) == []


def test_uncovered_refs_both_covered_is_empty():
    q = "Compara la Orden 23/2026 y la Orden 18/2026 sobre tasas"
    fused = [_doc(1, "ORDEN 23/2026, de tasas"), _doc(2, "ORDEN 18/2026, de tasas")]
    assert _uncovered_refs(q, fused) == []


def test_facet_targets_skips_the_primary_facet():
    q = "¿Cuál es la ayuda al alquiler DANA y cuál el Fondo de Cooperación Municipal?"
    facets = _facet_targets(q)
    # first facet (the primary question itself) is handled by the main retrieve
    # pass already; only the trailing facets are hop candidates
    assert len(facets) <= 2


def test_facet_targets_single_clause_question_is_empty():
    assert _facet_targets("¿Qué establece el Decreto 65/2022?") == []


def test_merge_additive_appends_without_reordering():
    pool = _pool(1, 2, 3)
    hop = _pool(3, 4, 5)  # 3 already present, 4/5 new
    added = _merge_additive(pool, hop.fused, hop)
    assert added == [4, 5]
    assert [d["document_id"] for d in pool.fused] == [1, 2, 3, 4, 5]


def test_merge_additive_carries_chunk_evidence_for_new_docs_only():
    pool = _pool(1)
    hop = PoolResult(
        fused=[_doc(2)],
        top_chunks={1: [{"text": "should not overwrite"}], 2: [{"text": "hop chunk"}]},
        chunk_candidates=[
            {"document_id": 2, "text": "hop chunk"},
            {"document_id": 9, "text": "unrelated"},
        ],
        counts={},
        rrf_expanded=False,
        soft_language=False,
    )
    added = _merge_additive(pool, hop.fused, hop)
    assert added == [2]
    assert pool.top_chunks[2] == [{"text": "hop chunk"}]
    assert 1 not in pool.top_chunks  # existing doc's evidence untouched
    assert pool.chunk_candidates == [{"document_id": 2, "text": "hop chunk"}]


def test_merge_additive_noop_when_all_already_present():
    pool = _pool(1, 2)
    hop = _pool(1, 2)
    added = _merge_additive(pool, hop.fused, hop)
    assert added == []
    assert [d["document_id"] for d in pool.fused] == [1, 2]


# --- hop-local compound-clause splitter -------------------------------------


def test_split_valencian_i_quin_second_clause():
    q = (
        "Per a l'OPE 2026, quina taxa d'examen es paga (Ordre 23/2026) i quin és "
        "el sou base del subgrup A1 (taules 2026) si s'aprova la plaça?"
    )
    halves = _split_compound_clauses(q)
    assert len(halves) == 2
    assert "sou base del subgrup A1" in halves[1]
    assert "taxa d'examen" in halves[0]


def test_split_prefers_last_conjunction_when_no_marker():
    # earlier "y" joins words INSIDE the first entity ("la DANA y los municipios");
    # the list-final "y el Fondo..." names the second one
    q = (
        "¿Qué dos tipos de ayuda relacionadas con la DANA y los municipios aparecen: "
        "las de alquiler de vivienda y el Fondo de Cooperación Municipal? "
        "Indica el importe comprometido de cada una."
    )
    halves = _split_compound_clauses(q)
    assert len(halves) == 2
    assert halves[1].startswith("el Fondo de Cooperación Municipal")


def test_split_rejects_anaphoric_second_clause():
    # second clause asks another fact about the SAME doc: no entity anchor
    q = (
        "¿Cuál es el importe del sueldo base mensual del subgrupo A1 según las "
        "tablas retributivas de la Generalitat de 2026, y qué subida salarial aplican?"
    )
    assert _split_compound_clauses(q) == []


def test_split_rejects_anaphoric_second_clause_valencian():
    q = (
        "Segons la resolució de concessió d'ajudes CMIART de febrer de 2026, fins a "
        "quina data han de justificar les persones beneficiàries i quin percentatge "
        "de suport s'aplica?"
    )
    assert _split_compound_clauses(q) == []


def test_split_no_conjunction_is_empty():
    assert _split_compound_clauses("¿Qué establece el Decreto 65/2022?") == []
    assert _split_compound_clauses("") == []


def test_facet_targets_puts_second_clause_first():
    q = (
        "En l'àmbit dels nomenaments universitaris de 2026, qui és el nou gerent de "
        "la Universitat de València i qui s'ha nomenat vocal del Ple del Consell "
        "Valencià d'Universitats?"
    )
    facets = _facet_targets(q)
    assert facets and "vocal del Ple" in facets[0]
