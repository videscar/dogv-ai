"""Offline unit tests for parse_references (plural) — the second-hop gate's
explicit multi-ref trigger."""

from __future__ import annotations

from api.dogv_resolver import parse_references


def test_two_ordenes_same_year():
    q = "Compara la Orden 23/2026 y la Orden 18/2026 sobre tasas de examen"
    refs = parse_references(q)
    assert [(r.tipo, r.numero, r.anyo) for r in refs] == [
        ("orden", 23, 2026),
        ("orden", 18, 2026),
    ]


def test_single_ref_returns_one():
    refs = parse_references("¿Qué establece el Decreto 65/2022?")
    assert len(refs) == 1
    assert (refs[0].tipo, refs[0].numero, refs[0].anyo) == ("decreto", 65, 2022)


def test_no_ref_returns_empty():
    assert parse_references("¿Qué dice la Ley de Transparencia?") == []
    assert parse_references("") == []


def test_duplicate_ref_deduped():
    q = "La Orden 5/2020 modifica la Orden 5/2020 anterior"
    refs = parse_references(q)
    assert len(refs) == 1


def test_distinct_tipos_same_num_year_kept_separate():
    q = "Compara el Decreto 3/2021 con la Orden 3/2021"
    refs = parse_references(q)
    assert {(r.tipo, r.numero, r.anyo) for r in refs} == {
        ("decreto", 3, 2021),
        ("orden", 3, 2021),
    }


def test_topic_terms_shared_across_refs():
    q = "Compara la Orden 23/2026 y la Orden 18/2026 sobre tasas de examen A1"
    refs = parse_references(q)
    assert refs[0].topic_terms == refs[1].topic_terms
    assert "examen" in refs[0].topic_terms
