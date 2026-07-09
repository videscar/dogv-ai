"""Single-citation collapse for explicit single-disposition queries (offline)."""

from __future__ import annotations

from agent.nodes.answer import _collapse_to_principal, _norm_target_doc_id
from api.dogv_resolver import (
    parse_named_norm_target,
    title_primary_tipo,
)

Q_REF = "¿En qué fecha se firmó el Decreto 185/2018?"
Q_VAGUE = "¿Qué ayudas hay para agricultores?"

CITS = [
    {
        "document_id": 1,
        "title": "DECRETO 185/2018, de 19 de octubre, del Consell, de declaración...",
    },
    {"document_id": 2, "title": "CORRECCIÓN de errores del Decreto 185/2025..."},
    {"document_id": 3, "title": "RESOLUCIÓN tangencial sin relación"},
]


def test_collapse_keeps_only_the_principal():
    out = _collapse_to_principal(CITS, Q_REF, None)
    assert [c["document_id"] for c in out] == [1]


def test_collapse_untouched_when_principal_absent():
    out = _collapse_to_principal(CITS[1:], Q_REF, None)
    assert len(out) == 2  # principal not present -> don't blank citations


def test_collapse_untouched_for_non_reference_query():
    out = _collapse_to_principal(CITS, Q_VAGUE, None)
    assert len(out) == 3


def test_collapse_single_citation_noop():
    out = _collapse_to_principal(CITS[:1], Q_REF, None)
    assert out == CITS[:1]


def test_collapse_prefers_pinned_among_multiple_matches():
    cits = [
        {"document_id": 10, "title": "DECRET 185/2018, de 19 d'octubre, del Consell..."},
        {"document_id": 11, "title": "DECRETO 185/2018, de 19 de octubre, del Consell..."},
    ]
    out = _collapse_to_principal(cits, Q_REF, ondemand_doc_id=11)
    assert [c["document_id"] for c in out] == [11]


# ---- norm-target citation recovery (the tester "main reference missing" fix) ----

# Named (no-number) target: synthesis cited a doc that only mentions the law, but
# the law itself is in the read set -> recover it as the principal.
LEY_FP = "LEY 4/2021, de 16 de abril, de la Generalitat, de la Función Pública Valenciana"
ORDEN_FP = "ORDEN 33/2024, de 17 de junio, de la Conselleria de Hacienda (temario)"
Q_NAMED_FP = "¿Qué regula la Ley de la Función Pública Valenciana?"


def _state(question, candidates, grounded_ids, ondemand=None):
    return {
        "question": question,
        "evidence": [{"doc_id": d, "quote": "x"} for d in grounded_ids],
        "full_docs": [],
        "candidate_docs": candidates,
        "ondemand_doc_id": ondemand,
    }


def test_named_target_recovers_principal_norm():
    cands = [
        {"document_id": 91449, "title": LEY_FP},
        {"document_id": 91687, "title": ORDEN_FP},
    ]
    state = _state(Q_NAMED_FP, cands, grounded_ids=[91449, 91687])
    assert _norm_target_doc_id(state) == 91449


def test_named_target_none_when_norm_not_in_read_set():
    # Only the tangential Orden is grounded; the law is a candidate but not read.
    cands = [
        {"document_id": 91449, "title": LEY_FP},
        {"document_id": 91687, "title": ORDEN_FP},
    ]
    state = _state(Q_NAMED_FP, cands, grounded_ids=[91687])
    assert _norm_target_doc_id(state) is None


def test_named_target_ambiguous_distinct_norms_bail():
    q = "¿Qué regula la ley de protección ambiental sostenible?"
    cands = [
        {
            "document_id": 1,
            "title": "LEY 1/2020, de protección ambiental sostenible de la Comunitat",
        },
        {"document_id": 2, "title": "LEY 2/2021, de protección ambiental sostenible valenciana"},
    ]
    state = _state(q, cands, grounded_ids=[1, 2])
    assert _norm_target_doc_id(state) is None


def test_numbered_target_prefers_pinned_twin():
    cands = [
        {
            "document_id": 10,
            "title": "DECRET 185/2018, de 19 d'octubre, del Consell, de declaració...",
        },
        {
            "document_id": 11,
            "title": "DECRETO 185/2018, de 19 de octubre, del Consell, de declaración...",
        },
    ]
    state = _state(Q_REF, cands, grounded_ids=[10, 11], ondemand=11)
    assert _norm_target_doc_id(state) == 11


def test_non_norm_query_returns_none():
    cands = [{"document_id": 5, "title": "RESOLUCIÓN de ayudas a agricultores"}]
    state = _state(Q_VAGUE, cands, grounded_ids=[5])
    assert _norm_target_doc_id(state) is None


def test_parse_named_norm_target_basics():
    t = parse_named_norm_target(Q_NAMED_FP)
    assert t is not None and t.tipo == "ley"
    # numbered references are handled by parse_reference, not here
    assert parse_named_norm_target("¿Qué regula la Ley 4/2021?") is None


def test_title_primary_tipo_excludes_secondary_norms():
    assert title_primary_tipo(LEY_FP) == "ley"
    assert title_primary_tipo("DECRETO LEY 4/2025, de modificación...") == "decreto ley"
    assert title_primary_tipo("RESOLUCIÓN de 2 de abril de 2025...") is None
    assert title_primary_tipo("CORRECCIÓN DE ERRORES de la Orden 19/2025...") is None
