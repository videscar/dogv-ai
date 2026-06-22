"""Single-citation collapse for explicit single-disposition queries (offline)."""

from __future__ import annotations

from agent.nodes.answer import _collapse_to_principal

Q_REF = "¿En qué fecha se firmó el Decreto 185/2018?"
Q_VAGUE = "¿Qué ayudas hay para agricultores?"

CITS = [
    {"document_id": 1, "title": "DECRETO 185/2018, de 19 de octubre, del Consell, de declaración..."},
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
