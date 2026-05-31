from __future__ import annotations

from api.reader_extractors import _numeric_evidence, _program_total_evidence


_CONV = {
    "document_id": 85222,
    "doc_subkind": "Convocatoria",
    "chunks": [
        "Distribución de los premios ... (207.000 €) del presupuesto de la Generalitat.",
        "La concesión de los premios para 2026 se efectuará a cargo de la sección G0105.",
    ],
}
_CONCESION = {
    "document_id": 87141,
    "doc_subkind": "Resultados",
    "chunks": ["L'import total dels premis ascendix a 157.000 € (cent cinquanta-set mil euros)."],
}


def test_numeric_evidence_regex_matches_real_amounts():
    # Regression: the regex was r"\b\\d[\\d\\.,]*\\b" (literal backslashes) and matched
    # nothing, so amount questions had no numeric backstop.
    docs = [{"document_id": 1, "chunks": ["La cuantia es de 1.366,74 euros mensuales."]}]
    out = _numeric_evidence("¿Cual es el importe?", docs)
    assert out and out[0]["doc_id"] == 1
    assert "1.366,74" in out[0]["quote"]


def test_program_total_prefers_convocatoria_over_concesion():
    out = _program_total_evidence(
        "¿Qué importe total máximo tiene la convocatoria de los premios?",
        [_CONV, _CONCESION],
    )
    assert out, "program-total evidence should fire for a convocatoria-total question"
    assert out[0]["doc_id"] == 85222
    assert "207.000" in out[0]["quote"]


def test_program_total_noop_for_individual_award_question():
    # Asking what was awarded to a specific beneficiary must NOT trigger the
    # convocatoria preference (avoids hijacking concesión questions).
    out = _program_total_evidence(
        "¿Qué importe se concedió al Centre Privat Pureza de María?",
        [_CONV, _CONCESION],
    )
    assert out == []


def test_program_total_noop_without_convocatoria_doc():
    out = _program_total_evidence(
        "¿Qué importe total máximo tiene la convocatoria?",
        [_CONCESION],
    )
    assert out == []
