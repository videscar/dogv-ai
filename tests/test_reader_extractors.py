from __future__ import annotations

from api.reader_extractors import (
    _numeric_evidence,
    _program_total_evidence,
    _subject_amount_evidence,
)


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


# --- W6: within-doc figure/row extraction ----------------------------------

_AUDIOVISUAL = {
    "document_id": 76674,
    "doc_subkind": "Convocatoria",
    "chunks": [
        "Les ajudes per a la producció d'obres audiovisuals es distribuïxen per "
        "modalitats i submodalitats segons la naturalesa del projecte presentat, "
        "dins dels límits pressupostaris establits per a cada anualitat. "
        "a) Modalitat. Animació: 1.070.000 euros, amb un import màxim per ajuda. "
        "3. Videojocs: 30.000 euros, amb un import màxim de 30.000 euros. "
        "b) Modalitat. Ficció: 4.860.000 euros Submodalitats per a llargmetratges "
        "i minisèries de ficció que millor s'ajusten a l'objecte de la convocatòria.",
    ],
}
_PER_ENTITY_ROWS = {
    "document_id": 89632,
    "doc_subkind": "Resultados",
    "chunks": [
        "Província d'Alacant EXPEDIENT ENTITAT CIF SUBVENCIÓ LOCALITAT "
        "DPIR-2026-03-012 CLUB BALONCESTO ATLÉTICO ALBATERA G54859657 369,00 € "
        "ALBATERA G54856687 5.230,00 € COMUNITAT VALENCIANA",
    ],
}
_JOCS_CONCESION = {
    "document_id": 89632,
    "doc_subkind": "Resultados",
    "chunks": [
        "Concedir a les entitats que s'indiquen les subvencions següents per un "
        "import total de 500.000 euros, amb càrrec al crèdit existent.",
    ],
}


def test_subject_amount_pins_the_modality_asked_not_a_neighbour():
    # The chunk lists Videojocs (30.000) and Ficció (4.860.000) figures; the
    # question asks the ficció dotación. The label-before-amount binding must
    # pin the ficció figure, not the neighbouring videojocs one.
    out = _subject_amount_evidence(
        "Quina dotació total tenen les subvencions per a la producció d'obres "
        "audiovisuals (anualitats 2026-2028) en la modalitat de ficció?",
        [_AUDIOVISUAL],
    )
    assert out and "4.860.000" in out[0]["quote"]
    assert "30.000 euros, amb" not in out[0]["quote"].split("4.860.000")[0][-20:]


def test_subject_amount_noop_without_amount_intent():
    out = _subject_amount_evidence(
        "¿Quién firma la resolución de la modalidad de ficción?", [_AUDIOVISUAL]
    )
    assert out == []


def test_subject_amount_skips_per_entity_award_rows():
    # A generic place word in the question (comunitat/valenciana) must NOT pin a
    # per-beneficiary award row — those list-like chunks are skipped.
    out = _subject_amount_evidence(
        "Quina és la quantia total de les subvencions a les entitats de la "
        "Comunitat Valenciana en el nivell d'iniciació al rendiment?",
        [_PER_ENTITY_ROWS],
    )
    assert out == []


def test_program_total_uses_concesion_for_awarded_total_question():
    # "subvencions concedides ... import total" → the figure lives in the
    # concesión (Resultados) doc, so the convocatoria-only guard is relaxed.
    out = _program_total_evidence(
        "Quina és la quantia total de les subvencions concedides a les entitats "
        "dels Jocs Esportius en el nivell d'iniciació al rendiment?",
        [_JOCS_CONCESION],
    )
    assert out and "500.000" in out[0]["quote"]


def test_program_total_concesion_still_noop_without_concedid_intent():
    # Same Resultados doc, but a plain convocatoria-total question must NOT pull
    # the concesión figure (preserves the W2 convocatoria-vs-concesión rule).
    out = _program_total_evidence(
        "¿Qué importe total máximo tiene la convocatoria de los Jocs Esportius?",
        [_JOCS_CONCESION],
    )
    assert out == []
