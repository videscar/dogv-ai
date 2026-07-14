from __future__ import annotations

from api.identifiers import (
    ExtractedIdentifier,
    detect_query_identifiers,
    extract_doc_identifiers,
    normalize_code,
    normalize_norm_key,
)


def _qkeys(question: str) -> set[tuple[str, str]]:
    return {(i.id_kind, i.id_key) for i in detect_query_identifiers(question)}


def _keys(idents: list[ExtractedIdentifier], kind: str) -> set[str]:
    return {i.id_key for i in idents if i.id_kind == kind}


def test_normalize_code_collapses_separators():
    assert normalize_code("GACUJIMA/2025/36") == "gacujima/2025/36"
    assert normalize_code("gacujima 2025 36") == "gacujima/2025/36"
    assert normalize_code("GACUJIMA-2025-36") == "gacujima/2025/36"
    assert normalize_code(" Gacujima / 2025 / 36 ") == "gacujima/2025/36"


def test_normalize_norm_key():
    assert normalize_norm_key("decreto", 74, 2026) == "decreto/74/2026"


def test_extract_slash_code_from_title():
    # Real title shape (doc 90640).
    idents = extract_doc_identifiers(
        "EXTRACTO de la Resolución de 8 de junio de 2026 ... proyecto GACUJIMA/2025/36 de la UJI",
        None,
    )
    assert "gacujima/2025/36" in _keys(idents, "code")


def test_extract_long_expedient_code():
    idents = extract_doc_identifiers(
        "ACUERDO ... del expediente de resarcimiento ERESAR/2026/39R07/0008", None
    )
    assert "eresar/2026/39r07/0008" in _keys(idents, "code")


def test_extract_bdns_from_body_only():
    # Real body shape (doc 85608, IP10 gold).
    idents = extract_doc_identifiers(
        "EXTRACTO de la Resolución de 24 de marzo de 2026, de la Vicepresidencia",
        "... Primero. BDNS (Identif.): 895054. Segundo. Beneficiarios ...",
    )
    assert "895054" in _keys(idents, "bdns")


def test_extract_norm_self_identity_from_title():
    idents = extract_doc_identifiers(
        "DECRETO 74/2026, de 19 de mayo, del Consell, de nombramiento de personal directivo",
        None,
    )
    assert "decreto/74/2026" in _keys(idents, "norm")


def test_compact_body_code():
    idents = extract_doc_identifiers("Convocatoria", "... codi de projecte 24I636 per a ...")
    assert "24i636" in _keys(idents, "code")


def test_no_false_code_from_ordinary_prose():
    # Slashes without a year segment must not become codes.
    idents = extract_doc_identifiers(
        "Servicios de transporte y/o alojamiento, velocidad km/h y otros", None
    )
    assert _keys(idents, "code") == set()


def test_plain_norm_ref_not_emitted_as_code():
    # "Ley 39/2015" is a norm-ref, not a letter-prefixed code: no code emitted
    # (the leading token '39' is not a >=3-letter acronym).
    idents = extract_doc_identifiers("Se aplica la Ley 39/2015 sobre procedimiento", None)
    assert _keys(idents, "code") == set()


def test_dedup_across_title_and_body():
    idents = extract_doc_identifiers(
        "beca GACUJIMA/2025/36",
        "En la beca GACUJIMA/2025/36 se convoca ...",
    )
    codes = [i for i in idents if i.id_kind == "code" and i.id_key == "gacujima/2025/36"]
    assert len(codes) == 1
    assert codes[0].source == "title"  # title occurrence wins


def test_extract_returns_empty_for_blank():
    assert extract_doc_identifiers("", None) == []


# --- query-side detector -----------------------------------------------------


def test_detect_slash_code():
    assert ("code", "gacujima/2025/36") in _qkeys(
        "En la beca GACUJIMA/2025/36 de la Universitat Jaume I"
    )


def test_detect_space_separated_code_buried_after_words():
    # IP03: acronym buried after 'beca'; trailing 'uji dotacion' junk. The real
    # code must still be a candidate.
    assert ("code", "gacujima/2025/36") in _qkeys("beca gacujima 2025 36 uji dotacion y plazo")


def test_detect_long_expedient_code_full_length():
    # IP05: the full 4-group code, not a truncated prefix.
    assert ("code", "eresar/2026/39r07/0008") in _qkeys(
        "expedient de rescabalament ERESAR/2026/39R07/0008 per quin import"
    )


def test_detect_norm_ref():
    assert ("norm", "decreto/74/2026") in _qkeys("Qué dispone el Decreto 74/2026 del Consell")


def test_detect_bdns():
    assert ("bdns", "895054") in _qkeys("cuantía máxima con código BDNS 895054")


def test_detect_ref_column_year_first():
    keys = _qkeys("publicació del DOGV amb referència 2026/4148")
    assert ("ref", "2026/4148") in keys


def test_detect_no_identifiers_in_plain_prose():
    # No acronym+year adjacency, no norm-ref, no year-first ref -> nothing.
    assert _qkeys("¿Qué ayudas de vivienda existen para familias numerosas?") == set()


def test_detect_norm_ref_not_confused_with_ref_column():
    # '74/2026' (year-last) is a norm, not a publication ref (year-first).
    keys = _qkeys("Decreto 74/2026")
    assert ("norm", "decreto/74/2026") in keys
    assert not any(kind == "ref" for kind, _ in keys)


def test_pin_lane_scoped_to_precise_machine_identifiers():
    # The pin lane pins code/bdns/ref (verbatim machine ids) but NOT norm-refs,
    # which are human-cited and error-prone (see v2-042 premise correction).
    from agent.nodes.identifier_pin import _PINNABLE_KINDS

    assert _PINNABLE_KINDS == {"code", "bdns", "ref"}
    assert "norm" not in _PINNABLE_KINDS
