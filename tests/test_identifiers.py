from __future__ import annotations

from api.identifiers import (
    ExtractedIdentifier,
    extract_doc_identifiers,
    normalize_code,
    normalize_norm_key,
)


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
