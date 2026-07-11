"""Offline unit tests for doc_reference extraction (no DB/network)."""

from __future__ import annotations

from api.doc_references import ExtractedReference, extract_references, resolve_target_document_id


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDb:
    """Stub SQLAlchemy session: first query always returns `first_rows`, any
    follow-up (organismo/topic disambiguation) query returns `second_rows`."""

    def __init__(self, first_rows, second_rows=None):
        self.first_rows = first_rows
        self.second_rows = second_rows or []
        self.calls = 0

    def execute(self, *args, **kwargs):
        self.calls += 1
        return _FakeResult(self.first_rows if self.calls == 1 else self.second_rows)


def test_extracts_num_year_reference_from_title_es():
    title = "ORDEN 5/2026, de 3 de marzo, por la que se resuelve la convocatoria aprobada por la Orden 12/2025, de 4 de enero."
    refs = extract_references(title, None)
    kinds = {(r.ref_tipo, r.ref_numero, r.ref_anyo): r.ref_kind for r in refs}
    assert ("orden", 12, 2025) in kinds
    assert kinds[("orden", 12, 2025)] == "resuelve"
    # the document's own identifier (Orden 5/2026) must not be emitted
    assert ("orden", 5, 2026) not in kinds


def test_extracts_num_year_reference_from_title_va():
    title = "ORDRE 5/2026, de 3 de març, per la qual es modifica el Decret 185/2018, del Consell."
    refs = extract_references(title, None)
    kinds = {(r.ref_tipo, r.ref_numero, r.ref_anyo): r.ref_kind for r in refs}
    assert ("decreto", 185, 2018) in kinds
    assert kinds[("decreto", 185, 2018)] == "modifica"
    assert ("orden", 5, 2026) not in kinds


def test_self_reference_skipped_even_when_repeated_in_body():
    title = "RESOLUCIÓN de 21 de mayo de 2026, de la Subsecretaría, de concesión de un premio."
    # the body echoes the title verbatim at the top, as DOGV documents do
    body = (
        title
        + "\nVicepresidencia Segunda y Conselleria de Presidencia\nEl Decreto 38/2022 creó el premio."
    )
    refs = extract_references(title, body)
    assert all(not (r.ref_tipo == "resolucion" and r.disp_day == 21) for r in refs)
    # the genuine citation (Decreto 38/2022) is still extracted
    assert any((r.ref_tipo, r.ref_numero, r.ref_anyo) == ("decreto", 38, 2022) for r in refs)


def test_deroga_kind_classification():
    title = "DECRETO 10/2026, de 1 de enero, del Consell, por el que se deroga el Decreto 20/2019."
    refs = extract_references(title, None)
    target = next(r for r in refs if (r.ref_numero, r.ref_anyo) == (20, 2019))
    assert target.ref_kind == "deroga"


def test_corrige_kind_classification():
    title = (
        "RESOLUCIÓN de 5 de mayo de 2026, corrección de errores de la Orden 8/2026, de 1 de marzo."
    )
    refs = extract_references(title, None)
    target = next(r for r in refs if (r.ref_tipo, r.ref_numero, r.ref_anyo) == ("orden", 8, 2026))
    assert target.ref_kind == "corrige"


def test_cita_is_default_kind_without_governing_verb():
    body = "Vista la Ley 4/2021, de 16 de abril, de la funció pública valenciana, se acuerda..."
    refs = extract_references("RESOLUCIÓN de 1 de enero de 2026, de un asunto cualquiera.", body)
    target = next(r for r in refs if (r.ref_tipo, r.ref_numero, r.ref_anyo) == ("ley", 4, 2021))
    assert target.ref_kind == "cita"


def test_resolucion_referenced_by_date_not_number():
    title = "RESOLUCIÓN de 21 de mayo de 2026, de la Subsecretaría, de concesión de los premios."
    body = (
        title + "\nMediante la Resolución de 12 de marzo de 2026, de la Vicepresidencia Segunda y "
        "Conselleria de Presidencia (DOGV núm. 10326, de 20.03.2026), se convocaron los premios."
    )
    refs = extract_references(title, body)
    date_refs = [r for r in refs if r.ref_tipo == "resolucion" and r.disp_day == 12]
    assert date_refs, "expected a date-based Resolución reference to be extracted"
    assert date_refs[0].disp_month == "marzo"
    assert date_refs[0].disp_year == 2026
    assert date_refs[0].ref_kind == "convoca"
    assert date_refs[0].disp_organismo is not None
    assert "presidencia" in date_refs[0].disp_organismo.lower()


def test_no_references_when_body_empty_and_title_has_no_numyear():
    refs = extract_references("RESOLUCIÓN sobre un asunto sin fecha ni número.", None)
    assert refs == []


def test_ambiguous_orden_number_resolves_to_none():
    # Orden numbers repeat across consellerias within a year: two unrelated
    # in-corpus matches, no organismo hint, no title overlap -> stay hands-off.
    ref = ExtractedReference(
        ref_tipo="orden", ref_numero=7, ref_anyo=2026, ref_kind="cita", raw_text=""
    )
    db = _FakeDb(first_rows=[(101, "Conselleria A"), (202, "Conselleria B")])
    target = resolve_target_document_id(
        db, source_document_id=999, ref=ref, source_title="Ayudas al comercio"
    )
    assert target is None


def test_unambiguous_orden_number_resolves():
    ref = ExtractedReference(
        ref_tipo="orden", ref_numero=7, ref_anyo=2026, ref_kind="cita", raw_text=""
    )
    db = _FakeDb(first_rows=[(101, "Conselleria A")])
    target = resolve_target_document_id(
        db, source_document_id=999, ref=ref, source_title="Ayudas al comercio"
    )
    assert target == 101


def test_ref_key_is_stable_and_non_null():
    title = "ORDEN 5/2026, de 3 de marzo, por la que se modifica el Decreto 3/2020."
    refs = extract_references(title, None)
    assert refs
    for r in refs:
        assert r.ref_key
        assert "None" not in r.ref_key or r.ref_numero is None
