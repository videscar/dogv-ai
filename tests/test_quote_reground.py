from __future__ import annotations

from api.reader import _reground_evidence

TABLE = (
    "Confrides Decaimiento No presenta solicitud de ayuda "
    "Cox Decaimiento No presenta solicitud de ayuda "
    "Daya Nueva Renuncia expresa Renuncia "
    "el Campello Decaimiento No presenta solicitud de ayuda "
    "Ibi Denegacion No cumple con la Sindicatura de Comptes "
    "Jacarilla Decaimiento No presenta solicitud de ayuda"
)


def _docs():
    return [
        {"document_id": 112345, "chunks": ["RESOLUCION de 28 de junio de 2026 titulo", TABLE]},
        {"document_id": 12417, "chunks": ["Los terminos municipales de Alfara del Patriarca quedan exentos"]},
    ]


def test_verbatim_quote_passes_through():
    ev = [{"doc_id": 112345, "quote": "Daya Nueva Renuncia expresa Renuncia", "detail": "d"}]
    assert _reground_evidence(ev, _docs()) == ev


def test_stitched_quote_is_replaced_with_source_window():
    stitched = "Daya Nueva Renuncia expresa Renuncia... Cox Decaimiento No presenta solicitud"
    ev = [{"doc_id": 112345, "quote": stitched, "detail": "d"}]
    out = _reground_evidence(ev, _docs())
    assert len(out) == 1
    assert out[0]["doc_id"] == 112345
    # replaced by the surrounding table window: majority cause + distinct case visible
    assert "Sindicatura de Comptes" in out[0]["quote"]
    assert out[0]["quote"].count("Decaimiento") >= 3


def test_unlocatable_quote_kept_as_is():
    ev = [{"doc_id": 112345, "quote": "texto totalmente inventado por el lector... otra parte inventada", "detail": "d"}]
    assert _reground_evidence(ev, _docs()) == ev


def test_quote_for_unknown_doc_kept_as_is():
    ev = [{"doc_id": 999, "quote": "cualquier cosa... otra cosa distinta aqui", "detail": "d"}]
    assert _reground_evidence(ev, _docs()) == ev


def test_whitespace_differences_still_count_as_verbatim():
    ev = [{"doc_id": 112345, "quote": "Daya  Nueva\nRenuncia   expresa Renuncia", "detail": "d"}]
    out = _reground_evidence(ev, _docs())
    # normalized-ws match -> treated as verbatim, original item returned
    assert out == ev


def test_doc_attribution_never_changes():
    # fragment text lives in doc 112345, but the quote claims 12417 -> only 12417's
    # chunks are searched; no cross-doc reattachment.
    stitched = "Cox Decaimiento No presenta solicitud... Ibi Denegacion No cumple"
    ev = [{"doc_id": 12417, "quote": stitched, "detail": "d"}]
    out = _reground_evidence(ev, _docs())
    assert out == ev
