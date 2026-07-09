from __future__ import annotations

import api.bis_editions as bis


def test_edition_pdf_urls_ordinary_and_bis():
    ordinary, bis_url = bis.edition_pdf_urls("2025-10-31", "10227", "es_es")
    assert ordinary.endswith("/2025/10/31/pdf/sumario_2025_10227_es.pdf")
    assert bis_url.endswith("/2025/10/31/pdf/sumario_2025_10227_bis_es.pdf")
    # Valencian suffix
    assert bis.edition_pdf_urls("2025-10-31", "10227", "va_va")[0].endswith("_va.pdf")


def test_search_strips_punctuation_and_caps_to_16_words(monkeypatch):
    """The portal search 440s on punctuation (commas, the N/YYYY slash) and is
    AND-semantics, so the query must be word-chars only and ~16 leading words."""
    captured = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"content": []}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["texto"] = json["texto"]
        return _Resp()

    monkeypatch.setattr(bis.requests, "post", fake_post)
    title = (
        "ORDEN 4/2025, de 30 de octubre de 2025, de la Vicepresidencia Primera y "
        "Conselleria de Servicios Sociales, Igualdad y Vivienda, por la que se modifica"
    )
    bis.search_dogv(title, "es_es")
    sent = captured["texto"]
    assert "/" not in sent and "," not in sent  # punctuation stripped
    assert len(sent.split()) <= 16  # capped


def test_resolve_signatura_ids_matches_codigo_insercion(monkeypatch):
    monkeypatch.setattr(
        bis,
        "search_dogv",
        lambda texto, lang: [
            {"id": 111, "codigoInsercion": "2025/00001"},
            {"id": 222, "codigoInsercion": "2025/99999"},  # a sibling that must not match
        ],
    )
    out = bis.resolve_signatura_ids({"2025/00001"}, {"2025/00001": "ORDEN 1/2025 ..."}, "es_es")
    assert out == {"2025/00001": 111}


def test_missing_sibling_picks_ordinary_when_ingested_is_bis(monkeypatch):
    """is_bis=True means the JSON gave us the bis -> the sibling to recover is the
    ORDINARY edition. Only signaturas absent from the corpus are resolved."""
    seen = {}
    monkeypatch.setattr(bis, "fetch_pdf", lambda url, **k: (seen.setdefault("url", url), b"pdf")[1])
    monkeypatch.setattr(bis, "parse_pdf_titles", lambda c: {"2025/1": "t1", "2025/2": "t2"})
    monkeypatch.setattr(
        bis, "existing_refs", lambda refs, lang: {"2025/1"}
    )  # 2025/1 already present
    monkeypatch.setattr(
        bis, "resolve_signatura_ids", lambda missing, titles, lang: {s: 900 for s in missing}
    )

    out = bis.missing_sibling_disposicion_ids("2025-10-31", "10227", "es_es", is_bis=True)
    assert seen["url"].endswith("sumario_2025_10227_es.pdf")  # ordinary sibling
    assert out == {"2025/2": 900}  # only the missing one


def test_missing_sibling_is_empty_when_no_sibling_pdf(monkeypatch):
    monkeypatch.setattr(bis, "fetch_pdf", lambda url, **k: None)  # 404 -> common no-sibling date
    assert bis.missing_sibling_disposicion_ids("2025-03-04", "10060", "es_es", is_bis=False) == {}
