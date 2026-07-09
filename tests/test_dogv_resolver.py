"""Offline unit tests for the on-demand resolver's pure logic (no network)."""

from __future__ import annotations

from api.dogv_resolver import (
    Reference,
    _query_lang,
    _safe_page_size,
    _title_language,
    _title_matches_ref,
    parse_reference,
)


def test_parse_reference_basic_decreto():
    r = parse_reference("¿Cuál es el objeto del Decreto 65/2022?")
    assert r is not None
    assert (r.tipo, r.numero, r.anyo) == ("decreto", 65, 2022)
    assert r.num_year == "65/2022"


def test_parse_reference_ley_and_topic_terms():
    r = parse_reference("¿Qué regula la Ley 5/2018 de la Generalitat de la Huerta de Valencia?")
    assert r is not None
    assert r.tipo == "ley" and r.num_year == "5/2018"
    # topic terms keep the disambiguating content, drop tipo/stopwords/numbers
    assert "huerta" in r.topic_terms
    assert "5/2018" not in r.topic_terms


def test_parse_reference_orden_keeps_conselleria_terms():
    r = parse_reference("¿Qué establece la Orden 3/2017 de Agricultura sobre ayudas PAC?")
    assert r is not None and r.tipo == "orden"
    assert "agricultura" in r.topic_terms


def test_parse_reference_requires_number_year():
    assert parse_reference("¿Qué dice la Ley de Transparencia?") is None
    assert parse_reference("") is None


def test_parse_reference_decreto_ley():
    r = parse_reference("¿Qué establece el Decreto Ley 4/2025?")
    assert r is not None and r.tipo == "decreto ley"


def test_search_text_is_bare_reference():
    # topic terms must NOT enter the query (DOGV search is AND-semantics)
    r = parse_reference("¿Qué regula la Orden 5/2019 de Medio Ambiente sobre gestión de residuos?")
    assert r.search_text() == "orden 5/2019"


def test_title_matches_ref_respects_tipo_and_number():
    ref = Reference(tipo="decreto", numero=3, anyo=2020)
    assert _title_matches_ref("DECRETO 3/2020, de 10 de enero, del Consell, de cese...", ref)
    # plain "decreto" must NOT swallow "DECRETO LEY"
    assert not _title_matches_ref("DECRETO LEY 3/2020, de 10 de abril, de adopción...", ref)
    # wrong number/year
    assert not _title_matches_ref("DECRETO 4/2020, de 1 de enero...", ref)
    # valencian spelling still matches the tipo
    assert _title_matches_ref("DECRET 3/2020, de 10 de gener, del Consell...", ref)


def test_title_matches_decreto_ley_tipo():
    ref = Reference(tipo="decreto ley", numero=4, anyo=2025)
    assert _title_matches_ref("DECRETO LEY 4/2025, de 4 de febrero, del Consell...", ref)
    assert not _title_matches_ref("DECRETO 4/2025, de 4 de febrero...", ref)


def test_title_language():
    assert _title_language("DECRETO 3/2020, de 10 de enero...") == "es"
    assert _title_language("DECRET 3/2020, de 10 de gener...") == "va"
    assert _title_language("LLEI 5/2018...") == "va"
    assert _title_language("RESOLUCIÓN de 2 de abril...") == "es"
    assert _title_language("RESOLUCIÓ de 12 de novembre...") == "va"


def test_safe_page_size_avoids_bigdecimal_crash():
    # only 2/5-smooth sizes (total/size always terminates) are allowed
    assert _safe_page_size(3) == 10
    assert _safe_page_size(30) == 50  # 30 has a factor of 3 -> snap up
    assert _safe_page_size(50) == 50
    assert _safe_page_size(10) == 10
    for s in (_safe_page_size(n) for n in range(1, 60)):
        assert s in (10, 20, 50, 100, 200, 500, 1000)


def test_query_lang_conservative():
    assert _query_lang("¿Cuál es el objeto del Decreto 65/2022 del Consell?") == "es_es"
    assert _query_lang("Què regula la Llei 5/2018 de l'Horta?") == "va_va"
