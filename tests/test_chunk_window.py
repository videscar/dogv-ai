from __future__ import annotations

from agent.nodes.read_chunks import _fold_preserving_length, _salient_keywords, _window_chunk_text


def test_short_chunk_untouched():
    assert _window_chunk_text("hola mundo", 1200, ["mundo"]) == "hola mundo"


def test_no_keywords_falls_back_to_prefix():
    text = "a" * 2000
    assert _window_chunk_text(text, 1200, []) == text[:1200]


def test_no_late_hits_keeps_exact_legacy_prefix():
    # keyword only appears inside the prefix half -> behaviour identical to [:cap]
    text = "cuantia individualizada " + "x" * 2000
    out = _window_chunk_text(text, 1200, ["cuantia", "individualizada"])
    assert out == text[:1200]


def test_deep_answer_text_is_recovered():
    # Answer-bearing text past the cap (the Q7-ES / Q12 shape: figure at
    # offset > cap in a chunk selected FOR that content).
    filler = "bases reguladoras de la convocatoria. " * 45  # ~1700 chars
    answer = "Duodecimo. Cuantia individualizada: cada centro percibira 500,00 euros."
    text = filler + answer + " mas texto posterior." * 10
    out = _window_chunk_text(text, 1200, ["cuantia", "individualizada"])
    assert len(out) <= 1200
    assert "500,00 euros" in out
    # prefix context is retained too
    assert out.startswith("bases reguladoras")
    assert " … " in out


def test_window_never_loses_legacy_prefix_answer_to_generic_keywords():
    # Q7-VA shape: the answer ("quantia individualitzada... 500,00") sits in the
    # legacy-visible zone [half:cap]; a deeper region repeats generic question
    # words (projecte, guardabosc) more densely. Marginal scoring must keep the
    # answer window, not the generic one.
    prefix = ("el projecte es desplegara segons les bases. " * 14)[:600]  # 'projecte' in prefix
    answer = "Dotze. Quantia individualitzada: cada centre percebra 500,00 euros. "
    generic = "el projecte guardabosc del projecte guardabosc als centres. " * 10
    text = prefix + answer + generic
    out = _window_chunk_text(text, 1200, ["quantia", "individualitzada", "projecte", "guardabosc"])
    assert "500,00 euros" in out


def test_window_falls_back_to_legacy_prefix_when_nothing_new():
    # every late window only repeats kinds already visible in the prefix half
    text = ("guardabosc " * 60) + ("guardabosc bla " * 150)
    out = _window_chunk_text(text, 1200, ["guardabosc"])
    assert out == text[:1200]


def test_accented_text_matches_folded_keywords_at_right_offset():
    filler = "y" * 800
    text = filler + " Retribucions íntegres anuals: 37.804,62 euros." + " z" * 400
    out = _window_chunk_text(text, 1000, ["retribucio", "anual"])
    assert "37.804,62" in out


def test_fold_preserves_length():
    s = "Cuantía individualizada — açò és için ﬁ"
    assert len(_fold_preserving_length(s)) == len(s)


def test_salient_keywords_are_folded():
    kws = _salient_keywords("Quina és la retribució anual del professorat?")
    assert "retribucio" in kws
    assert all(k == k.lower() for k in kws)
