from __future__ import annotations

from scripts.run_answer_eval import _contains_term


def test_contains_term_is_accent_insensitive():
    assert _contains_term("La ayuda aplica a personas jovenes.", "jóven")


def test_contains_term_is_case_and_whitespace_insensitive():
    assert _contains_term("Programa de transporte    escolar rural.", "Transporte escolar")
