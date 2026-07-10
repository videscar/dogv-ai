from __future__ import annotations

import re

import api.answer as answer


def test_strip_doc_id_artifacts_removes_all_forms_and_tidies_spacing():
    raw = (
        "Las dos resoluciones son la de convocatoria, RESOLUCIÓN de 12 de marzo de 2026 "
        "(doc_id: 85222), y la de concesión, RESOLUCIÓN de 21 de mayo de 2026 (doc_id: 87141). "
        "Ver también doc_id: 85222, la referencia (doc_ids: 85222, 87141), "
        "el anexo [doc_id 85222], la nota (doc id: 85222) y doc_id:85222."
    )
    cleaned = answer._strip_doc_id_artifacts(raw)

    # No doc_id token in any form survives.
    assert not re.search(r"doc[_ ]?ids?", cleaned, re.IGNORECASE)
    assert "85222" not in cleaned
    assert "87141" not in cleaned

    # Punctuation/spacing around the removed artifacts is clean.
    assert "de 2026, y la de concesión" in cleaned
    assert "  " not in cleaned  # no doubled spaces
    assert " ," not in cleaned and " ." not in cleaned  # no space before punctuation
    assert "()" not in cleaned and "[]" not in cleaned  # no emptied brackets


def test_strip_doc_id_artifacts_preserves_word_boundaries_and_leaves_prose_intact():
    # Bare (non-parenthesized) inline reference: the space between the surrounding
    # words must be preserved, not eaten (regression: "ver ... aqui" -> "veraqui").
    assert answer._strip_doc_id_artifacts("ver doc_ids: 85222, 87141 aqui") == "ver aqui"
    # Parenthetical form: no double space, no space before comma/period.
    assert (
        answer._strip_doc_id_artifacts(
            "RESOLUCIÓN de 12 de marzo de 2026 (doc_id: 85222), "
            "y la de concesión (doc_id: 87141)."
        )
        == "RESOLUCIÓN de 12 de marzo de 2026, y la de concesión."
    )
    # Bracket form mid-sentence.
    assert (
        answer._strip_doc_id_artifacts("marco normativo [doc_id 999] vigente")
        == "marco normativo vigente"
    )
    # Prose with a plain number and the substring "doc" must be left untouched.
    unchanged = "El documento identifica 3 casos importantes."
    assert answer._strip_doc_id_artifacts(unchanged) == unchanged


class _InlineDocIdClient:
    """Synthesis LLM that echoes an internal doc_id label into its prose."""

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0, **kwargs):
        return {
            "answer": (
                "La resolución de concesión es la RESOLUCIÓN de 21 de mayo de 2026 "
                "(doc_id: 87141), publicada en el DOGV."
            ),
            "citations": [87141],
        }


def test_build_answer_strips_inline_doc_id_but_keeps_citations(monkeypatch):
    monkeypatch.setattr(answer, "LlmClient", _InlineDocIdClient)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", False)

    result = answer.build_answer(
        question="¿Cuál es la resolución de concesión?",
        language="es",
        evidence=[{"doc_id": 87141, "quote": "...RESOLUCIÓN de 21 de mayo de 2026, se concede..."}],
    )

    # The doc_id artifact is gone from the answer text ...
    assert "doc_id" not in result["answer"].lower()
    assert "87141" not in result["answer"]
    assert "RESOLUCIÓN de 21 de mayo de 2026, publicada en el DOGV." in result["answer"]
    # ... but the structured citations are untouched.
    assert result["citations"] == [87141]
