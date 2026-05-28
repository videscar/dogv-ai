from __future__ import annotations

import api.answer as answer


class _IdentityClient:
    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0, **kwargs):
        return {
            "answer": "El president actual és Juan Francisco Pérez Llorca i l'ex-president és Carlos Mazón Guixot.",
            "citations": [42629, 42628],
        }


def test_identity_answer_has_no_beneficiary_injection(monkeypatch):
    monkeypatch.setattr(answer, "LlmClient", _IdentityClient)
    monkeypatch.setattr(answer.settings, "answer_mutators_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", True)
    monkeypatch.setattr(answer.settings, "answer_missing_notes_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_repair_attempts", 1)

    result = answer.build_answer(
        question="Qui son el president i ex-president de la generalitat valenciana?",
        language="va",
        evidence=[
            {"doc_id": 42628, "quote": "...cessament de Carlos Mazón Guixot..."},
            {"doc_id": 42629, "quote": "...nomena President Juan Francisco Pérez Llorca..."},
        ],
    )

    assert "beneficiari" not in result["answer"].lower()
    assert result["citations"] == [42629, 42628]
