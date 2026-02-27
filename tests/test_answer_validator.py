from __future__ import annotations

import api.answer as answer


def test_validate_answer_flags_unsupported_numeric_claim():
    errors = answer._validate_answer(
        answer_text="La cuantía es de 900 euros.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "La cuantía máxima es de 250 euros."}],
        full_docs=None,
    )

    assert "unsupported_numeric_or_ref_claim" in errors


def test_validate_answer_flags_citation_out_of_scope():
    errors = answer._validate_answer(
        answer_text="La cuantía máxima es de 250 euros.",
        citations=[999],
        evidence=[{"doc_id": 101, "quote": "La cuantía máxima es de 250 euros."}],
        full_docs=None,
    )

    assert "citation_out_of_scope" in errors


def test_validate_answer_flags_no_consta_when_evidence_exists():
    errors = answer._validate_answer(
        answer_text="No consta.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "La cuantía máxima es de 250 euros."}],
        full_docs=None,
    )

    assert "no_consta_only_with_evidence" in errors


class _RepairSuccessClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _RepairSuccessClient.calls += 1
        if _RepairSuccessClient.calls == 1:
            return {
                "answer": "La cuantía máxima es de 900 euros.",
                "citations": [999],
            }
        return {
            "answer": "La cuantía máxima es de 250 euros.",
            "citations": [101],
        }


class _RepairFailClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _RepairFailClient.calls += 1
        return {
            "answer": "La cuantía máxima es de 900 euros.",
            "citations": [999],
        }


def test_build_answer_repairs_once_and_returns_repaired_answer(monkeypatch):
    _RepairSuccessClient.calls = 0
    monkeypatch.setattr(answer, "OllamaClient", _RepairSuccessClient)
    monkeypatch.setattr(answer.settings, "answer_mutators_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", True)
    monkeypatch.setattr(answer.settings, "answer_missing_notes_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_repair_attempts", 1)

    result = answer.build_answer(
        question="¿Cuál es la cuantía máxima de la ayuda?",
        language="es",
        evidence=[{"doc_id": 101, "quote": "La cuantía máxima es de 250 euros."}],
    )

    assert _RepairSuccessClient.calls == 2
    assert result["answer"] == "La cuantía máxima es de 250 euros."
    assert result["citations"] == [101]


def test_build_answer_falls_back_after_failed_repair(monkeypatch):
    _RepairFailClient.calls = 0
    monkeypatch.setattr(answer, "OllamaClient", _RepairFailClient)
    monkeypatch.setattr(answer.settings, "answer_mutators_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", True)
    monkeypatch.setattr(answer.settings, "answer_missing_notes_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_repair_attempts", 1)

    result = answer.build_answer(
        question="¿Cuál es la cuantía máxima de la ayuda?",
        language="es",
        evidence=[{"doc_id": 101, "quote": "La cuantía máxima es de 250 euros."}],
    )

    assert _RepairFailClient.calls == 2
    assert result["answer"].startswith("Evidencias disponibles:")
    assert result["citations"] == [101]
