from __future__ import annotations

import api.answer as answer


def _set_defaults(monkeypatch) -> None:
    monkeypatch.setattr(answer.settings, "answer_mutators_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", True)
    monkeypatch.setattr(answer.settings, "answer_missing_notes_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_repair_attempts", 1)
    monkeypatch.setattr(answer.settings, "answer_claim_guard_mode", "unit_aware_strict")
    monkeypatch.setattr(answer.settings, "answer_repair_mode", "conditional")
    monkeypatch.setattr(answer.settings, "answer_fallback_style", "concise_summary")
    monkeypatch.setattr(answer.settings, "answer_fallback_max_items", 3)


def test_validate_answer_ignores_year_only_tokens(monkeypatch):
    monkeypatch.setattr(answer.settings, "answer_claim_guard_mode", "unit_aware_strict")
    errors = answer._validate_answer(
        answer_text="La convocatoria prevista para 2030 se regula aqui.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "Convocatoria 2025 para ayudas."}],
        full_docs=None,
    )
    assert "unsupported_numeric_or_ref_claim" not in errors


def test_validate_answer_flags_unsupported_currency_claim(monkeypatch):
    monkeypatch.setattr(answer.settings, "answer_claim_guard_mode", "unit_aware_strict")
    errors = answer._validate_answer(
        answer_text="La cuantia es de 900 euros.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
        full_docs=None,
    )
    assert "unsupported_numeric_or_ref_claim" in errors


def test_validate_answer_flags_reference_out_of_source(monkeypatch):
    monkeypatch.setattr(answer.settings, "answer_claim_guard_mode", "unit_aware_strict")
    errors = answer._validate_answer(
        answer_text="Segun la referencia 2025/99999, aplica la ayuda.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "Segun la referencia 2025/12345, aplica la ayuda."}],
        full_docs=None,
    )
    assert "unsupported_numeric_or_ref_claim" in errors


def test_validate_answer_flags_citation_out_of_scope():
    errors = answer._validate_answer(
        answer_text="La cuantia maxima es de 250 euros.",
        citations=[999],
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
        full_docs=None,
    )
    assert "citation_out_of_scope" in errors


def test_validate_answer_flags_no_consta_when_evidence_exists():
    errors = answer._validate_answer(
        answer_text="No consta.",
        citations=[101],
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
        full_docs=None,
    )
    assert "no_consta_only_with_evidence" in errors


class _StructuralOnlyClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _StructuralOnlyClient.calls += 1
        return {
            "answer": "La cuantia maxima es de 250 euros.",
            "citations": [],
        }


class _OutOfScopeCitationClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _OutOfScopeCitationClient.calls += 1
        return {
            "answer": "La cuantia maxima es de 250 euros.",
            "citations": [999, 101],
        }


class _RepairSuccessClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _RepairSuccessClient.calls += 1
        if _RepairSuccessClient.calls == 1:
            return {"answer": "La cuantia maxima es de 900 euros.", "citations": [101]}
        return {"answer": "La cuantia maxima es de 250 euros.", "citations": [101]}


def test_build_answer_fixes_missing_citations_without_repair(monkeypatch):
    _set_defaults(monkeypatch)
    _StructuralOnlyClient.calls = 0
    monkeypatch.setattr(answer, "LlmClient", _StructuralOnlyClient)

    result = answer.build_answer(
        question="Cual es la cuantia maxima de la ayuda?",
        language="es",
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
    )

    assert _StructuralOnlyClient.calls == 1
    assert result["citations"] == [101]
    assert result["diagnostics"]["repair_attempts"] == 0
    assert result["diagnostics"]["repair_skipped_reason"] == "structural_only"
    assert result["diagnostics"]["deterministic_fix_applied"] is True
    assert "citation_missing_if_evidence" in result["diagnostics"]["deterministic_fix_types"]


def test_build_answer_clamps_out_of_scope_citations_without_repair(monkeypatch):
    _set_defaults(monkeypatch)
    _OutOfScopeCitationClient.calls = 0
    monkeypatch.setattr(answer, "LlmClient", _OutOfScopeCitationClient)

    result = answer.build_answer(
        question="Cual es la cuantia maxima de la ayuda?",
        language="es",
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
    )

    assert _OutOfScopeCitationClient.calls == 1
    assert result["citations"] == [101]
    assert result["diagnostics"]["repair_attempts"] == 0
    assert result["diagnostics"]["repair_skipped_reason"] == "structural_only"
    assert result["diagnostics"]["deterministic_fix_applied"] is True
    assert "citation_out_of_scope" in result["diagnostics"]["deterministic_fix_types"]


def test_build_answer_uses_conditional_repair_for_semantic_errors(monkeypatch):
    _set_defaults(monkeypatch)
    _RepairSuccessClient.calls = 0
    monkeypatch.setattr(answer, "LlmClient", _RepairSuccessClient)

    result = answer.build_answer(
        question="Cual es la cuantia maxima de la ayuda?",
        language="es",
        evidence=[{"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."}],
    )

    assert _RepairSuccessClient.calls == 2
    assert result["answer"] == "La cuantia maxima es de 250 euros."
    assert result["citations"] == [101]
    assert result["diagnostics"]["repair_attempts"] == 1
    assert result["diagnostics"]["repair_success"] is True
