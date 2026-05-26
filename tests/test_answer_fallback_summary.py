from __future__ import annotations

import api.answer as answer


class _AlwaysSemanticFailClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _AlwaysSemanticFailClient.calls += 1
        return {
            "answer": "La cuantia maxima es de 900 euros segun 2025/99999.",
            "citations": [101],
        }


def _set_defaults(monkeypatch) -> None:
    monkeypatch.setattr(answer, "OllamaClient", _AlwaysSemanticFailClient)
    monkeypatch.setattr(answer.settings, "answer_mutators_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_validator_enabled", True)
    monkeypatch.setattr(answer.settings, "answer_missing_notes_enabled", False)
    monkeypatch.setattr(answer.settings, "answer_repair_attempts", 1)
    monkeypatch.setattr(answer.settings, "answer_claim_guard_mode", "unit_aware_strict")
    monkeypatch.setattr(answer.settings, "answer_repair_mode", "conditional")
    monkeypatch.setattr(answer.settings, "answer_fallback_style", "concise_summary")


def test_validation_fallback_uses_concise_summary(monkeypatch):
    _AlwaysSemanticFailClient.calls = 0
    _set_defaults(monkeypatch)
    monkeypatch.setattr(answer.settings, "answer_fallback_max_items", 3)

    result = answer.build_answer(
        question="Cual es la cuantia maxima de la ayuda?",
        language="es",
        evidence=[
            {"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."},
            {"doc_id": 102, "quote": "El plazo es de 10 dias."},
            {"doc_id": 103, "quote": "Las solicitudes se presentan telematicamente."},
        ],
        full_docs=[
            {"document_id": 101, "issue_date": "2025-01-10", "title": "Resolucion ayudas 2025", "ref": "2025/123"},
            {"document_id": 102, "issue_date": "2025-01-11", "title": "Extracto ayudas 2025", "ref": "2025/124"},
            {"document_id": 103, "issue_date": "2025-01-12", "title": "Bases ayudas 2025", "ref": "2025/125"},
        ],
    )

    assert _AlwaysSemanticFailClient.calls == 2
    assert result["diagnostics"]["fallback_reason"] == "validation_failed"
    assert "Publicaciones relevantes encontradas:" in result["answer"]
    assert not result["answer"].startswith("Evidencias disponibles:")
    assert result["citations"] == [101]


def test_validation_fallback_respects_max_items(monkeypatch):
    _AlwaysSemanticFailClient.calls = 0
    _set_defaults(monkeypatch)
    monkeypatch.setattr(answer.settings, "answer_fallback_max_items", 2)

    result = answer.build_answer(
        question="Cual es la cuantia maxima de la ayuda?",
        language="es",
        evidence=[
            {"doc_id": 101, "quote": "La cuantia maxima es de 250 euros."},
            {"doc_id": 102, "quote": "El plazo es de 10 dias."},
            {"doc_id": 103, "quote": "Las solicitudes se presentan telematicamente."},
            {"doc_id": 104, "quote": "Texto adicional."},
        ],
        full_docs=[
            {"document_id": 101, "issue_date": "2025-01-10", "title": "Resolucion ayudas 2025", "ref": "2025/123"},
            {"document_id": 102, "issue_date": "2025-01-11", "title": "Extracto ayudas 2025", "ref": "2025/124"},
            {"document_id": 103, "issue_date": "2025-01-12", "title": "Bases ayudas 2025", "ref": "2025/125"},
            {"document_id": 104, "issue_date": "2025-01-13", "title": "Anuncio ayudas 2025", "ref": "2025/126"},
        ],
    )

    bullet_lines = [line for line in result["answer"].splitlines() if line.startswith("- (")]
    assert len(bullet_lines) == 2
