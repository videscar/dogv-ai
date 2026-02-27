from __future__ import annotations

import api.answer as answer


class _FlakyClient:
    calls = 0

    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        _FlakyClient.calls += 1
        if _FlakyClient.calls == 1:
            raise ValueError("transient parse error")
        return {
            "answer": "Las bases incluyen requisitos y cuantías con un importe máximo de 250 euros para personas jóvenes.",
            "citations": [12144],
        }


class _AlwaysFailClient:
    def __init__(self, *args, **kwargs):
        return None

    def chat_json(self, messages, temperature=0.0):
        raise ValueError("permanent parse error")


def test_build_answer_retries_once_before_fallback(monkeypatch):
    _FlakyClient.calls = 0
    monkeypatch.setattr(answer, "OllamaClient", _FlakyClient)

    result = answer.build_answer(
        question="Resume las bases y cuantías de las ayudas para jóvenes.",
        language="es",
        evidence=[{"doc_id": 12144, "quote": "Base novena. Cuantía de la ayuda... 250 euros"}],
    )

    assert _FlakyClient.calls == 2
    assert not result["answer"].startswith("Evidencias disponibles:")
    assert result["citations"] == [12144]


def test_build_answer_falls_back_after_retry_exhausted(monkeypatch):
    monkeypatch.setattr(answer, "OllamaClient", _AlwaysFailClient)

    result = answer.build_answer(
        question="¿Qué importe máximo tienen las ayudas para jóvenes?",
        language="es",
        evidence=[{"doc_id": 12144, "quote": "Base novena. Cuantía de la ayuda... 250 euros"}],
    )

    assert result["answer"].startswith("Evidencias disponibles:")
    assert result["citations"] == [12144]
