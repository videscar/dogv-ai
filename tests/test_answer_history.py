from __future__ import annotations

import api.answer as ans


def test_format_history_empty_returns_blank():
    assert ans._format_history(None) == ""
    assert ans._format_history([]) == ""
    assert ans._format_history([{"role": "user", "content": "   "}]) == ""


def test_format_history_has_grounding_instruction_and_turns():
    block = ans._format_history(
        [
            {"role": "user", "content": "¿ayudas para los pueblos?"},
            {"role": "assistant", "content": "Hay ayudas a municipios…"},
        ]
    )
    assert "interpretar la pregunta actual" in block
    assert "evidencia de este turno" in block
    assert "Usuario: ¿ayudas para los pueblos?" in block
    assert "Asistente: Hay ayudas a municipios" in block


def _capture(monkeypatch):
    """Stub the LLM + validator/mutators so build_answer only constructs messages."""
    captured: dict = {}

    class _DummyClient:
        model = "qwen3.6-27b"

        def __init__(self, *a, **k):
            pass

    def _fake_retry(client, messages, **kwargs):
        captured["messages"] = messages
        return {"answer": "respuesta", "citations": [11]}

    monkeypatch.setattr(ans, "LlmClient", _DummyClient)
    monkeypatch.setattr(ans, "chat_json_with_retry", _fake_retry)
    monkeypatch.setattr(ans.settings, "answer_validator_enabled", False)
    return captured


def test_build_answer_includes_history_block_when_present(monkeypatch):
    captured = _capture(monkeypatch)
    ans.build_answer(
        "asignación del Fondo de Cooperación Municipal",
        "es_es",
        evidence=[{"doc_id": 11, "quote": "El Fondo de Cooperación Municipal asigna…"}],
        history=[{"role": "user", "content": "¿ayudas para los pueblos?"}],
    )
    user_msg = captured["messages"][1]["content"]
    assert "Conversacion previa" in user_msg
    assert "¿ayudas para los pueblos?" in user_msg


def test_build_answer_no_history_block_when_absent(monkeypatch):
    captured = _capture(monkeypatch)
    ans.build_answer(
        "¿qué ayudas hay?",
        "es_es",
        evidence=[{"doc_id": 11, "quote": "Ayudas…"}],
    )
    user_msg = captured["messages"][1]["content"]
    assert "Conversacion previa" not in user_msg


def test_history_does_not_leak_into_citations(monkeypatch):
    # Citations come from the model result, not from history; ensure a history that
    # mentions a different doc id never becomes a citation.
    _capture(monkeypatch)
    result = ans.build_answer(
        "y los plazos?",
        "es_es",
        evidence=[{"doc_id": 11, "quote": "Plazo hasta el 3 de julio."}],
        history=[{"role": "assistant", "content": "doc_id 999 trataba de otra cosa"}],
    )
    assert 999 not in (result.get("citations") or [])
