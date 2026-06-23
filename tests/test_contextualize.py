from __future__ import annotations

import agent.nodes.contextualize as ctx


class _FakeClient:
    """Records chat_json calls and returns a canned result (or raises)."""

    calls = 0
    result: dict | None = None
    exc: Exception | None = None

    def __init__(self, *a, **k):
        pass

    def chat_json(self, messages, temperature=0.0, enable_thinking=False):
        type(self).calls += 1
        if type(self).exc is not None:
            raise type(self).exc
        return type(self).result or {}


def _install(monkeypatch, *, result=None, exc=None, enabled=True):
    _FakeClient.calls = 0
    _FakeClient.result = result
    _FakeClient.exc = exc
    monkeypatch.setattr(ctx, "LlmClient", _FakeClient)
    monkeypatch.setattr(ctx.settings, "ask_contextualize_enabled", enabled)


def test_empty_history_is_noop_no_llm_call(monkeypatch):
    _install(monkeypatch, result={"query": "X", "is_followup": True})
    out = ctx.contextualize_query_node({"question": "hola", "history": []})
    assert "question" not in out  # verbatim question preserved (not overwritten)
    assert out["raw_question"] == "hola"
    assert out["contextualized"] is False
    assert _FakeClient.calls == 0


def test_disabled_is_noop_no_llm_call(monkeypatch):
    _install(monkeypatch, result={"query": "X", "is_followup": True}, enabled=False)
    out = ctx.contextualize_query_node(
        {"question": "hola", "history": [{"role": "user", "content": "antes"}]}
    )
    assert "question" not in out
    assert out["contextualized"] is False
    assert _FakeClient.calls == 0


def test_followup_correction_is_adopted(monkeypatch):
    _install(
        monkeypatch,
        result={
            "query": "asignación del Fondo de Cooperación Municipal 2026",
            "is_followup": True,
        },
    )
    state = {
        "question": "no, me refería al fondo de cooperación municipal",
        "history": [
            {"role": "user", "content": "¿ayudas para los pueblos?"},
            {"role": "assistant", "content": "Hay ayudas a municipios en riesgo…"},
        ],
    }
    out = ctx.contextualize_query_node(state)
    assert out["question"] == "asignación del Fondo de Cooperación Municipal 2026"
    assert out["raw_question"] == state["question"]
    assert out["contextualized"] is True
    assert _FakeClient.calls == 1


def test_anaphora_followup_is_adopted(monkeypatch):
    _install(
        monkeypatch,
        result={"query": "plazos de las becas universitarias 2026", "is_followup": True},
    )
    out = ctx.contextualize_query_node(
        {
            "question": "¿y los plazos?",
            "history": [{"role": "user", "content": "becas universitarias"}],
        }
    )
    assert out["question"] == "plazos de las becas universitarias 2026"
    assert out["contextualized"] is True


def test_new_topic_not_adopted(monkeypatch):
    # Model judges the message a fresh, standalone question -> keep it verbatim even
    # if it echoes a (possibly drifted) query back.
    _install(
        monkeypatch,
        result={"query": "algo distinto reescrito", "is_followup": False},
    )
    out = ctx.contextualize_query_node(
        {
            "question": "¿qué hay sobre certificados de valenciano?",
            "history": [{"role": "user", "content": "becas universitarias"}],
        }
    )
    assert "question" not in out
    assert out["contextualized"] is False


def test_empty_rewrite_not_adopted(monkeypatch):
    _install(monkeypatch, result={"query": "  ", "is_followup": True})
    out = ctx.contextualize_query_node(
        {"question": "¿y eso?", "history": [{"role": "user", "content": "x"}]}
    )
    assert "question" not in out
    assert out["contextualized"] is False


def test_llm_error_falls_back_to_raw_question(monkeypatch):
    _install(monkeypatch, exc=RuntimeError("boom"))
    out = ctx.contextualize_query_node(
        {"question": "¿y los requisitos?", "history": [{"role": "user", "content": "x"}]}
    )
    assert "question" not in out
    assert out["raw_question"] == "¿y los requisitos?"
    assert out["contextualized"] is False
