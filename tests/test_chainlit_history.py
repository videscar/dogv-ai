from __future__ import annotations

import asyncio
import importlib
import sys
import types


class _FakeMessage:
    def __init__(self, content: str = ""):
        self.content = content

    async def send(self):
        return self

    async def update(self):
        return self

    async def remove(self):
        return self


class _FakeSession(dict):
    def set(self, key, value):
        self[key] = value


def _load_chainlit_app(monkeypatch):
    fake_chainlit = types.ModuleType("chainlit")
    fake_chainlit.on_chat_start = lambda fn: fn
    fake_chainlit.on_message = lambda fn: fn
    fake_chainlit.Message = _FakeMessage
    fake_chainlit.user_session = _FakeSession()

    monkeypatch.setitem(sys.modules, "chainlit", fake_chainlit)
    sys.modules.pop("ui.chainlit_app", None)
    return importlib.import_module("ui.chainlit_app")


def test_append_history_accumulates_user_then_assistant(monkeypatch):
    app = _load_chainlit_app(monkeypatch)
    app.cl.user_session.set("history", [])
    app._append_history("pregunta uno", "respuesta uno")
    assert app._get_history() == [
        {"role": "user", "content": "pregunta uno"},
        {"role": "assistant", "content": "respuesta uno"},
    ]


def test_history_truncates_to_max_messages(monkeypatch):
    monkeypatch.setenv("DEMO_HISTORY_MAX_TURNS", "2")  # cap = 4 messages
    app = _load_chainlit_app(monkeypatch)
    app.cl.user_session.set("history", [])
    for i in range(5):
        app._append_history(f"q{i}", f"a{i}")
    history = app._get_history()
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "q3"}


def test_on_message_passes_history_and_stores_plain_answer(monkeypatch):
    app = _load_chainlit_app(monkeypatch)
    monkeypatch.setattr(app, "STREAM_ENABLED", False)
    calls: dict = {}

    class _FakeClient:
        async def ask(self, question, history=None):
            calls["question"] = question
            calls["history"] = history
            return {
                "answer": "Respuesta sin citas",
                "citations": [
                    {
                        "document_id": 11,
                        "title": "Doc",
                        "ref": "2026/11",
                        "issue_date": None,
                        "pdf_url": None,
                        "html_url": None,
                    }
                ],
            }

    app.cl.user_session.set("dogv_client", _FakeClient())
    prior = [
        {"role": "user", "content": "previa"},
        {"role": "assistant", "content": "resp previa"},
    ]
    app.cl.user_session.set("history", list(prior))

    asyncio.run(app.on_message(_FakeMessage("nueva pregunta")))

    # The prior transcript is sent to the backend on this turn…
    assert calls["history"] == prior
    # …and the new exchange is appended, storing the plain answer (no citation block).
    history = app._get_history()
    assert history[-2] == {"role": "user", "content": "nueva pregunta"}
    assert history[-1] == {"role": "assistant", "content": "Respuesta sin citas"}
    assert "### Citas" not in history[-1]["content"]


def test_on_chat_start_resets_history(monkeypatch):
    app = _load_chainlit_app(monkeypatch)
    app.cl.user_session.set("history", [{"role": "user", "content": "vieja"}])

    class _ReadyClient:
        async def get_ready(self):
            return {"ready": True}

    monkeypatch.setattr(app, "_client", lambda: _ReadyClient())
    asyncio.run(app.on_chat_start())
    assert app._get_history() == []
