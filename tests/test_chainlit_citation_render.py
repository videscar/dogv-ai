from __future__ import annotations

import importlib
import sys
import types


class _FakeMessage:
    def __init__(self, content: str = ""):
        self.content = content

    async def send(self):
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


def test_render_citations_prefers_html_url(monkeypatch):
    app = _load_chainlit_app(monkeypatch)

    block = app._render_citations(
        [
            {
                "document_id": 100,
                "title": "Documento DOGV",
                "ref": "2026/100",
                "issue_date": "2026-02-23",
                "html_url": "https://dogv.gva.es/html/100",
                "pdf_url": "https://dogv.gva.es/pdf/100",
            }
        ]
    )

    assert "### Citas" in block
    assert "[Documento DOGV](https://dogv.gva.es/html/100)" in block
    assert "https://dogv.gva.es/pdf/100" not in block


def test_render_citations_falls_back_to_pdf_url(monkeypatch):
    app = _load_chainlit_app(monkeypatch)

    block = app._render_citations(
        [
            {
                "document_id": 101,
                "title": "Documento PDF",
                "ref": "2026/101",
                "issue_date": "2026-02-23",
                "html_url": None,
                "pdf_url": "https://dogv.gva.es/pdf/101",
            }
        ]
    )

    assert "[Documento PDF](https://dogv.gva.es/pdf/101)" in block


def test_render_citations_empty_list_returns_empty_string(monkeypatch):
    app = _load_chainlit_app(monkeypatch)
    assert app._render_citations([]) == ""
