from __future__ import annotations

from agent.graph import _prepend_recent_relevant_docs


def test_recent_relevant_docs_are_prepended_for_recent_queries():
    question = "Resumeix la convocatòria més recent relacionada amb habitatge."
    keywords = ["habitatge"]
    rerank_candidates = [
        {
            "document_id": 10,
            "issue_date": "2025-10-30",
            "title": "Resolució d'ajudes habitatge 2025",
            "snippet": "Programa d'habitatge per al lloguer.",
        },
        {
            "document_id": 20,
            "issue_date": "2026-02-25",
            "title": "Convocatòria recent d'habitatge",
            "snippet": "Ajudes d'habitatge i requisits.",
        },
        {
            "document_id": 30,
            "issue_date": "2026-02-26",
            "title": "Convocatòria de mobilitat",
            "snippet": "No relacionada amb habitatge.",
        },
    ]
    selected_doc_ids = [10, 11]

    merged = _prepend_recent_relevant_docs(
        question=question,
        keywords=keywords,
        rerank_candidates=rerank_candidates,
        doc_ids=selected_doc_ids,
    )

    assert merged[0] == 20
    assert 10 in merged
    assert 11 in merged
