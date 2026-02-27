from __future__ import annotations

from datetime import date

from api.intent import normalize_intent


def test_normalize_intent_infers_year_range_when_missing():
    intent = normalize_intent(
        {
            "language": "es",
            "doc_kind": "Ayudas",
            "doc_subkind": "Bases",
            "keywords": ["alquiler", "jovenes", "2026"],
            "since_date": None,
            "until_date": None,
            "needs_online": True,
        },
        question="¿Qué importe máximo tienen las ayudas de alquiler para personas jóvenes en 2026?",
    )

    assert intent["since_date"] == date(2026, 1, 1)
    assert intent["until_date"] == date(2026, 12, 31)


def test_normalize_intent_keeps_explicit_dates():
    intent = normalize_intent(
        {
            "language": "es",
            "doc_kind": "Ayudas",
            "doc_subkind": "Bases",
            "since_date": "2025-03-01",
            "until_date": "2025-03-31",
            "needs_online": False,
        },
        question="Ayudas de alquiler en 2026",
    )

    assert intent["since_date"] == date(2025, 3, 1)
    assert intent["until_date"] == date(2025, 3, 31)


def test_normalize_intent_infers_multi_year_bounds():
    intent = normalize_intent(
        {"since_date": None, "until_date": None},
        question="Compara ayudas de transporte de 2024 y 2026.",
    )

    assert intent["since_date"] == date(2024, 1, 1)
    assert intent["until_date"] == date(2026, 12, 31)
