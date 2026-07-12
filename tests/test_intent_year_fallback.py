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


def test_normalize_intent_infers_month_window_va():
    # "a l'agost de 2024" (Valencian) — the intent LLM regularly drops these;
    # the deterministic fallback must produce the month window (probe P13).
    intent = normalize_intent(
        {"language": "ca", "keywords": ["programa CODI"], "needs_online": False},
        question=(
            "Quin import màxim finançava el programa CODI de capacitació digital "
            "per a l'alumnat convocat a l'agost de 2024?"
        ),
    )
    assert intent["since_date"] == date(2024, 8, 1)
    assert intent["until_date"] == date(2024, 8, 31)


def test_normalize_intent_infers_month_window_es():
    intent = normalize_intent(
        {"language": "es", "keywords": [], "needs_online": False},
        question="¿Qué acordó la Dirección General de Patrimonio en julio de 2025?",
    )
    assert intent["since_date"] == date(2025, 7, 1)
    assert intent["until_date"] == date(2025, 7, 31)


def test_normalize_intent_month_window_not_triggered_by_norm_refs():
    # "Ley 39/2015" has a year but no month phrase -> falls through to the year
    # inference, unchanged behavior.
    intent = normalize_intent(
        {"language": "es", "keywords": [], "needs_online": False},
        question="¿Qué establece la Ley 39/2015 sobre el silencio administrativo?",
    )
    assert intent["since_date"] == date(2015, 1, 1)
    assert intent["until_date"] == date(2015, 12, 31)


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
