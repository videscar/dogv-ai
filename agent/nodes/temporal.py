from __future__ import annotations

import logging
import time

from agent.shared import QAState, return_with_profile
from api.config import get_settings
from api.query_expansion import guess_language, is_relative_time_query

settings = get_settings()
logger = logging.getLogger("dogv.graph")


def _temporal_reject_message(language: str | None) -> str:
    if (language or "").startswith(("va", "ca")):
        return (
            "La consulta fa referencia a una data relativa (hui, esta setmana, etc.). "
            "Indica un rang de dates concret (YYYY-MM-DD a YYYY-MM-DD) per a poder buscar al DOGV."
        )
    return (
        "La consulta hace referencia a una fecha relativa (hoy, esta semana, etc.). "
        "Indica un rango de fechas concreto (YYYY-MM-DD a YYYY-MM-DD) para poder buscar en el DOGV."
    )


def temporal_guard_node(state: QAState) -> QAState:
    start = time.monotonic()
    policy = (settings.ask_temporal_policy or "reject").lower()
    question = state.get("question") or ""
    if policy == "reject" and is_relative_time_query(question):
        language = state.get("language") or guess_language(question)
        return return_with_profile(
            state,
            "temporal_guard",
            {
                "answer": _temporal_reject_message(language),
                "citations": [],
                "temporal_reject": True,
            },
            elapsed_seconds=round(time.monotonic() - start, 3),
            policy=policy,
            rejected=True,
        )
    return return_with_profile(
        state,
        "temporal_guard",
        {"temporal_reject": False},
        elapsed_seconds=round(time.monotonic() - start, 3),
        policy=policy,
        rejected=False,
    )
