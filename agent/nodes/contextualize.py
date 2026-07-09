from __future__ import annotations

import logging
import time
from typing import Any

from agent.shared import QAState, return_with_profile
from api.config import get_settings
from api.llm import LlmClient

settings = get_settings()
logger = logging.getLogger("dogv.graph")

# Rewrite the latest user message into a self-contained DOGV query using the prior
# turns. This is what lets follow-ups retrieve correctly: "no, me refería al fondo de
# cooperación municipal" or "¿y los plazos?" become standalone queries before intent
# + retrieval run. The model is told to leave already-standalone or topic-shifting
# messages unchanged (is_followup=false) so a fresh question resets context and we
# never drift a good query (cf. HyDE breaking reference-queries).
CONTEXTUALIZE_SYSTEM = (
    "Eres un asistente del DOGV (Diari Oficial de la Generalitat Valenciana). "
    "A partir de la conversacion previa y el ULTIMO mensaje del usuario, escribe UNA "
    "consulta de busqueda autonoma, BREVE y especifica, centrada en lo que el usuario "
    "quiere AHORA. "
    "Si el ultimo mensaje corrige o concreta la peticion ('no, me referia a X', 'el de Y', "
    "'¿y los plazos?'), centra la consulta en ESA correccion y DESCARTA el encuadre vago "
    "anterior: NO encadenes la pregunta general antigua con los terminos nuevos. "
    "Usa la terminologia oficial mas especifica que aporte el usuario (nombre del programa, "
    "organo, año) y omite muletillas ('hi ha res sobre', 'que hay de', 'volia saber'). "
    "Ejemplo: previa='ajudes per als pobles?' + ultimo='no, el Fons de Cooperacio Municipal' "
    "-> query='assignacio del Fons de Cooperacio Municipal als municipis i entitats locals'. "
    "Si el ultimo mensaje ya es autonomo o cambia de tema, devuelvelo SIN cambios y marca "
    "is_followup=false. No inventes datos, normas ni nombres que no aparezcan en la "
    "conversacion. Responde en el idioma del usuario. "
    "Devuelve SOLO JSON con campos: query (string), is_followup (boolean)."
)

CONTEXTUALIZE_USER = """Conversacion previa:
{history}

Ultimo mensaje del usuario:
{question}

Devuelve SOLO JSON {{"query": ..., "is_followup": ...}}."""

_MAX_TURN_CHARS = 800


def _render_history(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in history:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if len(content) > _MAX_TURN_CHARS:
            content = content[:_MAX_TURN_CHARS] + "…"
        speaker = "Usuario" if role == "user" else "Asistente"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def contextualize_query_node(state: QAState) -> QAState:
    """Front node: rewrite a follow-up into a standalone query using history.

    No-op (zero LLM calls) when history is empty or the feature is disabled. On any
    failure, keeps the verbatim question so the turn always proceeds.
    """
    start = time.monotonic()
    request_id = state.get("request_id")
    question = state["question"]
    history = state.get("history") or []

    # Default passthrough state (single-turn path is byte-for-byte unchanged).
    payload: dict[str, Any] = {"raw_question": question, "contextualized": False}

    if not getattr(settings, "ask_contextualize_enabled", True) or not history:
        return return_with_profile(
            state,
            "contextualize",
            payload,
            elapsed_seconds=round(time.monotonic() - start, 3),
            applied=False,
            reason="no_history" if not history else "disabled",
        )

    rewritten = question
    is_followup = False
    reason = "ok"
    try:
        client = LlmClient()
        result = client.chat_json(
            [
                {"role": "system", "content": CONTEXTUALIZE_SYSTEM},
                {
                    "role": "user",
                    "content": CONTEXTUALIZE_USER.format(
                        history=_render_history(history), question=question
                    ),
                },
            ],
            temperature=0.0,
            enable_thinking=False,
        )
        candidate = str((result or {}).get("query") or "").strip()
        is_followup = bool((result or {}).get("is_followup"))
        # Only adopt the rewrite for genuine follow-ups, and never adopt an empty
        # rewrite. A standalone / new-topic message keeps its verbatim text so we
        # don't drift a query that already retrieves well.
        if is_followup and candidate:
            rewritten = candidate
    except Exception as exc:  # keep the verbatim question; never break the turn
        reason = f"error:{type(exc).__name__}"
        logger.warning("contextualize.fallback req=%s reason=%s", request_id, exc)

    applied = rewritten != question
    if applied:
        payload["question"] = rewritten
    payload["contextualized"] = applied

    elapsed = time.monotonic() - start
    logger.info(
        "contextualize.done req=%s applied=%s is_followup=%s turns=%s elapsed=%.2fs",
        request_id,
        applied,
        is_followup,
        len(history),
        elapsed,
    )
    return return_with_profile(
        state,
        "contextualize",
        payload,
        elapsed_seconds=round(elapsed, 3),
        applied=applied,
        is_followup=is_followup,
        reason=reason,
        raw_question=question,
        rewritten_question=(rewritten if applied else None),
    )
