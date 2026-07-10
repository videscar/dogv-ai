from __future__ import annotations

import logging
import re
from typing import Any

from .answer_fallback import (
    fallback_from_evidence,
    no_evidence_fallback,
    validation_fallback_answer,
)
from .answer_validator import (
    chat_json_with_retry,
    collect_citation_ids,
    is_no_consta_only_answer,
    normalize_citations,
    validate_and_repair,
)
from .config import get_settings
from .llm import LlmClient

ANSWER_SYSTEM = (
    "Eres un asistente del DOGV. Responde solo con evidencia. "
    "Usa solo las citas textuales (quote) para responder. "
    "Adapta el formato a la pregunta; si hay varias subpreguntas, cubrelas una por una. "
    "No inventes beneficiarios, cuantias o plazos si no aparecen en las evidencias "
    "o en los documentos completos. Si falta un dato concreto, indica 'No consta' solo para ese dato, "
    "no como respuesta completa cuando haya evidencia relacionada. "
    "Si la pregunta cita una norma por numero o fecha (decreto, orden, resolucion, acuerdo) y la "
    "evidencia contiene esa MISMA norma (mismo objeto y contenido) pero con la referencia mal citada, "
    "senala brevemente la referencia correcta segun la evidencia y responde con ella. Hazlo solo cuando "
    "sea claramente la misma norma con la referencia equivocada; si en la evidencia solo hay normas "
    "DISTINTAS sobre un tema relacionado (no la que se pregunta), no las presentes como 'la referencia "
    "correcta' ni como si fueran la norma preguntada: indica que no consta esa norma concreta. Aun asi, "
    "puedes usar esa evidencia relacionada para responder un dato concreto y general (por ejemplo un "
    "plazo o un requisito comun) si aclaras que procede de otra norma o convocatoria. Si la referencia "
    "de la pregunta ya aparece en la evidencia, no la cuestiones. "
    "Devuelve SOLO JSON con campos: answer (texto), citations (lista de doc_id)."
)

ANSWER_USER = """Idioma de respuesta: {language}
{conversation}
Pregunta:
{question}

Evidencias:
{evidence}

Documentos completos (si existen):
{full_docs}

Si no hay evidencia suficiente y no hay documentos completos, di que no hay publicaciones encontradas y pide mas detalles. Si hay evidencia que es claramente la misma norma que pregunta pero con la referencia (numero o fecha) mal citada, corrige la referencia y responde con ella; pero no sustituyas la norma preguntada por otra distinta solo porque trate un tema parecido.
Si hay documentos completos, puedes usarlos para aportar detalle, pero cita los doc_id relevantes.
Responde con el nivel de detalle que permita la evidencia y cubre cada parte de la pregunta.
"""

settings = get_settings()
logger = logging.getLogger("dogv.answer")
ANSWER_TIMEOUT = settings.llm_timeout


# Evidence/full-doc blocks fed to the synthesis LLM label each source with a
# `doc_id: {N}` line so the model can cite by id. The model sometimes echoes that
# label back into its prose, e.g. "RESOLUCIÓN de 12 de marzo de 2026 (doc_id: 85222)".
# These are an internal artifact and must never surface in the returned answer text.
# Citations are a SEPARATE structured field (built from result["citations"], not parsed
# from prose), so stripping the label from the text cannot affect them.
_DOC_ID_ARTIFACT_RE = re.compile(
    r"[\(\[]?\s*doc[_ ]?ids?\s*[:#]?\s*\d+(?:\s*,\s*\d+)*\s*[\)\]]?",
    re.IGNORECASE,
)


def _strip_doc_id_artifacts(text: str) -> str:
    """Remove `doc_id`/`doc_ids` reference artifacts from answer prose and tidy fallout.

    Handles (case-insensitively) forms with or without surrounding parentheses/brackets,
    singular/plural, an optional `:`/`#` separator, and comma-separated id lists, e.g.
    "(doc_id: 85222)", "doc_id:85222", "(doc_ids: 85222, 87141)", "[doc_id 85222]".
    After removal it collapses doubled spaces, drops a space left before `,`/`.`/`;`/`:`,
    removes emptied `()`/`[]`, and strips trailing whitespace. No other content is altered.
    """
    if not text:
        return text
    cleaned = _DOC_ID_ARTIFACT_RE.sub("", text)
    # Drop parens/brackets emptied out by the removal above.
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    # Collapse doubled spaces/tabs left where the artifact sat between words.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Remove a space left dangling before punctuation ("2026 , y" -> "2026, y").
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    # Strip trailing spaces at line ends / end of string.
    cleaned = re.sub(r"[ \t]+(\n|$)", r"\1", cleaned)
    return cleaned.strip()


_HISTORY_MAX_TURN_CHARS = 600


def _format_history(history: list[dict[str, Any]] | None) -> str:
    """Render prior turns as an interpret-only context block, or '' when absent.

    The block is explicitly scoped to disambiguating the current question — the
    answer must still come from this turn's evidence (guards against the model
    answering from the earlier assistant text instead of the retrieved docs).
    """
    if not history:
        return ""
    lines: list[str] = []
    for turn in history:
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if len(content) > _HISTORY_MAX_TURN_CHARS:
            content = content[:_HISTORY_MAX_TURN_CHARS] + "…"
        speaker = "Usuario" if str(turn.get("role")) == "user" else "Asistente"
        lines.append(f"{speaker}: {content}")
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "\nConversacion previa (usala SOLO para interpretar la pregunta actual; "
        "responde con la evidencia de este turno y cita sus doc_id, no inventes a "
        "partir del historial):\n" + body + "\n"
    )


def _format_full_docs(full_docs: list[dict[str, Any]] | None) -> str:
    if not full_docs:
        return ""
    blocks = []
    for doc in full_docs:
        blocks.append(
            "doc_id: {doc_id}\nfecha: {date}\ntitulo: {title}\nref: {ref}\ntexto:\n{text}".format(
                doc_id=doc.get("document_id"),
                date=doc.get("issue_date"),
                title=doc.get("title"),
                ref=doc.get("ref"),
                text=doc.get("text"),
            )
        )
    return "\n\n---\n\n".join(blocks)


def _evidence_id_line(doc_id: Any, meta: dict[int, dict[str, Any]] | None) -> str:
    """`doc_id:` line plus the document's title/ref/date when known.

    Quotes are extracts from a norm's preamble/articles and rarely restate the
    disposition's own name; without the title the synthesis LLM can't tell which
    quote belongs to a norm asked for by number/date, and abstains ("No consta")
    even when the right doc's content is present. The title comes from the DB via
    candidate_docs, so it's grounded metadata, not model-generated."""
    head = f"doc_id: {doc_id}"
    try:
        info = (meta or {}).get(int(doc_id))
    except (TypeError, ValueError):
        info = None
    if not info:
        return head
    title = str(info.get("title") or "").strip()
    if title:
        head += f"\ntitulo: {title}"
    ref = str(info.get("ref") or "").strip()
    date = str(info.get("issue_date") or info.get("date") or "").strip()
    if ref or date:
        head += "\n" + " | ".join(
            p for p in (f"ref: {ref}" if ref else "", f"fecha: {date}" if date else "") if p
        )
    return head


def _format_evidence(
    evidence: list[dict[str, Any]] | None,
    doc_meta: dict[int, dict[str, Any]] | None = None,
) -> str:
    if not evidence:
        return ""
    blocks = []
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        blocks.append(f"{_evidence_id_line(doc_id, doc_meta)}\nquote: {quote}")
    return "\n\n---\n\n".join(blocks)


def build_answer(
    question: str,
    language: str,
    evidence: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    doc_meta: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = LlmClient(timeout=ANSWER_TIMEOUT)
    evidence_block = _format_evidence(evidence, doc_meta) or "Ninguna."
    full_docs_block = _format_full_docs(full_docs)
    conversation_block = _format_history(history)
    system_prompt = ANSWER_SYSTEM
    if "gpt-oss" in str(getattr(client, "model", "") or "").lower():
        system_prompt = f"{ANSWER_SYSTEM}\nReasoning: high"
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": ANSWER_USER.format(
                question=question,
                language=language,
                conversation=conversation_block,
                evidence=evidence_block,
                full_docs=full_docs_block,
            ),
        },
    ]
    diagnostics: dict[str, Any] = {
        "validation_errors_initial": [],
        "validation_errors_final": [],
        "repair_attempts": 0,
        "repair_success": False,
        "fallback_reason": None,
        "deterministic_fix_applied": False,
        "deterministic_fix_types": [],
        "repair_skipped_reason": None,
    }
    try:
        result = chat_json_with_retry(
            client,
            messages,
            temperature=settings.ask_synthesis_temperature,
            enable_thinking=settings.ask_synthesis_thinking,
        )
        answer_text = str(result.get("answer") or "").strip()
        citations = normalize_citations(result.get("citations"))

        if settings.answer_validator_enabled:
            outcome = validate_and_repair(
                client=client,
                question=question,
                language=language,
                evidence=evidence,
                full_docs=full_docs,
                evidence_block=evidence_block,
                full_docs_block=full_docs_block,
                answer_text=answer_text,
                citations=citations,
            )
            answer_text = outcome["answer_text"]
            citations = outcome["citations"]
            diagnostics = outcome["diagnostics"]
            if outcome.get("fallback_required"):
                # Validator rejected the synthesis: validation_fallback_answer is an
                # abstention-style summary that lists only the top few pubs. Cap the
                # attached citations to that same count so a low-confidence / out-of-
                # scope fallback doesn't surface a long citation tail — the citation
                # floor can pad `evidence` with topical siblings the summary never shows.
                fallback_max = max(1, int(getattr(settings, "answer_fallback_max_items", 3) or 3))
                fallback_citations = (citations or collect_citation_ids(evidence))[:fallback_max]
                # Keep the rejected synthesis so we can see *what* the validator threw
                # away (vs. only knowing that it dumped). Truncated; debug-only.
                diagnostics["rejected_answer"] = (answer_text or "")[:600]
                return {
                    "answer": _strip_doc_id_artifacts(
                        validation_fallback_answer(language, evidence, full_docs)
                    ),
                    "citations": fallback_citations,
                    "diagnostics": diagnostics,
                }
        else:
            if not citations:
                citations = collect_citation_ids(evidence)
            if evidence and is_no_consta_only_answer(answer_text):
                diagnostics["fallback_reason"] = "no_consta_only"
                logger.info(
                    "answer.fallback reason=no_consta_only evidence_docs=%s",
                    len(evidence),
                )
                answer_text = fallback_from_evidence(language, evidence)
        return {
            "answer": _strip_doc_id_artifacts(answer_text),
            "citations": citations,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        diagnostics["fallback_reason"] = "chat_json_failed"
        logger.warning(
            "answer.fallback reason=chat_json_failed error=%s",
            type(exc).__name__,
        )
        return {
            "answer": _strip_doc_id_artifacts(fallback_from_evidence(language, evidence)),
            "citations": collect_citation_ids(evidence),
            "diagnostics": diagnostics,
        }


def no_evidence_answer(language: str) -> dict[str, Any]:
    return {"answer": no_evidence_fallback(language), "citations": []}
