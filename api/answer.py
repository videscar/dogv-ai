from __future__ import annotations

import logging
import re
from typing import Any

from .answer_fallback import (
    fallback_from_evidence,
    no_evidence_fallback,
    validation_fallback_answer,
)
from .answer_mutators import apply_coverage_mutators
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
    "Devuelve SOLO JSON con campos: answer (texto), citations (lista de doc_id)."
)

ANSWER_USER = """Idioma de respuesta: {language}

Pregunta:
{question}

Evidencias:
{evidence}

Documentos completos (si existen):
{full_docs}

Si no hay evidencia suficiente y no hay documentos completos, di que no hay publicaciones encontradas y pide mas detalles.
Si hay documentos completos, puedes usarlos para aportar detalle, pero cita los doc_id relevantes.
Responde con el nivel de detalle que permita la evidencia y cubre cada parte de la pregunta.
Notas:
{missing_notes}
"""

settings = get_settings()
logger = logging.getLogger("dogv.answer")
ANSWER_TIMEOUT = settings.llm_timeout


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


def _format_evidence(evidence: list[dict[str, Any]] | None) -> str:
    if not evidence:
        return ""
    blocks = []
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        blocks.append(f"doc_id: {doc_id}\nquote: {quote}")
    return "\n\n---\n\n".join(blocks)


def _needs_amount(question: str) -> bool:
    return bool(re.search(r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", question, re.IGNORECASE))


def _needs_beneficiary(question: str) -> bool:
    if re.search(
        r"\b(beneficiari|beneficiario|sol·licit[a-z]*|solicit[a-z]*|destinatari|destinatario|propietari|propietario)\b",
        question,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"\b(qui|qui[eé]n)(?:\s+\w+){0,3}\s+(pot|poden|puede|pueden|sol·licit[a-z]*|solicit[a-z]*)\b",
            question,
            re.IGNORECASE,
        )
    )


def _notes_for_missing_fields(question: str, evidence: list[dict[str, Any]], full_docs: list[dict[str, Any]] | None) -> str:
    text = " ".join(str(item.get("quote", "")) for item in (evidence or []))
    if full_docs:
        text += " " + " ".join(str(doc.get("text", "")) for doc in full_docs)
    notes = []
    if _needs_amount(question):
        if not re.search(r"\d", text):
            notes.append("No hay cuantias explicitadas en las evidencias.")
    if _needs_beneficiary(question) and not re.search(r"\b(benefici|sol·licitud|solicitud|destinatari|propietari)\b", text, re.IGNORECASE):
        notes.append("No se indica quien puede solicitar/beneficiarios en las evidencias.")
    return "\n".join(notes) if notes else "Sin notas."


def build_answer(
    question: str,
    language: str,
    evidence: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = LlmClient(timeout=ANSWER_TIMEOUT)
    evidence_block = _format_evidence(evidence) or "Ninguna."
    full_docs_block = _format_full_docs(full_docs)
    missing_notes = (
        _notes_for_missing_fields(question, evidence, full_docs)
        if settings.answer_missing_notes_enabled
        else "Sin notas."
    )
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_USER.format(
                question=question,
                language=language,
                evidence=evidence_block,
                full_docs=full_docs_block,
                missing_notes=missing_notes,
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
        result = chat_json_with_retry(client, messages, temperature=1.0, enable_thinking=True)
        answer_text = str(result.get("answer") or "").strip()
        citations = normalize_citations(result.get("citations"))

        if settings.answer_mutators_enabled:
            answer_text = apply_coverage_mutators(answer_text, question, language)

        if settings.answer_validator_enabled:
            coverage_mutators = apply_coverage_mutators if settings.answer_mutators_enabled else None
            outcome = validate_and_repair(
                client=client,
                question=question,
                language=language,
                evidence=evidence,
                full_docs=full_docs,
                evidence_block=evidence_block,
                full_docs_block=full_docs_block,
                missing_notes=missing_notes,
                answer_text=answer_text,
                citations=citations,
                coverage_mutators=coverage_mutators,
            )
            answer_text = outcome["answer_text"]
            citations = outcome["citations"]
            diagnostics = outcome["diagnostics"]
            if outcome.get("fallback_required"):
                fallback_citations = citations or collect_citation_ids(evidence)
                return {
                    "answer": validation_fallback_answer(language, evidence, full_docs),
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
        return {"answer": answer_text, "citations": citations, "diagnostics": diagnostics}
    except Exception as exc:
        diagnostics["fallback_reason"] = "chat_json_failed"
        logger.warning(
            "answer.fallback reason=chat_json_failed error=%s",
            type(exc).__name__,
        )
        return {
            "answer": fallback_from_evidence(language, evidence),
            "citations": collect_citation_ids(evidence),
            "diagnostics": diagnostics,
        }


def no_evidence_answer(language: str) -> dict[str, Any]:
    return {"answer": no_evidence_fallback(language), "citations": []}
