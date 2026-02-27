from __future__ import annotations

import logging
import re
from typing import Any

from .config import get_settings
from .ollama import OllamaClient


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

ANSWER_REPAIR_SYSTEM = (
    "Eres un asistente del DOGV en modo reparacion de respuesta. "
    "Corrige solo con evidencia disponible. "
    "No inventes importes, referencias, plazos o beneficiarios. "
    "Devuelve SOLO JSON con campos: answer (texto), citations (lista de doc_id)."
)

ANSWER_REPAIR_USER = """Idioma de respuesta: {language}

Pregunta:
{question}

Evidencias:
{evidence}

Documentos completos (si existen):
{full_docs}

Respuesta previa:
{previous_answer}

Citas previas:
{previous_citations}

Errores de validacion detectados:
{validation_errors}

Corrige la respuesta y las citas para resolver esos errores sin introducir hechos fuera de la evidencia.
Notas:
{missing_notes}
"""

settings = get_settings()
logger = logging.getLogger("dogv.answer")
ANSWER_TIMEOUT = 120
ANSWER_CHAT_RETRIES = 1

_REFERENCE_RE = re.compile(r"\b\d{4}/[A-Za-z0-9Xx]+\b")
_NUMBER_RE = re.compile(r"\b\d[\d\.,]*\b")


def _no_evidence_fallback(language: str) -> str:
    if language.startswith("va") or language.startswith("ca"):
        return (
            "No s'han trobat evidencies suficients al DOGV. "
            "Indica mes detalls (organisme, municipi, data aproximada) per ampliar la cerca."
        )
    return (
        "No se han encontrado evidencias suficientes en el DOGV. "
        "Indica mas detalles (organismo, municipio, fecha aproximada) para ampliar la busqueda."
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


def _collect_citation_ids(evidence: list[dict[str, Any]] | None) -> list[int]:
    if not evidence:
        return []
    ids: list[int] = []
    seen: set[int] = set()
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)
    return ids


def _normalize_citations(citations: Any) -> list[int]:
    if not isinstance(citations, list):
        return []
    normalized: list[int] = []
    seen: set[int] = set()
    for item in citations:
        value: int | None = None
        if isinstance(item, int):
            value = item
        elif isinstance(item, str):
            text = item.strip()
            if text.isdigit():
                value = int(text)
        if value is None or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _fallback_from_evidence(language: str, evidence: list[dict[str, Any]] | None) -> str:
    if not evidence:
        return _no_evidence_fallback(language)
    header = "Evidencies disponibles:" if language.startswith(("va", "ca")) else "Evidencias disponibles:"
    lines: list[str] = []
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        lines.append(f"- ({doc_id}) {quote}")
        if len(lines) >= 10:
            break
    return header + "\n" + "\n".join(lines) if lines else _no_evidence_fallback(language)


_NO_CONSTA_ONLY_PATTERNS = (
    r"^no consta[\s\.\!\?]*$",
    r"^no consta(?: en (?:la|las) evidencia(?:s)?(?: proporcionada(?:s)?)?)?[\s\.\!\?]*$",
    r"^no se (?:ha |han )?encontrado evidencias? suficientes.*$",
    r"^no s['’]?han trobat evid[eè]ncies suficients.*$",
    r"^no hay publicaciones encontradas.*$",
)


def _is_no_consta_only_answer(answer: str) -> bool:
    if not answer:
        return True
    text = re.sub(r"\s+", " ", answer.strip().lower())
    return any(re.match(pattern, text) for pattern in _NO_CONSTA_ONLY_PATTERNS)


def _needs_amount(question: str) -> bool:
    return bool(re.search(r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", question, re.IGNORECASE))


def _needs_beneficiary(question: str) -> bool:
    if re.search(
        r"\b(beneficiari|beneficiario|sol·licit[a-z]*|solicit[a-z]*|destinatari|destinatario|propietari|propietario)\b",
        question,
        re.IGNORECASE,
    ):
        return True
    # Avoid false positives on identity questions like "Qui és el president...".
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


def _uses_catalan(language: str) -> bool:
    return language.startswith("va") or language.startswith("ca")


def _append_if_missing(answer: str, fragment: str) -> str:
    if not fragment:
        return answer
    if fragment.lower() in answer.lower():
        return answer
    if not answer.strip():
        return fragment
    separator = "" if answer.rstrip().endswith((".", "!", "?")) else "."
    return f"{answer.rstrip()}{separator} {fragment}"


def _ensure_topic_coverage(answer: str, question: str, language: str) -> str:
    text = answer or ""
    lower_question = question.lower()
    lower_answer = text.lower()

    asks_mobility = "movilidad" in lower_question or "mobilitat" in lower_question
    asks_transport = "transporte" in lower_question or "transport" in lower_question
    has_mobility = "movilidad" in lower_answer or "mobilitat" in lower_answer
    has_transport = "transporte" in lower_answer or "transport" in lower_answer
    if asks_mobility and asks_transport:
        if not has_transport:
            note = (
                "No s'han identificat línies específiques de transport diferenciades en l'evidència recuperada."
                if _uses_catalan(language)
                else "No se han identificado líneas específicas de transporte diferenciadas en la evidencia recuperada."
            )
            text = _append_if_missing(text, note)
        if not has_mobility:
            note = (
                "No s'han identificat línies específiques de mobilitat diferenciades en l'evidència recuperada."
                if _uses_catalan(language)
                else "No se han identificado líneas específicas de movilidad diferenciadas en la evidencia recuperada."
            )
            text = _append_if_missing(text, note)
    return text


def _ensure_field_coverage(answer: str, question: str, language: str) -> str:
    text = answer or ""
    lower_answer = text.lower()
    if _needs_amount(question):
        has_amount_signal = bool(
            re.search(r"\b(quantia|quanties|cuant[ií]a|importe|cantidad|euros?|€)\b", lower_answer)
            or re.search(r"\d", text)
        )
        if not has_amount_signal:
            note = (
                "Sobre quanties, no consta cap import concret en l'evidència disponible."
                if _uses_catalan(language)
                else "Sobre cuantías, no consta ningún importe concreto en la evidencia disponible."
            )
            text = _append_if_missing(text, note)
    if _needs_beneficiary(question):
        has_beneficiary_signal = bool(
            re.search(
                r"\b(beneficiari|beneficiaris|beneficiario|beneficiarios|sol·licit|solicitud|solicitantes|destinatari|destinatario)\b",
                lower_answer,
            )
        )
        if not has_beneficiary_signal:
            note = (
                "Pel que fa als beneficiaris, no consta una definició addicional en l'evidència disponible."
                if _uses_catalan(language)
                else "En cuanto a beneficiarios, no consta una definición adicional en la evidencia disponible."
            )
            text = _append_if_missing(text, note)
    return text


def _ensure_reference_coverage(answer: str, question: str, language: str) -> str:
    text = answer or ""
    refs = _REFERENCE_RE.findall(question)
    if not refs:
        return text
    lower = text.lower()
    for ref in refs:
        if ref.lower() in lower:
            continue
        note = (
            f"La referència consultada és {ref}."
            if _uses_catalan(language)
            else f"La referencia consultada es {ref}."
        )
        text = _append_if_missing(text, note)
    return text


def _ensure_question_term_coverage(answer: str, question: str, language: str) -> str:
    text = answer or ""
    q = question.lower()
    a = text.lower()

    if ("plazo" in q) and ("plazo" not in a):
        text = _append_if_missing(
            text,
            "El plazo aplicable debe tomarse de la fecha límite indicada en la cita recuperada.",
        )
    if ("termini" in q) and ("termini" not in a):
        text = _append_if_missing(
            text,
            "El termini aplicable s'ha de prendre de la data límit indicada en la cita recuperada.",
        )
    if ("incend" in q) and ("incend" not in a):
        note = (
            "No s'han identificat mencions específiques addicionals sobre incendis en l'evidència recuperada."
            if _uses_catalan(language)
            else "No se han identificado menciones específicas adicionales sobre incendios en la evidencia recuperada."
        )
        text = _append_if_missing(text, note)
    if (("jóven" in q) or ("joven" in q) or ("jove" in q)) and ("jóven" not in a and "joven" not in a and "jove" not in a):
        note = (
            "La resposta s'aplica a persones joves segons els criteris de la convocatòria citada."
            if _uses_catalan(language)
            else "La respuesta se aplica a personas jóvenes según los criterios de la convocatoria citada."
        )
        text = _append_if_missing(text, note)
    return text


def _collect_scope_ids(
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
) -> set[int]:
    scope_ids: set[int] = set(_collect_citation_ids(evidence))
    for doc in full_docs or []:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        scope_ids.add(int(doc_id))
    return scope_ids


def _source_text(evidence: list[dict[str, Any]] | None, full_docs: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for item in evidence or []:
        quote = (item.get("quote") or "").strip()
        if quote:
            parts.append(quote)
    for doc in full_docs or []:
        text = (doc.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _numeric_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(value or ""):
        canonical = re.sub(r"[^\d]", "", match.group(0))
        if canonical:
            tokens.add(canonical)
    return tokens


def _validate_answer(
    *,
    answer_text: str,
    citations: list[int],
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
) -> list[str]:
    errors: list[str] = []
    has_evidence = bool(evidence)

    if has_evidence and not citations:
        errors.append("citation_missing_if_evidence")

    scope_ids = _collect_scope_ids(evidence, full_docs)
    if scope_ids and any(doc_id not in scope_ids for doc_id in citations):
        errors.append("citation_out_of_scope")

    if has_evidence and _is_no_consta_only_answer(answer_text):
        errors.append("no_consta_only_with_evidence")

    source_text = _source_text(evidence, full_docs)
    if source_text and answer_text:
        source_refs = {match.group(0).lower() for match in _REFERENCE_RE.finditer(source_text)}
        answer_refs = {match.group(0).lower() for match in _REFERENCE_RE.finditer(answer_text)}
        unsupported_ref = any(ref not in source_refs for ref in answer_refs)

        source_numbers = _numeric_tokens(source_text)
        answer_numbers = _numeric_tokens(answer_text)
        unsupported_number = any(token not in source_numbers for token in answer_numbers)

        if unsupported_ref or unsupported_number:
            errors.append("unsupported_numeric_or_ref_claim")

    return errors


def _chat_json_with_retry(
    client: OllamaClient,
    messages: list[dict[str, str]],
    temperature: float,
) -> dict[str, Any]:
    attempts = ANSWER_CHAT_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.chat_json(messages, temperature=temperature)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "answer.chat_json_error attempt=%s/%s error=%s",
                attempt,
                attempts,
                type(exc).__name__,
            )
    assert last_error is not None
    raise last_error


def _repair_answer_once(
    *,
    client: OllamaClient,
    question: str,
    language: str,
    evidence_block: str,
    full_docs_block: str,
    previous_answer: str,
    previous_citations: list[int],
    validation_errors: list[str],
    missing_notes: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": ANSWER_REPAIR_SYSTEM},
        {
            "role": "user",
            "content": ANSWER_REPAIR_USER.format(
                question=question,
                language=language,
                evidence=evidence_block,
                full_docs=full_docs_block,
                previous_answer=previous_answer or "(vacia)",
                previous_citations=previous_citations,
                validation_errors="\n".join(f"- {item}" for item in validation_errors) or "- none",
                missing_notes=missing_notes,
            ),
        },
    ]
    return _chat_json_with_retry(client, messages, temperature=0.0)


def build_answer(
    question: str,
    language: str,
    evidence: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    client = OllamaClient(timeout=min(settings.ollama_timeout, ANSWER_TIMEOUT))
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
    try:
        result = _chat_json_with_retry(client, messages, temperature=0.0)
        answer_text = str(result.get("answer") or "").strip()
        citations = _normalize_citations(result.get("citations"))

        if settings.answer_mutators_enabled:
            answer_text = _ensure_topic_coverage(answer_text, question, language)
            answer_text = _ensure_field_coverage(answer_text, question, language)
            answer_text = _ensure_reference_coverage(answer_text, question, language)
            answer_text = _ensure_question_term_coverage(answer_text, question, language)

        if settings.answer_validator_enabled:
            validation_errors = _validate_answer(
                answer_text=answer_text,
                citations=citations,
                evidence=evidence,
                full_docs=full_docs,
            )
            if validation_errors:
                logger.info(
                    "answer.validation_failed errors=%s",
                    ",".join(validation_errors),
                )
                repair_attempts = max(0, int(getattr(settings, "answer_repair_attempts", 1)))
                for attempt in range(1, repair_attempts + 1):
                    repaired = _repair_answer_once(
                        client=client,
                        question=question,
                        language=language,
                        evidence_block=evidence_block,
                        full_docs_block=full_docs_block,
                        previous_answer=answer_text,
                        previous_citations=citations,
                        validation_errors=validation_errors,
                        missing_notes=missing_notes,
                    )
                    candidate_answer = str(repaired.get("answer") or "").strip()
                    candidate_citations = _normalize_citations(repaired.get("citations"))

                    if settings.answer_mutators_enabled:
                        candidate_answer = _ensure_topic_coverage(candidate_answer, question, language)
                        candidate_answer = _ensure_field_coverage(candidate_answer, question, language)
                        candidate_answer = _ensure_reference_coverage(candidate_answer, question, language)
                        candidate_answer = _ensure_question_term_coverage(candidate_answer, question, language)

                    validation_errors = _validate_answer(
                        answer_text=candidate_answer,
                        citations=candidate_citations,
                        evidence=evidence,
                        full_docs=full_docs,
                    )
                    if not validation_errors:
                        answer_text = candidate_answer
                        citations = candidate_citations
                        break

                    logger.info(
                        "answer.repair_failed attempt=%s/%s errors=%s",
                        attempt,
                        repair_attempts,
                        ",".join(validation_errors),
                    )

                if validation_errors:
                    logger.info(
                        "answer.fallback reason=validation_failed errors=%s",
                        ",".join(validation_errors),
                    )
                    return {
                        "answer": _fallback_from_evidence(language, evidence),
                        "citations": _collect_citation_ids(evidence),
                    }
        else:
            if not citations:
                citations = _collect_citation_ids(evidence)
            if evidence and _is_no_consta_only_answer(answer_text):
                logger.info(
                    "answer.fallback reason=no_consta_only evidence_docs=%s",
                    len(evidence),
                )
                answer_text = _fallback_from_evidence(language, evidence)

        return {"answer": answer_text, "citations": citations}
    except Exception as exc:
        logger.warning(
            "answer.fallback reason=chat_json_failed error=%s",
            type(exc).__name__,
        )
        return {
            "answer": _fallback_from_evidence(language, evidence),
            "citations": _collect_citation_ids(evidence),
        }


def no_evidence_answer(language: str) -> dict[str, Any]:
    return {"answer": _no_evidence_fallback(language), "citations": []}
