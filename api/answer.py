from __future__ import annotations

from typing import Any
import re

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

settings = get_settings()
ANSWER_TIMEOUT = 120


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
    return bool(
        re.search(
            r"\b(qui|qu[ií]en|beneficiari|beneficiario|sol·licit|solicitar|destinatari|destinatario|propietari|propietario)\b",
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
    client = OllamaClient(timeout=min(settings.ollama_timeout, ANSWER_TIMEOUT))
    evidence_block = _format_evidence(evidence) or "Ninguna."
    full_docs_block = _format_full_docs(full_docs)
    missing_notes = _notes_for_missing_fields(question, evidence, full_docs)
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
        result = client.chat_json(messages, temperature=0.0)
        citations_raw = result.get("citations")
        normalized: list[int] = []
        if isinstance(citations_raw, list):
            for item in citations_raw:
                if isinstance(item, int):
                    normalized.append(item)
                elif isinstance(item, str):
                    text = item.strip()
                    if text.isdigit():
                        normalized.append(int(text))
        if not normalized:
            normalized = _collect_citation_ids(evidence)
        answer_text = str(result.get("answer") or "").strip()
        if evidence and _is_no_consta_only_answer(answer_text):
            result["answer"] = _fallback_from_evidence(language, evidence)
        result["citations"] = normalized
        return result
    except Exception:
        return {
            "answer": _fallback_from_evidence(language, evidence),
            "citations": _collect_citation_ids(evidence),
        }


def no_evidence_answer(language: str) -> dict[str, Any]:
    return {"answer": _no_evidence_fallback(language), "citations": []}
