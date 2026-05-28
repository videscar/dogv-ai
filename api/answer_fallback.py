from __future__ import annotations

from typing import Any

from .config import get_settings

settings = get_settings()


def no_evidence_fallback(language: str) -> str:
    if language.startswith("va") or language.startswith("ca"):
        return (
            "No s'han trobat evidencies suficients al DOGV. "
            "Indica mes detalls (organisme, municipi, data aproximada) per ampliar la cerca."
        )
    return (
        "No se han encontrado evidencias suficientes en el DOGV. "
        "Indica mas detalles (organismo, municipio, fecha aproximada) para ampliar la busqueda."
    )


def fallback_from_evidence(language: str, evidence: list[dict[str, Any]] | None) -> str:
    if not evidence:
        return no_evidence_fallback(language)
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
    return header + "\n" + "\n".join(lines) if lines else no_evidence_fallback(language)


def _fallback_summary_from_sources(
    language: str,
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
    max_items: int,
) -> str:
    if not evidence and not full_docs:
        return no_evidence_fallback(language)

    doc_meta: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []

    for item in evidence or []:
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
        current = doc_meta.setdefault(doc_id, {})
        quote = (item.get("quote") or "").strip()
        if quote and not current.get("quote"):
            current["quote"] = quote

    for doc in full_docs or []:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
        current = doc_meta.setdefault(doc_id, {})
        for key in ("title", "issue_date", "ref", "text"):
            value = doc.get(key)
            if value and not current.get(key):
                current[key] = value

    if language.startswith(("va", "ca")):
        intro = "No puc confirmar una resposta unica amb seguretat. Publicacions rellevants trobades:"
        unknown_title = "Titol no disponible"
        date_label = "data"
    else:
        intro = "No puedo confirmar una respuesta unica con seguridad. Publicaciones relevantes encontradas:"
        unknown_title = "Titulo no disponible"
        date_label = "fecha"

    lines: list[str] = []
    for doc_id in ordered_ids[: max(1, max_items)]:
        meta = doc_meta.get(doc_id) or {}
        title = str(meta.get("title") or "").strip()
        issue_date = str(meta.get("issue_date") or "").strip()
        ref = str(meta.get("ref") or "").strip()
        quote = str(meta.get("quote") or "").strip()
        snippet = quote or str(meta.get("text") or "").strip()
        if not title and snippet:
            title = snippet[:160]
        title = title or unknown_title
        parts = [f"- ({doc_id}) {title}"]
        if issue_date:
            parts.append(f"{date_label}: {issue_date}")
        if ref:
            parts.append(f"ref: {ref}")
        lines.append(" | ".join(parts))

    if not lines:
        return fallback_from_evidence(language, evidence)
    return intro + "\n" + "\n".join(lines)


def _fallback_validation_message(language: str) -> str:
    if language.startswith(("va", "ca")):
        return (
            "No puc validar una resposta unica amb seguretat amb l'evidencia disponible. "
            "Revise les publicacions citades."
        )
    return (
        "No puedo validar una respuesta unica con seguridad con la evidencia disponible. "
        "Revise las publicaciones citadas."
    )


def validation_fallback_answer(
    language: str,
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
) -> str:
    style = str(getattr(settings, "answer_fallback_style", "concise_summary") or "concise_summary")
    max_items = max(1, int(getattr(settings, "answer_fallback_max_items", 3) or 3))
    if style == "raw_evidence":
        return fallback_from_evidence(language, evidence)
    if style == "explicit_validation_error":
        return _fallback_validation_message(language)
    return _fallback_summary_from_sources(language, evidence, full_docs, max_items=max_items)
