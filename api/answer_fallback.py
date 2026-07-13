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


def fallback_from_evidence(
    language: str,
    evidence: list[dict[str, Any]] | None,
    doc_meta: dict[int, dict[str, Any]] | None = None,
) -> str:
    """List the evidence documents by TITLE, never by internal id or raw chunk.

    The old form dumped `- (113315) <chunk fragment>` lines: internal doc ids
    mean nothing to users and raw chunk fragments (annex rows, mid-sentence
    text) read as garbage. With `doc_meta` (doc_id -> {title, ref, issue_date},
    DB-backed) each document gets a human line; documents without known
    metadata are skipped rather than leaked as fragments.
    """
    if not evidence:
        return no_evidence_fallback(language)
    va = language.startswith(("va", "ca"))
    header = (
        "No puc confirmar una resposta unica amb seguretat. Publicacions rellevants trobades:"
        if va
        else "No puedo confirmar una respuesta unica con seguridad. Publicaciones relevantes encontradas:"
    )
    date_label = "data" if va else "fecha"
    lines: list[str] = []
    seen: set[int] = set()
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        info = (doc_meta or {}).get(doc_id) or {}
        title = str(info.get("title") or "").strip()
        if not title:
            continue
        parts = [f"- {title}"]
        issue_date = str(info.get("issue_date") or "").strip()
        ref = str(info.get("ref") or "").strip()
        if issue_date:
            parts.append(f"{date_label}: {issue_date}")
        if ref:
            parts.append(f"ref: {ref}")
        lines.append(" | ".join(parts))
        if len(lines) >= 10:
            break
    return header + "\n" + "\n".join(lines) if lines else no_evidence_fallback(language)


def _fallback_summary_from_sources(
    language: str,
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
    max_items: int,
    doc_meta: dict[int, dict[str, Any]] | None = None,
) -> str:
    if not evidence and not full_docs:
        return no_evidence_fallback(language)

    meta_by_id: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []

    for item in evidence or []:
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
        current = meta_by_id.setdefault(doc_id, {})
        for key in ("title", "issue_date", "ref"):
            value = ((doc_meta or {}).get(doc_id) or {}).get(key)
            if value and not current.get(key):
                current[key] = value

    for doc in full_docs or []:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
        current = meta_by_id.setdefault(doc_id, {})
        for key in ("title", "issue_date", "ref"):
            value = doc.get(key)
            if value and not current.get(key):
                current[key] = value

    if language.startswith(("va", "ca")):
        intro = (
            "No puc confirmar una resposta unica amb seguretat. Publicacions rellevants trobades:"
        )
        unknown_title = "Titol no disponible"
        date_label = "data"
    else:
        intro = "No puedo confirmar una respuesta unica con seguridad. Publicaciones relevantes encontradas:"
        unknown_title = "Titulo no disponible"
        date_label = "fecha"

    lines: list[str] = []
    for doc_id in ordered_ids[: max(1, max_items)]:
        meta = meta_by_id.get(doc_id) or {}
        # Titles come from DB-backed metadata only: internal doc ids and raw
        # chunk fragments must never surface in user-visible output.
        title = str(meta.get("title") or "").strip() or unknown_title
        issue_date = str(meta.get("issue_date") or "").strip()
        ref = str(meta.get("ref") or "").strip()
        parts = [f"- {title}"]
        if issue_date:
            parts.append(f"{date_label}: {issue_date}")
        if ref:
            parts.append(f"ref: {ref}")
        lines.append(" | ".join(parts))

    if not lines:
        return fallback_from_evidence(language, evidence, doc_meta=doc_meta)
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
    doc_meta: dict[int, dict[str, Any]] | None = None,
) -> str:
    style = str(getattr(settings, "answer_fallback_style", "concise_summary") or "concise_summary")
    max_items = max(1, int(getattr(settings, "answer_fallback_max_items", 3) or 3))
    if style == "raw_evidence":
        return fallback_from_evidence(language, evidence, doc_meta=doc_meta)
    if style == "explicit_validation_error":
        return _fallback_validation_message(language)
    return _fallback_summary_from_sources(
        language, evidence, full_docs, max_items=max_items, doc_meta=doc_meta
    )
