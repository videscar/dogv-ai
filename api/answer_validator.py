from __future__ import annotations

import logging
import re
from typing import Any

from .config import get_settings
from .llm import LlmClient


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
"""

settings = get_settings()
logger = logging.getLogger("dogv.answer")
ANSWER_CHAT_RETRIES = 1

_REFERENCE_RE = re.compile(r"\b\d{4}/[A-Za-z0-9Xx]+\b")
_NUMBER_RE = re.compile(r"\b\d[\d\.,]*\b")
_CVE_RE = re.compile(r"\bCVE[:\s-]*[A-Za-z0-9-]+\b", re.IGNORECASE)
_PERCENT_NUMBER_RE = re.compile(r"\b\d[\d\.,]*\s*%")
_PERCENT_WORD_RE = re.compile(r"\b(percent(?:atge|aje|ual)?s?)\b", re.IGNORECASE)
_CURRENCY_WORD_RE = re.compile(
    r"(€|\b(?:eur|euros?|euro|milions?|millones?|millon(?:es)?|milers?|miles?)\b)",
    re.IGNORECASE,
)


_NO_CONSTA_ONLY_PATTERNS = (
    r"^no consta[\s\.\!\?]*$",
    r"^no consta(?: en (?:la|las) evidencia(?:s)?(?: proporcionada(?:s)?)?)?[\s\.\!\?]*$",
    r"^no se (?:ha |han )?encontrado evidencias? suficientes.*$",
    r"^no s['’]?han trobat evid[eè]ncies suficients.*$",
    r"^no hay publicaciones encontradas.*$",
)


def is_no_consta_only_answer(answer: str) -> bool:
    if not answer:
        return True
    text = re.sub(r"\s+", " ", answer.strip().lower())
    return any(re.match(pattern, text) for pattern in _NO_CONSTA_ONLY_PATTERNS)


def collect_citation_ids(evidence: list[dict[str, Any]] | None) -> list[int]:
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


def normalize_citations(citations: Any) -> list[int]:
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


def _collect_scope_ids(
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
) -> set[int]:
    scope_ids: set[int] = set(collect_citation_ids(evidence))
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


def _source_text_for_citations(
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
    citations: list[int],
) -> str:
    if not citations:
        return _source_text(evidence, full_docs)
    scope = set(citations)
    parts: list[str] = []
    for item in evidence or []:
        doc_id = item.get("doc_id") or item.get("document_id")
        if doc_id is None or int(doc_id) not in scope:
            continue
        quote = (item.get("quote") or "").strip()
        if quote:
            parts.append(quote)
    for doc in full_docs or []:
        doc_id = doc.get("document_id")
        if doc_id is None or int(doc_id) not in scope:
            continue
        text = (doc.get("text") or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return _source_text(evidence, full_docs)
    return "\n".join(parts)


def _canonical_number_token(raw: str) -> str:
    # Normalise es/ca numbers so the same value matches regardless of how cents are
    # written: "1.600.000,00" and "1.600.000" must be equal (otherwise the claim
    # guard falsely flags a grounded figure when the answer adds ",00" cents the
    # source omits — verified dumping v2-049/v2-044). Convention: "." groups
    # thousands, "," is the decimal mark. Falls back to digit-stripping if unparsable.
    s = (raw or "").strip()
    if not s:
        return ""
    if "," in s:
        norm = s.replace(".", "").replace(",", ".")
    else:
        norm = s.replace(".", "")
    norm = re.sub(r"[^\d.]", "", norm)
    try:
        value = float(norm)
    except ValueError:
        return re.sub(r"[^\d]", "", s)
    # Currency/percent precision: keep ≤2 decimals, drop trailing zeros so
    # 1600000.00 -> "1600000" and 1366.74 -> "1366.74".
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _legacy_numeric_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(value or ""):
        canonical = _canonical_number_token(match.group(0))
        if canonical:
            tokens.add(canonical)
    return tokens


def _extract_reference_claims(value: str) -> set[str]:
    refs = {match.group(0).lower() for match in _REFERENCE_RE.finditer(value or "")}
    for match in _CVE_RE.finditer(value or ""):
        normalized = re.sub(r"\s+", "", match.group(0).lower())
        refs.add(normalized)
    return refs


def _extract_unit_claims(value: str) -> dict[str, set[str]]:
    text = value or ""
    lower = text.lower()
    currency_numbers: set[str] = set()
    percent_numbers: set[str] = set()

    for match in _PERCENT_NUMBER_RE.finditer(text):
        token = _canonical_number_token(match.group(0))
        if token:
            percent_numbers.add(token)

    for match in _NUMBER_RE.finditer(text):
        token = _canonical_number_token(match.group(0))
        if not token:
            continue
        start = max(0, match.start() - 16)
        end = min(len(text), match.end() + 16)
        window = lower[start:end]
        if "%" in window or _PERCENT_WORD_RE.search(window):
            percent_numbers.add(token)
            continue
        if _CURRENCY_WORD_RE.search(window):
            currency_numbers.add(token)

    return {"currency": currency_numbers, "percent": percent_numbers}


def _detect_unsupported_claim(
    *,
    answer_text: str,
    source_text: str,
    mode: str,
) -> bool:
    answer_refs = _extract_reference_claims(answer_text)
    source_refs = _extract_reference_claims(source_text)
    unsupported_ref = any(ref not in source_refs for ref in answer_refs)
    if mode == "refs_only":
        return unsupported_ref

    if mode == "current_strict":
        source_numbers = _legacy_numeric_tokens(source_text)
        answer_numbers = _legacy_numeric_tokens(answer_text)
        unsupported_number = any(token not in source_numbers for token in answer_numbers)
        return unsupported_ref or unsupported_number

    answer_units = _extract_unit_claims(answer_text)
    # Answer-side stays unit-aware: only currency/percent claims are policed, so
    # incidental numbers (dates, article nums, counts) don't trigger dumps. But the
    # source side checks figure *presence* among all numbers, not currency-adjacency:
    # source tables write figures bare ("A1 1.366,74 52,60 ...") while answers write
    # them with a unit ("1.366,74 euros"). Requiring currency-adjacency in the source
    # falsely flagged correct, grounded figures as unsupported (verified on v2-032/089).
    source_numbers = _legacy_numeric_tokens(source_text)
    unsupported_currency = any(token not in source_numbers for token in answer_units["currency"])
    unsupported_percent = any(token not in source_numbers for token in answer_units["percent"])
    return unsupported_ref or unsupported_currency or unsupported_percent


def validate_answer_details(
    *,
    answer_text: str,
    citations: list[int],
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    structural_errors: list[str] = []
    semantic_errors: list[str] = []
    has_evidence = bool(evidence)
    scope_ids = _collect_scope_ids(evidence, full_docs)

    if has_evidence and not citations:
        structural_errors.append("citation_missing_if_evidence")

    out_of_scope = [doc_id for doc_id in citations if scope_ids and doc_id not in scope_ids]
    if out_of_scope:
        structural_errors.append("citation_out_of_scope")

    if has_evidence and is_no_consta_only_answer(answer_text):
        semantic_errors.append("no_consta_only_with_evidence")

    source_text = _source_text_for_citations(evidence, full_docs, citations)
    claim_guard_mode = str(getattr(settings, "answer_claim_guard_mode", "unit_aware_strict") or "unit_aware_strict")
    if source_text and answer_text and _detect_unsupported_claim(
        answer_text=answer_text,
        source_text=source_text,
        mode=claim_guard_mode,
    ):
        semantic_errors.append("unsupported_numeric_or_ref_claim")

    errors = structural_errors + semantic_errors
    return {
        "errors": errors,
        "stage_structural": structural_errors,
        "stage_semantic": semantic_errors,
        "fixable": bool(structural_errors) and not semantic_errors,
        "requires_llm_repair": bool(semantic_errors),
        "scope_ids": scope_ids,
    }


def apply_deterministic_citation_fixes(
    *,
    citations: list[int],
    evidence: list[dict[str, Any]] | None,
    scope_ids: set[int],
    validation: dict[str, Any],
) -> tuple[list[int], list[str]]:
    updated = list(citations)
    fix_types: list[str] = []
    structural = set(validation.get("stage_structural") or [])

    if "citation_out_of_scope" in structural and scope_ids:
        clamped = [doc_id for doc_id in updated if doc_id in scope_ids]
        if clamped != updated:
            updated = clamped
            fix_types.append("citation_out_of_scope")

    if "citation_missing_if_evidence" in structural and evidence:
        evidence_ids = collect_citation_ids(evidence)
        if evidence_ids and not updated:
            updated = evidence_ids
            fix_types.append("citation_missing_if_evidence")

    return normalize_citations(updated), fix_types


def chat_json_with_retry(
    client: LlmClient,
    messages: list[dict[str, str]],
    temperature: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    attempts = ANSWER_CHAT_RETRIES + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.chat_json(messages, temperature=temperature, enable_thinking=enable_thinking)
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
    client: LlmClient,
    question: str,
    language: str,
    evidence_block: str,
    full_docs_block: str,
    previous_answer: str,
    previous_citations: list[int],
    validation_errors: list[str],
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
            ),
        },
    ]
    return chat_json_with_retry(client, messages, temperature=0.0)


def validate_and_repair(
    *,
    client: LlmClient,
    question: str,
    language: str,
    evidence: list[dict[str, Any]] | None,
    full_docs: list[dict[str, Any]] | None,
    evidence_block: str,
    full_docs_block: str,
    answer_text: str,
    citations: list[int],
) -> dict[str, Any]:
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

    validation = validate_answer_details(
        answer_text=answer_text,
        citations=citations,
        evidence=evidence,
        full_docs=full_docs,
    )
    validation_errors = list(validation.get("errors") or [])
    diagnostics["validation_errors_initial"] = list(validation_errors)

    citations, fix_types = apply_deterministic_citation_fixes(
        citations=citations,
        evidence=evidence,
        scope_ids=set(validation.get("scope_ids") or set()),
        validation=validation,
    )
    if fix_types:
        diagnostics["deterministic_fix_applied"] = True
        diagnostics["deterministic_fix_types"] = fix_types
        validation = validate_answer_details(
            answer_text=answer_text,
            citations=citations,
            evidence=evidence,
            full_docs=full_docs,
        )
        validation_errors = list(validation.get("errors") or [])
        if not validation_errors:
            diagnostics["repair_skipped_reason"] = "structural_only"

    if validation_errors:
        logger.info("answer.validation_failed errors=%s", ",".join(validation_errors))
        repair_mode = str(getattr(settings, "answer_repair_mode", "conditional") or "conditional")
        repair_required = False
        if repair_mode == "always":
            repair_required = True
        elif repair_mode == "none":
            diagnostics["repair_skipped_reason"] = "disabled_by_config"
        else:
            repair_required = bool(validation.get("requires_llm_repair"))
            if not repair_required:
                diagnostics["repair_skipped_reason"] = "structural_only"

        if repair_required:
            repair_attempts = max(0, int(getattr(settings, "answer_repair_attempts", 1)))
            for attempt in range(1, repair_attempts + 1):
                diagnostics["repair_attempts"] = attempt
                repaired = _repair_answer_once(
                    client=client,
                    question=question,
                    language=language,
                    evidence_block=evidence_block,
                    full_docs_block=full_docs_block,
                    previous_answer=answer_text,
                    previous_citations=citations,
                    validation_errors=validation_errors,
                )
                candidate_answer = str(repaired.get("answer") or "").strip()
                candidate_citations = normalize_citations(repaired.get("citations"))

                candidate_validation = validate_answer_details(
                    answer_text=candidate_answer,
                    citations=candidate_citations,
                    evidence=evidence,
                    full_docs=full_docs,
                )
                candidate_citations, repair_fix_types = apply_deterministic_citation_fixes(
                    citations=candidate_citations,
                    evidence=evidence,
                    scope_ids=set(candidate_validation.get("scope_ids") or set()),
                    validation=candidate_validation,
                )
                if repair_fix_types:
                    diagnostics["deterministic_fix_applied"] = True
                    all_fix_types = set(diagnostics.get("deterministic_fix_types") or [])
                    all_fix_types.update(repair_fix_types)
                    diagnostics["deterministic_fix_types"] = sorted(all_fix_types)
                    candidate_validation = validate_answer_details(
                        answer_text=candidate_answer,
                        citations=candidate_citations,
                        evidence=evidence,
                        full_docs=full_docs,
                    )

                validation_errors = list(candidate_validation.get("errors") or [])
                if not validation_errors:
                    answer_text = candidate_answer
                    citations = candidate_citations
                    diagnostics["repair_success"] = True
                    break

                logger.info(
                    "answer.repair_failed attempt=%s/%s errors=%s",
                    attempt,
                    repair_attempts,
                    ",".join(validation_errors),
                )
        elif not diagnostics.get("repair_skipped_reason"):
            diagnostics["repair_skipped_reason"] = "not_required"

        if validation_errors:
            diagnostics["validation_errors_final"] = list(validation_errors)
            diagnostics["fallback_reason"] = "validation_failed"
            logger.info(
                "answer.fallback reason=validation_failed errors=%s",
                ",".join(validation_errors),
            )
            return {
                "answer_text": answer_text,
                "citations": citations,
                "diagnostics": diagnostics,
                "fallback_required": True,
            }

    diagnostics["validation_errors_final"] = []
    return {
        "answer_text": answer_text,
        "citations": citations,
        "diagnostics": diagnostics,
        "fallback_required": False,
    }
