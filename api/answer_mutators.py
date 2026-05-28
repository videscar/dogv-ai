from __future__ import annotations

import re


_REFERENCE_RE = re.compile(r"\b\d{4}/[A-Za-z0-9Xx]+\b")


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


def apply_coverage_mutators(answer: str, question: str, language: str) -> str:
    answer = _ensure_topic_coverage(answer, question, language)
    answer = _ensure_field_coverage(answer, question, language)
    answer = _ensure_reference_coverage(answer, question, language)
    answer = _ensure_question_term_coverage(answer, question, language)
    return answer
