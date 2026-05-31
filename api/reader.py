from __future__ import annotations

from typing import Any

from .llm import LlmClient
from .reader_extractors import (
    _clean_evidence,
    _eligibility_evidence,
    _extract_entities,
    _extract_keywords,
    _extract_phrases,
    _fallback_evidence,
    _looks_like_list,
    _matched_keywords,
    _program_total_evidence,
    _score_text,
)


READER_SYSTEM = (
    "Eres un analista legal del DOGV. "
    "Extrae evidencia primero: cita fragmentos textuales (quote) y luego resume en detail. "
    "No hagas afirmaciones sin quote literal. "
    "Devuelve SOLO JSON con un campo 'evidence' que sea una lista de objetos "
    "con: doc_id, quote, detail (resumen corto). "
    "Devuelve hasta 8 citas. Prioriza importes totales, cuantias, beneficiarios, requisitos, plazos y referencias. "
    "Si la pregunta pide quién puede solicitar o requisitos, incluye citas de requisitos/beneficiarios. "
    "Evita listas de personas o expedientes; si solo hay lineas de anexo, marcala como ejemplo y no generalices."
)

READER_USER = """Pregunta:
{question}

Documentos seleccionados (cada uno con extractos):
{docs}
"""


_PINNED_DETAILS = {
    "Extracto con cantidades.",
    "Importe total/global de la convocatoria o bases.",
}


def _phrase_hits(text: str, phrases: list[str]) -> int:
    if not text or not phrases:
        return 0
    lower = text.lower()
    return sum(1 for phrase in phrases if phrase in lower)


def _matched_phrases(text: str, phrases: list[str]) -> set[str]:
    if not text or not phrases:
        return set()
    lower = text.lower()
    return {phrase for phrase in phrases if phrase in lower}


def _entity_hits(text: str, entities: list[str]) -> int:
    if not text or not entities:
        return 0
    lower = text.lower()
    return sum(1 for entity in entities if entity in lower)


def _matched_entities(text: str, entities: list[str]) -> set[str]:
    if not text or not entities:
        return set()
    lower = text.lower()
    return {entity for entity in entities if entity in lower}


def _select_diverse_chunks(
    chunks: list[str],
    keywords: list[str],
    phrases: list[str],
    entities: list[str],
    max_chunks: int,
    already_covered: set[str] | None = None,
) -> list[str]:
    if not chunks or max_chunks <= 0:
        return []
    targets = set(keywords) | set(phrases) | set(entities)
    if not targets:
        return []
    uncovered = set(targets)
    if already_covered:
        uncovered -= already_covered
    if not uncovered:
        return []
    remaining = list(chunks)
    selected: list[str] = []

    while remaining and len(selected) < max_chunks:
        best_idx = None
        best_score = (0, 0, 0, 0)
        for idx, chunk in enumerate(remaining):
            phrase_matches = _matched_phrases(chunk, phrases)
            keyword_matches = _matched_keywords(chunk, keywords)
            entity_matches = _matched_entities(chunk, entities)
            if not phrase_matches and not keyword_matches and not entity_matches:
                continue
            hits = phrase_matches | keyword_matches | entity_matches
            new_hits = hits & uncovered if uncovered else hits
            if not new_hits:
                continue
            score = (
                len(new_hits),
                len(phrase_matches) + len(entity_matches),
                len(keyword_matches),
                _score_text(chunk, list(keyword_matches)) + len(entity_matches),
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        uncovered -= _matched_phrases(chosen, phrases) | _matched_keywords(chosen, keywords)
        if not uncovered:
            break
    return selected


def _coverage_candidates(
    docs: list[dict[str, Any]],
    keywords: list[str],
    phrases: list[str],
    entities: list[str],
) -> dict[int, tuple[tuple[int, int, int, int], str]]:
    best: dict[int, tuple[tuple[int, int, int, int], str]] = {}
    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        best_score = (0, 0, 0, 0)
        best_chunk = ""
        for chunk in doc.get("chunks") or []:
            matches = _matched_keywords(chunk, keywords)
            phrase_score = _phrase_hits(chunk, phrases)
            entity_score = _entity_hits(chunk, entities)
            if not matches and phrase_score == 0 and entity_score == 0:
                continue
            score = (phrase_score, entity_score, len(matches), _score_text(chunk, list(matches)))
            if score > best_score:
                best_score = score
                best_chunk = chunk
        if best_score[0] > 0 or best_score[1] > 0 or best_score[2] > 0:
            best[int(doc_id)] = (best_score, best_chunk.strip())
    return best


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in items:
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        key = (int(doc_id), quote)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"doc_id": int(doc_id), "quote": quote, "detail": item.get("detail")})
    return deduped


def _coverage_rank_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    max_items: int = 8,
    max_extra: int = 4,
    per_doc_chunks: int = 3,
) -> list[dict[str, Any]]:
    if not evidence or not docs:
        return evidence
    keywords = _extract_keywords(question)
    if not keywords:
        return evidence
    phrases = _extract_phrases(question)
    entities = _extract_entities(question)
    best_by_doc = _coverage_candidates(docs, keywords, phrases, entities)

    scored: list[tuple[int, int, int, int, int, bool, bool, dict[str, Any]]] = []
    for idx, item in enumerate(evidence):
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "")
        matches = _matched_keywords(quote, keywords)
        phrase_score = _phrase_hits(quote, phrases)
        entity_score = _entity_hits(quote, entities)
        score = (phrase_score, entity_score, len(matches), _score_text(quote, list(matches)))
        if score[0] == 0 and score[1] == 0 and score[2] == 0 and doc_id is not None:
            fallback = best_by_doc.get(int(doc_id))
            if fallback:
                score = fallback[0]
        detail = item.get("detail") or ""
        is_pinned = detail in _PINNED_DETAILS
        is_list_like = _looks_like_list(quote)
        scored.append((-score[0], -score[1], -score[2], -score[3], idx, is_pinned, is_list_like, item))

    existing_docs = {
        int(item.get("doc_id") or item.get("document_id"))
        for item in evidence
        if item.get("doc_id") or item.get("document_id")
    }
    extras: list[dict[str, Any]] = []
    extra_limit = max_extra
    if phrases:
        extra_limit = max_extra + 2
    if entities:
        extra_limit = max(extra_limit, max_extra + 2)
    for doc_id, (score, chunk) in sorted(best_by_doc.items(), key=lambda x: x[1][0], reverse=True):
        if doc_id in existing_docs:
            continue
        if score[0] == 0 and score[1] == 0 and score[2] == 0:
            continue
        if entities and score[1] == 0:
            continue
        if phrases and score[0] == 0 and score[1] == 0:
            continue
        extras.append(
            {
                "doc_id": doc_id,
                "quote": (chunk or "")[:800],
                "detail": "Extracto con coincidencias lexicas.",
            }
        )
        if len(extras) >= extra_limit:
            break

    if extras:
        idx = len(scored)
        for extra in extras:
            quote = (extra.get("quote") or "")
            matches = _matched_keywords(quote, keywords)
            phrase_score = _phrase_hits(quote, phrases)
            entity_score = _entity_hits(quote, entities)
            score = (phrase_score, entity_score, len(matches), _score_text(quote, list(matches)))
            is_list_like = _looks_like_list(quote)
            scored.append((-score[0], -score[1], -score[2], -score[3], idx, False, is_list_like, extra))
            idx += 1

    has_phrase_hits = False
    has_entity_hits = False
    if phrases:
        for phrase_score, _, _, _, _, is_pinned, _, _ in scored:
            if is_pinned:
                continue
            if phrase_score < 0:
                has_phrase_hits = True
                break
    if entities:
        for _, entity_score, _, _, _, is_pinned, _, _ in scored:
            if is_pinned:
                continue
            if entity_score < 0:
                has_entity_hits = True
                break

    if has_entity_hits:
        scored = [item for item in scored if item[5] or item[1] < 0]
    elif has_phrase_hits:
        scored = [item for item in scored if item[5] or item[0] < 0]

    scored.sort()
    ordered = [item for *_, item in scored]

    combined = _dedupe_evidence(ordered)
    if len(combined) <= max_items:
        return _augment_with_diverse_chunks(
            combined,
            docs,
            keywords,
            phrases,
            entities,
            max_items=max_items,
            per_doc_chunks=per_doc_chunks,
        )

    pinned = [item for item in combined if (item.get("detail") or "") in _PINNED_DETAILS]
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for item in pinned:
        doc_id = item.get("doc_id")
        quote = (item.get("quote") or "")
        if doc_id is None or not quote:
            continue
        key = (int(doc_id), quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= max_items:
            return result

    for item in combined:
        doc_id = item.get("doc_id")
        quote = (item.get("quote") or "")
        if doc_id is None or not quote:
            continue
        key = (int(doc_id), quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) >= max_items:
            break

    return _augment_with_diverse_chunks(
        result,
        docs,
        keywords,
        phrases,
        entities,
        max_items=max_items,
        per_doc_chunks=per_doc_chunks,
    )


def _augment_with_diverse_chunks(
    evidence: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    keywords: list[str],
    phrases: list[str],
    entities: list[str],
    max_items: int,
    per_doc_chunks: int,
) -> list[dict[str, Any]]:
    if not evidence or not docs or len(evidence) >= max_items:
        return evidence
    if per_doc_chunks <= 1:
        return evidence

    doc_chunks: dict[int, list[str]] = {}
    for doc in docs:
        doc_id = doc.get("document_id")
        if doc_id is None:
            continue
        chunks = [c for c in (doc.get("chunks") or []) if c]
        if chunks:
            doc_chunks[int(doc_id)] = chunks

    per_doc_quotes: dict[int, set[str]] = {}
    ordered_doc_ids: list[int] = []
    for item in evidence:
        doc_id = item.get("doc_id") or item.get("document_id")
        quote = (item.get("quote") or "").strip()
        if not doc_id or not quote:
            continue
        doc_id = int(doc_id)
        per_doc_quotes.setdefault(doc_id, set()).add(quote)
        if doc_id not in ordered_doc_ids:
            ordered_doc_ids.append(doc_id)

    added: list[dict[str, Any]] = []
    for doc_id in ordered_doc_ids:
        if len(evidence) + len(added) >= max_items:
            break
        chunks = doc_chunks.get(doc_id) or []
        if not chunks:
            continue
        covered: set[str] = set()
        for quote in per_doc_quotes.get(doc_id, set()):
            covered |= _matched_keywords(quote, keywords)
            covered |= _matched_phrases(quote, phrases)
            covered |= _matched_entities(quote, entities)
        selected_chunks = _select_diverse_chunks(
            chunks,
            keywords,
            phrases,
            entities,
            per_doc_chunks,
            already_covered=covered,
        )
        for chunk in selected_chunks:
            if len(evidence) + len(added) >= max_items:
                break
            quote = chunk.strip()[:800]
            if not quote:
                continue
            if quote in per_doc_quotes.get(doc_id, set()):
                continue
            per_doc_quotes.setdefault(doc_id, set()).add(quote)
            added.append(
                {
                    "doc_id": doc_id,
                    "quote": quote,
                    "detail": "Extracto con coincidencias lexicas.",
                }
            )

    if not added:
        return evidence
    return _dedupe_evidence(evidence + added)


def extract_evidence(
    question: str,
    docs: list[dict[str, Any]],
    full_docs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    blocks = []
    for doc in docs:
        chunks = "\n".join(doc.get("chunks") or [])
        blocks.append(
            f"doc_id: {doc['document_id']}\n"
            f"fecha: {doc.get('issue_date')}\n"
            f"titulo: {doc.get('title')}\n"
            f"ref: {doc.get('ref')}\n"
            f"extractos:\n{chunks}"
        )

    client = LlmClient()
    messages = [
        {"role": "system", "content": READER_SYSTEM},
        {"role": "user", "content": READER_USER.format(question=question, docs="\n\n---\n\n".join(blocks))},
    ]
    try:
        result = client.chat_json(messages, temperature=0.0, enable_thinking=False)
        evidence = _clean_evidence(result.get("evidence") or [])
        eligibility = _eligibility_evidence(question, docs, full_docs=full_docs)
        program_total = _program_total_evidence(question, docs, full_docs=full_docs)
        if evidence:
            combined = evidence
        else:
            combined = _fallback_evidence(question, docs, full_docs=full_docs)
        if eligibility:
            combined = combined + eligibility
        # NB: _numeric_evidence (count-based "most numbers wins") is intentionally NOT
        # mixed in: when active it pins amount-dense-but-wrong chunks that displace the
        # correct figure (v2-049: 1.6M chunk evicted by an umbral table) or inject
        # confusing figures that trip the validator (v2-044 dump). The targeted
        # _program_total pin gives the convocatoria total without that collateral.
        if program_total:
            combined = program_total + combined
        return _coverage_rank_evidence(question, combined, docs)
    except Exception:
        fallback = _fallback_evidence(question, docs, full_docs=full_docs)
        eligibility = _eligibility_evidence(question, docs, full_docs=full_docs)
        program_total = _program_total_evidence(question, docs, full_docs=full_docs)
        combined = fallback
        if eligibility:
            combined = combined + eligibility
        if program_total:
            combined = program_total + combined
        return _coverage_rank_evidence(question, combined, docs)
