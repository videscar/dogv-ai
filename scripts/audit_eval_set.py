from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.config import enabled_lanes, get_settings
from api.db import SessionLocal
from api.ollama import OllamaClient
from api.query_expansion import build_bm25_queries, guess_language
from api.retrieval import (
    RetrievalFilters,
    bm25_search,
    rrf_fuse,
    title_bm25_search,
    title_vector_search,
    top_chunks_for_docs,
    vector_search,
)


AUDIT_SYSTEM = (
    "Eres un auditor de ground truth para evaluacion de recuperacion documental del DOGV. "
    "Tu objetivo es proponer conjuntos minimos y completos de documentos necesarios para responder la pregunta. "
    "Si una sola disposicion basta, devuelve un set de 1 doc_id. "
    "Si para responder completamente hacen falta varias disposiciones, devuelve sets multidocumento. "
    "Puedes devolver varios sets equivalentes alternativos (maximo 3). "
    "Usa SOLO ids de la lista de candidatos. "
    "Devuelve SOLO JSON con campos: status (ok|uncertain), reason, gold_sets (lista de listas de int)."
)

AUDIT_USER = """Pregunta:
{question}

Doc_ids actuales (legacy): {existing_doc_ids}

Candidatos (id | fecha | ref | titulo | extracto):
{candidates}

Reglas:
- Un gold_set debe cubrir TODAS las partes de la pregunta.
- Prefiere sets minimos.
- No inventes ids fuera de candidatos.
"""

settings = get_settings()

VERIFY_SYSTEM = (
    "Eres un validador de ground truth para recuperacion documental. "
    "Debes decidir entre set legacy y set propuesto. "
    "Si la pregunta implica actualidad/novedad, prioriza el documento mas reciente que responda la pregunta. "
    "Devuelve SOLO JSON con campos: decision (keep_legacy|accept_proposed|both_valid), reason."
)

VERIFY_USER = """Pregunta:
{question}

Set legacy: {legacy}
Set propuesto: {proposed}

Candidatos relevantes (id | fecha | ref | titulo):
{candidates}
"""


@dataclass
class AuditResult:
    gold_sets: list[list[int]]
    status: str
    reason: str
    changed: bool


def _is_compound_question(text: str) -> bool:
    lower = (text or "").lower()
    return lower.count("?") >= 2 or bool(
        re.search(r"\b(y|i|adem[aá]s|tamb[ié]n|quina|cu[aá]l|qui[eé]n)\b", lower)
    )


def _needs_strict_detail(text: str) -> bool:
    return bool(
        re.search(
            r"\b(quantia|cuant[ií]a|importe|cantidad|beneficiar|beneficiari|beneficiario|sol·licit|solicitar|requisit|requisito|destinatari|destinatario)\b",
            (text or "").lower(),
        )
    )


def _looks_like_modifier_title(title: str) -> bool:
    lower = (title or "").lower()
    return bool(
        re.search(
            r"\b(incrementa|incremento|modifica|modificaci[oó]n|correcci[oó]n|correcció|pr[oò]rroga|ampl[ií]a|habilita nuevo plazo|nou termini)\b",
            lower,
        )
    )


def _required_signals(question: str) -> set[str]:
    required: set[str] = set()
    lower = (question or "").lower()
    if re.search(r"\b(quantia|cuant[ií]a|importe|cantidad|euros?|€)\b", lower):
        required.add("amount")
    if re.search(
        r"\b(beneficiar|beneficiari|beneficiario|sol·licit|solicitar|requisit|requisito|destinatari|destinatario|pot|puede|poden|pueden)\b",
        lower,
    ):
        required.add("eligibility")
    if re.search(
        r"\b(termini|plazo|fecha l[ií]mite|fins|hasta|presentaci[oó]n de solicitudes|presentaci[oó] de sol·licituds)\b",
        lower,
    ):
        required.add("deadline")
    return required


def _text_signals(text: str) -> set[str]:
    signals: set[str] = set()
    lower = (text or "").lower()
    if not lower:
        return signals
    if re.search(r"(€|euros?|\\b\\d[\\d\\.,]*\\b)", lower) and re.search(
        r"\b(quantia|cuant[ií]a|importe|cantidad|ayuda|ajuda|subvenci[oó]n|subvenci[oó])\b",
        lower,
    ):
        signals.add("amount")
    if re.search(
        r"\b(beneficiar|beneficiari|beneficiario|sol·licit|solicitud|requisit|requisito|destinatari|destinatario|titular|podran|podrán|pueden|pot)\b",
        lower,
    ):
        signals.add("eligibility")
    if re.search(
        r"\b(termini|plazo|fecha l[ií]mite|fins al|hasta el|presentaci[oó]n de solicitudes|presentaci[oó] de sol·licituds)\b",
        lower,
    ):
        signals.add("deadline")
    return signals


def _normalize_set(values: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    if not isinstance(values, list):
        return out
    for item in values:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return sorted(out)


def _normalize_gold_sets(raw: Any) -> list[list[int]]:
    if not isinstance(raw, list):
        return []
    sets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for item in raw:
        normalized = _normalize_set(item)
        if not normalized:
            continue
        key = tuple(normalized)
        if key in seen:
            continue
        seen.add(key)
        sets.append(normalized)
    return sets


def _legacy_doc_ids(entry: dict[str, Any]) -> list[int]:
    if isinstance(entry.get("doc_ids"), list):
        values = _normalize_set(entry.get("doc_ids"))
        if values:
            return values
    if entry.get("doc_id") is not None:
        try:
            return [int(entry["doc_id"])]
        except (TypeError, ValueError):
            return []
    return []


def _retrieve_candidate_ids(
    question: str,
    language: str | None,
    max_candidates: int,
) -> tuple[list[int], list[dict[str, Any]], list[float] | None]:
    client = OllamaClient(timeout=min(settings.ollama_timeout, 90))
    lanes = enabled_lanes(settings)
    query_embedding: list[float] | None = None
    intent = {
        "language": "ca" if (language or "").startswith("va") else "es",
        "doc_kind": None,
        "doc_subkind": None,
        "keywords": [],
        "since_date": None,
        "until_date": None,
        "needs_online": False,
        "entities": {},
    }
    bm25_query, _ = build_bm25_queries(question, intent)
    language_value = language or guess_language(question)
    filters = RetrievalFilters(language=language_value)
    with SessionLocal() as db:
        if "vector" in lanes:
            query_embedding = client.embed(question)
            vector_hits = vector_search(db, query_embedding, filters, limit=max_candidates)
        else:
            vector_hits = []
        if "bm25" in lanes:
            bm25_hits = bm25_search(db, bm25_query, filters, limit=max_candidates)
        else:
            bm25_hits = []
        if "title" in lanes and query_embedding is not None:
            title_hits = title_vector_search(db, query_embedding, filters, limit=max_candidates)
        else:
            title_hits = []
        if "title" in lanes:
            title_lexical_hits = title_bm25_search(db, bm25_query, filters, limit=max_candidates)
        else:
            title_lexical_hits = []

    sources: list[list[dict[str, Any]]] = []
    weights: list[float] = []
    if "vector" in lanes:
        sources.append(vector_hits)
        weights.append(getattr(settings, "ask_rrf_weight_vector", 1.0))
    if "bm25" in lanes:
        sources.append(bm25_hits)
        weights.append(getattr(settings, "ask_rrf_weight_bm25", 1.0))
    if "title" in lanes:
        sources.append(title_hits)
        weights.append(getattr(settings, "ask_rrf_weight_title", 1.0))
        if title_lexical_hits:
            sources.append(title_lexical_hits)
            weights.append(getattr(settings, "ask_rrf_weight_title_lexical", 0.8))

    fused = rrf_fuse(sources, max_docs=max_candidates, weights=weights) if sources else []
    out: list[int] = []
    seen: set[int] = set()
    for item in fused[:max_candidates]:
        doc_id = item.get("document_id")
        if doc_id is None:
            continue
        doc_id = int(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out, fused, query_embedding


def _extract_ref_tokens(text: str) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"\b\d{1,4}/\d{4}\b", text)
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out[:6]


def _find_docs_by_ref_tokens(tokens: list[str], max_per_token: int = 5) -> list[int]:
    if not tokens:
        return []
    found: list[int] = []
    seen: set[int] = set()
    with SessionLocal() as db:
        for token in tokens:
            rows = db.execute(
                sa_text(
                    """
                    SELECT dd.id
                    FROM dogv_documents dd
                    JOIN dogv_issues di ON di.id = dd.issue_id
                    WHERE dd.title ILIKE :token OR dd.ref ILIKE :token
                    ORDER BY di.date DESC, dd.id DESC
                    LIMIT :limit
                    """
                ),
                {"token": f"%{token}%", "limit": max_per_token},
            ).mappings().all()
            for row in rows:
                doc_id = int(row["id"])
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                found.append(doc_id)
    return found


def _fetch_candidate_rows(
    doc_ids: list[int],
    query_embedding: list[float] | None,
    max_snippet_chars: int,
) -> list[dict[str, Any]]:
    if not doc_ids:
        return []
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                """
                SELECT
                    dd.id AS document_id,
                    dd.title,
                    dd.ref,
                    dd.doc_kind,
                    dd.doc_subkind,
                    di.date AS issue_date,
                    di.language,
                    rd.summary
                FROM dogv_documents dd
                JOIN dogv_issues di ON di.id = dd.issue_id
                LEFT JOIN rag_doc rd ON rd.document_id = dd.id
                WHERE dd.id = ANY(:doc_ids)
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
        chunk_map = top_chunks_for_docs(db, query_embedding, doc_ids, per_doc=1) if query_embedding else {}

    mapped = {int(row["document_id"]): dict(row) for row in rows}
    result: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        row = mapped.get(int(doc_id))
        if not row:
            continue
        chunk_items = chunk_map.get(int(doc_id)) or []
        snippet = ""
        if chunk_items:
            snippet = (chunk_items[0].get("text") or "").strip()
        if not snippet:
            snippet = (row.get("summary") or "").strip()
        if snippet:
            snippet = " ".join(snippet.split())
            snippet = snippet[:max_snippet_chars]
        row["snippet"] = snippet
        result.append(row)
    return result


def _fetch_text_map(doc_ids: list[int], max_chars: int = 60000) -> dict[int, str]:
    if not doc_ids:
        return {}
    with SessionLocal() as db:
        rows = db.execute(
            sa_text(
                """
                SELECT id AS document_id, text
                FROM dogv_documents
                WHERE id = ANY(:doc_ids)
                """
            ),
            {"doc_ids": doc_ids},
        ).mappings().all()
    return {
        int(row["document_id"]): (row.get("text") or "")[:max_chars]
        for row in rows
    }


def _format_candidates(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            "{id} | {date} | {ref} | {title} | {snippet}".format(
                id=row.get("document_id"),
                date=row.get("issue_date") or "",
                ref=row.get("ref") or "",
                title=(row.get("title") or "").replace("\n", " ")[:180],
                snippet=(row.get("snippet") or "").replace("\n", " ")[:280],
            )
        )
    return "\n".join(lines)


def _llm_audit(
    question: str,
    existing_doc_ids: list[int],
    candidates: list[dict[str, Any]],
) -> tuple[list[list[int]], str, str]:
    client = OllamaClient(timeout=min(settings.ollama_timeout, 45))
    messages = [
        {"role": "system", "content": AUDIT_SYSTEM},
        {
            "role": "user",
            "content": AUDIT_USER.format(
                question=question,
                existing_doc_ids=existing_doc_ids,
                candidates=_format_candidates(candidates),
            ),
        },
    ]
    try:
        raw = client.chat_json(messages, temperature=0.0)
    except Exception as exc:
        return [], "uncertain", f"llm_error: {exc}"

    status = str(raw.get("status") or "ok").strip().lower()
    if status not in {"ok", "uncertain"}:
        status = "ok"
    reason = str(raw.get("reason") or "")
    sets = _normalize_gold_sets(raw.get("gold_sets"))
    return sets, status, reason


def _llm_verify_change(
    question: str,
    legacy_set: list[int],
    proposed_sets: list[list[int]],
    candidates: list[dict[str, Any]],
) -> tuple[str, str]:
    if not legacy_set or not proposed_sets:
        return "accept_proposed", ""
    client = OllamaClient(timeout=min(settings.ollama_timeout, 90))
    candidate_lines = []
    for row in candidates:
        candidate_lines.append(
            "{id} | {date} | {ref} | {title}".format(
                id=row.get("document_id"),
                date=row.get("issue_date") or "",
                ref=row.get("ref") or "",
                title=(row.get("title") or "").replace("\n", " ")[:200],
            )
        )
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM},
        {
            "role": "user",
            "content": VERIFY_USER.format(
                question=question,
                legacy=legacy_set,
                proposed=proposed_sets,
                candidates="\n".join(candidate_lines),
            ),
        },
    ]
    try:
        raw = client.chat_json(messages, temperature=0.0)
    except Exception as exc:
        return "keep_legacy", f"verify_error: {exc}"
    decision = str(raw.get("decision") or "keep_legacy").strip().lower()
    if decision not in {"keep_legacy", "accept_proposed", "both_valid"}:
        decision = "keep_legacy"
    reason = str(raw.get("reason") or "")
    return decision, reason


def _audit_entry(
    entry: dict[str, Any],
    max_candidates: int,
    max_snippet_chars: int,
) -> tuple[dict[str, Any], AuditResult]:
    question = str(entry.get("question") or "").strip()
    if not question:
        raise ValueError("missing question")

    existing_doc_ids = _legacy_doc_ids(entry)
    language = str(entry.get("language") or "").strip() or None
    retrieved_ids, _, query_embedding = _retrieve_candidate_ids(
        question,
        language=language,
        max_candidates=max_candidates,
    )

    ref_tokens = _extract_ref_tokens(question + " " + str(entry.get("answer") or ""))
    ref_ids = _find_docs_by_ref_tokens(ref_tokens)

    candidate_ids: list[int] = []
    seen: set[int] = set()
    for doc_id in existing_doc_ids + retrieved_ids + ref_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        candidate_ids.append(doc_id)

    candidates = _fetch_candidate_rows(candidate_ids[:max_candidates], query_embedding, max_snippet_chars=max_snippet_chars)
    candidate_id_set = {int(row["document_id"]) for row in candidates}

    if not candidates:
        fallback = [existing_doc_ids] if existing_doc_ids else []
        result = AuditResult(gold_sets=fallback, status="uncertain", reason="no_candidates", changed=False)
    else:
        sets, status, reason = _llm_audit(question, existing_doc_ids, candidates)
        filtered_sets: list[list[int]] = []
        for group in sets:
            pruned = sorted({int(doc_id) for doc_id in group if int(doc_id) in candidate_id_set})
            if pruned:
                filtered_sets.append(pruned)
        # Deduplicate after pruning.
        normalized = _normalize_gold_sets(filtered_sets)
        if not normalized:
            normalized = [existing_doc_ids] if existing_doc_ids else []
            status = "uncertain"
            reason = (reason + " | empty_after_prune").strip(" |")
        legacy_set = existing_doc_ids
        if legacy_set and normalized and normalized != [legacy_set]:
            decision, verify_reason = _llm_verify_change(question, legacy_set, normalized, candidates)
            if decision == "keep_legacy":
                normalized = [legacy_set]
                status = "uncertain"
                reason = (reason + " | verify_keep_legacy: " + verify_reason).strip(" |")
            elif decision == "both_valid":
                normalized = _normalize_gold_sets([legacy_set] + normalized)
                status = "uncertain"
                reason = (reason + " | verify_both_valid: " + verify_reason).strip(" |")
            else:
                reason = (reason + " | verify_accept_proposed: " + verify_reason).strip(" |")
        changed = normalized != ([legacy_set] if legacy_set else [])
        result = AuditResult(gold_sets=normalized, status=status, reason=reason, changed=changed)

    if existing_doc_ids and result.gold_sets and result.gold_sets != [existing_doc_ids]:
        candidate_by_id = {int(row["document_id"]): row for row in candidates}
        if (
            len(result.gold_sets) == 1
            and len(result.gold_sets[0]) == 1
            and _is_compound_question(question)
            and _needs_strict_detail(question)
        ):
            chosen_id = int(result.gold_sets[0][0])
            chosen_title = str((candidate_by_id.get(chosen_id) or {}).get("title") or "")
            if _looks_like_modifier_title(chosen_title):
                merged = sorted({*existing_doc_ids, chosen_id})
                result = AuditResult(
                    gold_sets=[merged],
                    status="uncertain",
                    reason=(result.reason + " | augment_with_legacy_for_modifier_doc").strip(" |"),
                    changed=merged != existing_doc_ids,
                )

    required = _required_signals(question)
    if required and result.gold_sets:
        primary = sorted(set(result.gold_sets[0]))
        candidate_id_list = [int(row["document_id"]) for row in candidates]
        text_map = _fetch_text_map(sorted(set(primary + candidate_id_list)))
        covered: set[str] = set()
        for doc_id in primary:
            covered |= _text_signals(text_map.get(doc_id, ""))
        missing = set(required) - covered
        if missing:
            extras: list[int] = []
            remaining = [doc_id for doc_id in candidate_id_list if doc_id not in primary]
            while missing and remaining:
                best_doc = None
                best_new: set[str] = set()
                for doc_id in remaining:
                    doc_new = _text_signals(text_map.get(doc_id, "")) & missing
                    if len(doc_new) > len(best_new):
                        best_doc = doc_id
                        best_new = doc_new
                if best_doc is None or not best_new:
                    break
                extras.append(best_doc)
                missing -= best_new
                remaining = [doc_id for doc_id in remaining if doc_id != best_doc]
            if extras:
                merged = sorted(set(primary + extras))
                result = AuditResult(
                    gold_sets=[merged],
                    status="uncertain",
                    reason=(
                        result.reason
                        + f" | augment_missing_signals required={sorted(required)} added={extras}"
                    ).strip(" |"),
                    changed=merged != existing_doc_ids,
                )

    union_ids = sorted({doc_id for group in result.gold_sets for doc_id in group})
    final_entry = {
        "id": int(entry.get("id")),
        "question": question,
        "gold_sets": result.gold_sets,
        "doc_ids": union_ids,
        "language": entry.get("language"),
        "doc_kind": entry.get("doc_kind"),
        "doc_subkind": entry.get("doc_subkind"),
    }
    return final_entry, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_set_curated_last12m_v2.json")
    parser.add_argument("--output", default="data/eval_set.json")
    parser.add_argument("--report", default="data/eval_audit_report.json")
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--max-snippet-chars", type=int, default=320)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list) or not raw:
        raise SystemExit("Input eval set is empty or invalid")

    source = raw[: args.limit] if args.limit and args.limit > 0 else raw

    final_entries: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    changed = 0
    uncertain = 0

    for index, entry in enumerate(source, start=1):
        final_entry, result = _audit_entry(
            entry,
            max_candidates=max(10, args.max_candidates),
            max_snippet_chars=max(120, args.max_snippet_chars),
        )
        final_entry["id"] = index
        final_entries.append(final_entry)

        legacy_doc_ids = _legacy_doc_ids(entry)
        row = {
            "id": int(entry.get("id") or index),
            "question": final_entry["question"],
            "legacy_doc_ids": legacy_doc_ids,
            "gold_sets": result.gold_sets,
            "status": result.status,
            "reason": result.reason,
            "changed": result.changed,
        }
        audit_rows.append(row)

        if result.changed:
            changed += 1
        if result.status != "ok":
            uncertain += 1

        if index % 5 == 0 or index == len(source):
            print(f"[progress] audited {index}/{len(source)}", flush=True)

    report = {
        "input": args.input,
        "output": args.output,
        "size": len(final_entries),
        "changed": changed,
        "uncertain": uncertain,
        "rows": audit_rows,
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(final_entries, fh, ensure_ascii=False, indent=2)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"Wrote audited eval set to {args.output}")
    print(f"Wrote audit report to {args.report}")
    print(f"Changed entries: {changed} / {len(final_entries)}")
    print(f"Uncertain entries: {uncertain} / {len(final_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
