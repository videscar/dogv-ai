from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
from typing import Iterable

from sqlalchemy import text as sa_text

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.taxonomy import canonical_doc_kind, canonical_doc_subkind

MIN_QUESTION_CHARS = 24
MAX_TITLE_CHARS = 240
MAX_REF_CHARS = 80
DATE_RECENT_DAYS = 180
DATE_MID_DAYS = 540


@dataclass(frozen=True)
class EvalDoc:
    doc_id: int
    title: str
    ref: str | None
    issue_date: date | None
    doc_kind: str | None
    doc_subkind: str | None
    language: str | None


class GroupCursor:
    def __init__(
        self,
        docs: list[EvalDoc],
        recent_start: date,
        mid_start: date,
        bin_order: Iterable[str],
    ) -> None:
        self.bin_order = list(bin_order)
        self.bins = {name: [] for name in self.bin_order}
        for doc in docs:
            bucket = _date_bucket(doc.issue_date, recent_start, mid_start)
            self.bins.setdefault(bucket, []).append(doc)
        for bucket, items in self.bins.items():
            items.sort(key=lambda d: (_date_sort_key(d.issue_date), d.doc_id), reverse=True)
            self.bins[bucket] = items
        self.bin_index = 0
        self.offsets = {name: 0 for name in self.bins}

    def next_doc(self) -> EvalDoc | None:
        if not self.bins:
            return None
        for _ in range(len(self.bin_order)):
            bucket = self.bin_order[self.bin_index]
            items = self.bins.get(bucket, [])
            offset = self.offsets.get(bucket, 0)
            if offset < len(items):
                doc = items[offset]
                self.offsets[bucket] = offset + 1
                self.bin_index = (self.bin_index + 1) % len(self.bin_order)
                return doc
            self.bin_index = (self.bin_index + 1) % len(self.bin_order)
        return None


def _date_sort_key(value: date | None) -> date:
    return value or date.min


def _date_bucket(value: date | None, recent_start: date, mid_start: date) -> str:
    if not value:
        return "unknown"
    if value >= recent_start:
        return "recent"
    if value >= mid_start:
        return "mid"
    return "older"


def _trim(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip()


def _templates(language: str) -> list[str]:
    if language.startswith(("va", "ca")):
        return [
            "¿On puc consultar el {kind} \"{title}\"{date}?",
            "¿S'ha publicat \"{title}\"{ref} al DOGV{date}?",
            "¿Quin és el {kind} titulat \"{title}\"{ref}{date}?",
        ]
    return [
        "¿Dónde puedo consultar el {kind} \"{title}\"{date}?",
        "¿Se publicó \"{title}\"{ref} en el DOGV{date}?",
        "¿Cuál es el {kind} titulado \"{title}\"{ref}{date}?",
    ]


def _build_question(doc: EvalDoc) -> str:
    title = _trim(doc.title, MAX_TITLE_CHARS)
    ref = _trim(doc.ref, MAX_REF_CHARS) if doc.ref else ""
    kind_bits = " ".join(part for part in (doc.doc_kind, doc.doc_subkind) if part)
    kind = kind_bits or "documento"
    date_clause = f" del {doc.issue_date.isoformat()}" if doc.issue_date else ""
    ref_clause = f" (ref {ref})" if ref else ""

    templates = _templates(doc.language or "es_es")
    template = templates[doc.doc_id % len(templates)]
    return template.format(kind=kind, title=title, ref=ref_clause, date=date_clause)


def _fetch_docs() -> list[EvalDoc]:
    sql = sa_text(
        """
        SELECT
            dd.id AS doc_id,
            dd.title AS title,
            dd.ref AS ref,
            dd.doc_kind AS doc_kind,
            dd.doc_subkind AS doc_subkind,
            di.date AS issue_date,
            di.language AS language
        FROM dogv_documents dd
        JOIN dogv_issues di ON di.id = dd.issue_id
        WHERE dd.title IS NOT NULL
        ORDER BY di.date DESC, dd.id ASC
        """
    )
    with SessionLocal() as db:
        rows = db.execute(sql).mappings().all()
    docs = []
    for row in rows:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        docs.append(
            EvalDoc(
                doc_id=int(row["doc_id"]),
                title=title,
                ref=row.get("ref"),
                issue_date=row.get("issue_date"),
                doc_kind=row.get("doc_kind"),
                doc_subkind=row.get("doc_subkind"),
                language=row.get("language"),
            )
        )
    return docs


def _group_docs(docs: list[EvalDoc]) -> dict[tuple[str, str], list[EvalDoc]]:
    grouped: dict[tuple[str, str], list[EvalDoc]] = {}
    for doc in docs:
        kind = canonical_doc_kind(doc.doc_kind) or "Otros"
        subkind = canonical_doc_subkind(doc.doc_subkind) or "SinSubtipo"
        key = (kind, subkind)
        grouped.setdefault(key, []).append(doc)
    return grouped


def _build_entries(docs: list[EvalDoc], target_size: int) -> list[dict[str, object]]:
    if not docs:
        raise SystemExit("No documents available to build eval set")

    max_date = max((doc.issue_date for doc in docs if doc.issue_date), default=None)
    if not max_date:
        raise SystemExit("No issue dates available to build eval set")

    recent_start = max_date - timedelta(days=DATE_RECENT_DAYS)
    mid_start = max_date - timedelta(days=DATE_MID_DAYS)

    grouped = _group_docs(docs)
    group_keys = sorted(grouped.keys(), key=lambda item: (item[0], item[1]))

    cursors = {
        key: GroupCursor(grouped[key], recent_start, mid_start, ["recent", "mid", "older", "unknown"])
        for key in group_keys
    }

    entries: list[dict[str, object]] = []
    used_ids: set[int] = set()
    used_questions: set[str] = set()

    while len(entries) < target_size:
        progress = False
        for key in group_keys:
            if len(entries) >= target_size:
                break
            cursor = cursors[key]
            attempts = 0
            while attempts < 5:
                doc = cursor.next_doc()
                attempts += 1
                if not doc:
                    break
                if doc.doc_id in used_ids:
                    continue
                question = _build_question(doc)
                if not question or len(question) < MIN_QUESTION_CHARS:
                    continue
                if question in used_questions:
                    continue
                entry = {
                    "id": len(entries) + 1,
                    "question": question,
                    "doc_id": doc.doc_id,
                    "ref": doc.ref,
                    "issue_date": doc.issue_date.isoformat() if doc.issue_date else None,
                    "doc_kind": doc.doc_kind,
                    "doc_subkind": doc.doc_subkind,
                    "title": doc.title,
                    "language": doc.language,
                }
                entries.append(entry)
                used_ids.add(doc.doc_id)
                used_questions.add(question)
                progress = True
                break
        if not progress:
            break

    if len(entries) < target_size:
        raise SystemExit(f"Expected {target_size} entries, got {len(entries)}")

    return entries


def _validate_entries(entries: list[dict[str, object]]) -> None:
    seen_questions: set[str] = set()
    seen_doc_ids: set[int] = set()
    for entry in entries:
        question = str(entry.get("question") or "").strip()
        if not question:
            raise SystemExit(f"Empty question for id {entry.get('id')}")
        if len(question) < MIN_QUESTION_CHARS:
            raise SystemExit(f"Question too short for id {entry.get('id')}")
        if question in seen_questions:
            raise SystemExit(f"Duplicate question for id {entry.get('id')}")
        seen_questions.add(question)

        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, int):
            raise SystemExit(f"Invalid doc_id for id {entry.get('id')}")
        if doc_id in seen_doc_ids:
            raise SystemExit(f"Duplicate doc_id {doc_id}")
        seen_doc_ids.add(doc_id)

        language = str(entry.get("language") or "").strip()
        if not language:
            raise SystemExit(f"Missing language for doc_id {doc_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/eval_set_v1.json")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    docs = _fetch_docs()
    entries = _build_entries(docs, args.size)
    _validate_entries(entries)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
