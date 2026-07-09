"""Surface fact-rich DOGV documents for grounding eval questions.

Usage: python eval_v2/mine.py <preset> [limit]
Presets pull a stratified sample with full cleaned text so questions can be
authored against real amounts / dates / refs / names, with known gold doc ids.
"""

from __future__ import annotations

import json
import os
import re
import sys

import psycopg2
import psycopg2.extras

DSN = os.environ.get("DOGV_DB_DSN", "postgresql://dogv_ai:dogv_ai@localhost:5432/dogv_ai")

EURO = re.compile(r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?\s*(?:€|euros)", re.I)
DATE = re.compile(r"\b\d{1,2} d[e']\w+ (?:de )?\d{4}\b", re.I)
PCT = re.compile(r"\b\d{1,2}(?:,\d+)?\s*%")


def richness(text: str) -> int:
    if not text:
        return 0
    return len(EURO.findall(text)) * 3 + len(PCT.findall(text)) + len(DATE.findall(text))


def fetch(where: str, params: dict, limit: int) -> list[dict]:
    sql = f"""
        SELECT dd.id, dd.ref, di.language, di.date AS issue_date, dd.doc_kind,
               dd.doc_subkind, dd.conselleria, dd.section, dd.text_source,
               dd.title, dd.text,
               (SELECT count(*) FROM rag_chunk rc WHERE rc.document_id = dd.id) AS n_chunks
        FROM dogv_documents dd
        JOIN dogv_issues di ON di.id = dd.issue_id
        WHERE {where}
        ORDER BY di.date DESC
        LIMIT {limit}
    """
    with psycopg2.connect(DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["issue_date"] = r["issue_date"].isoformat() if r["issue_date"] else None
        r["richness"] = richness(r.get("text") or "")
        r["text"] = (r.get("text") or "")[:4000]
    return rows


PRESETS = {
    # kind, language, text_source filter, extra where
    "ayudas_es": ("doc_kind='Ayudas' AND di.language='es_es'", 400),
    "ayudas_va": ("doc_kind='Ayudas' AND di.language='va_va'", 400),
    "subv_es": ("doc_kind='Subvenciones' AND di.language='es_es'", 400),
    "subv_va": ("doc_kind='Subvenciones' AND di.language='va_va'", 400),
    "becas_es": ("doc_kind='Becas' AND di.language='es_es'", 300),
    "becas_va": ("doc_kind='Becas' AND di.language='va_va'", 300),
    "premios_es": ("doc_kind='Premios' AND di.language='es_es'", 300),
    "premios_va": ("doc_kind='Premios' AND di.language='va_va'", 300),
    "empleo_es": ("doc_kind='Empleo Publico' AND di.language='es_es'", 400),
    "empleo_va": ("doc_kind='Empleo Publico' AND di.language='va_va'", 400),
    "annex_es": (
        "text_source='pdf' AND di.language='es_es' AND doc_kind IN ('Subvenciones','Ayudas','Becas','Premios')",
        400,
    ),
    "annex_va": (
        "text_source='pdf' AND di.language='va_va' AND doc_kind IN ('Subvenciones','Ayudas','Becas','Premios')",
        400,
    ),
    "annex_empleo_es": (
        "text_source='pdf' AND di.language='es_es' AND doc_kind='Empleo Publico'",
        300,
    ),
    "nombr_es": (
        "doc_kind='Otros' AND di.language='es_es' AND (dd.title ILIKE '%nombr%' OR dd.title ILIKE '%cesa%' OR dd.title ILIKE '%cese%')",
        300,
    ),
    "nombr_va": (
        "doc_kind='Otros' AND di.language='va_va' AND (dd.title ILIKE '%nomen%' OR dd.title ILIKE '%cess%')",
        300,
    ),
}


def main() -> int:
    preset = sys.argv[1]
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    where, limit = PRESETS[preset]
    rows = fetch(where, {}, limit)
    rows.sort(key=lambda r: r["richness"], reverse=True)
    rows = rows[:topn]
    out = f"data/eval_v2/cand_{preset}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"{preset}: wrote {len(rows)} rows -> {out}")
    for r in rows[:topn]:
        print(
            f"  id={r['id']} ref={r['ref']} rich={r['richness']} src={r['text_source']} chunks={r['n_chunks']} | {(r['title'] or '')[:80]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
