"""Dump dense fact lines for given doc ids, to ground eval questions.

Usage: python eval_v2/facts.py <doc_id> [doc_id ...]
For each doc prints meta + the chunk_index/section/text of the chunks that
carry concrete facts (amounts, %, dates, deadlines, names, refs, quantities).
"""
from __future__ import annotations

import os
import re
import sys

import psycopg2
import psycopg2.extras

DSN = os.environ.get("DOGV_DB_DSN", "postgresql://dogv_ai:dogv_ai@localhost:5432/dogv_ai")

FACT = re.compile(
    r"(€|euros|\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\b|\b\d+,\d+\b|\d{1,2}\s*%|"
    r"plazo|termini|presentaci|benefici|cuant|quant|import|dotaci|nombr|nomen|"
    r"cese|cessa|requisit|requisito|adjud|concede|concedei|deneg|venciment|"
    r"hasta el|fins el|fins al|días|dies|mes\b|meses|mesos)",
    re.I,
)


def main() -> int:
    ids = [int(x) for x in sys.argv[1:]]
    with psycopg2.connect(DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for did in ids:
                cur.execute(
                    """SELECT dd.id, dd.ref, di.language, di.date, dd.doc_kind, dd.doc_subkind,
                              dd.conselleria, dd.section, dd.text_source, dd.title
                       FROM dogv_documents dd JOIN dogv_issues di ON di.id=dd.issue_id
                       WHERE dd.id=%s""",
                    (did,),
                )
                m = cur.fetchone()
                if not m:
                    print(f"### doc {did}: NOT FOUND")
                    continue
                print(f"\n### doc_id={m['id']} ref={m['ref']} lang={m['language']} date={m['date']} "
                      f"kind={m['doc_kind']}/{m['doc_subkind']} src={m['text_source']}")
                print(f"    conselleria={m['conselleria']}")
                print(f"    TITLE: {m['title']}")
                cur.execute(
                    "SELECT chunk_index, text FROM rag_chunk WHERE document_id=%s ORDER BY chunk_index",
                    (did,),
                )
                rows = cur.fetchall()
                shown = 0
                for r in rows:
                    txt = " ".join((r["text"] or "").split())
                    if not FACT.search(txt):
                        continue
                    print(f"  [chunk {r['chunk_index']}] {txt[:550]}")
                    shown += 1
                    if shown >= 6:
                        break
                if shown == 0 and rows:
                    print(f"  [chunk {rows[0]['chunk_index']}] {' '.join((rows[0]['text'] or '').split())[:550]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
