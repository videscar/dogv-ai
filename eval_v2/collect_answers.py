"""Collect production /ask answers (+ retrieved evidence) for the v2 suite.

Resumable: appends one JSON line per item to answers_raw.jsonl and skips ids
already present. Captures answer, citations, the evidence quotes the synthesis
LLM saw (debug.evidence), candidate doc ids, and answer-stage diagnostics — the
inputs needed to judge correctness AND faithfulness separately.

Usage: python eval_v2/collect_answers.py [--base-url http://127.0.0.1:8088]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import httpx

IN = "data/eval_v2/answer_input.json"
OUT = "data/eval_v2/answers_raw.jsonl"  # overridable via --out


def done_ids(path: str) -> set[str]:
    ids = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8088")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--rescue-timeout", type=float, default=900.0)
    args = ap.parse_args()

    out_path = args.out
    items = json.load(open(IN, encoding="utf-8"))
    skip = done_ids(out_path)
    todo = [i for i in items if i["id"] not in skip]
    print(f"total={len(items)} done={len(skip)} todo={len(todo)}", flush=True)

    fout = open(out_path, "a", encoding="utf-8")
    base = args.base_url.rstrip("/")
    for idx, it in enumerate(todo, start=1):
        q = it["question"]
        started = time.perf_counter()
        answer, citations, evidence, cand, profile, error = "", [], [], [], {}, None
        payload = {"question": q, "debug": True}
        for attempt, tmo in enumerate([args.timeout, args.rescue_timeout]):
            try:
                with httpx.Client(base_url=base, timeout=tmo) as cl:
                    r = cl.post("/ask", json=payload)
                r.raise_for_status()
                d = r.json()
                answer = str(d.get("answer") or "")
                citations = d.get("citations") or []
                dbg = d.get("debug") or {}
                evidence = dbg.get("evidence") or []
                cand = dbg.get("candidate_docs") or []
                profile = (dbg.get("profile") or {}).get("answer") or {}
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 0:
                    continue
        latency = round(time.perf_counter() - started, 1)
        rec = {
            "id": it["id"], "category": it["category"], "language": it["language"],
            "should_abstain": it["should_abstain"], "question": q,
            "expected_answer": it["expected_answer"],
            "gold_refs": it.get("gold_refs", []), "gold_doc_ids": it.get("doc_ids", []),
            "answer": answer,
            "cited_doc_ids": [c.get("document_id") for c in citations],
            "cited_refs": [c.get("ref") for c in citations],
            "n_citations": len(citations),
            "evidence": [{"doc_id": e.get("doc_id"), "quote": (e.get("quote") or "")[:500]}
                         for e in evidence][:10],
            "cand_doc_ids": [c.get("document_id") for c in cand][:15],
            "fallback_reason": profile.get("fallback_reason"),
            "latency_s": latency, "error": error,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{idx}/{len(todo)}] {it['id']} cites={rec['n_citations']} "
              f"lat={latency}s{' ERR' if error else ''}", flush=True)
    fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
