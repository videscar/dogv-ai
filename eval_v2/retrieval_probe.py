"""WS0 retrieval measurement harness — fast, deterministic, no synthesis.

For each eval question, runs the pipeline through read_docs (the LLM synthesis is
NOT run; extract_evidence is stubbed to capture the read payload) and reports
per-stage pipeline recall, so we can see WHERE the gold answer is lost:

  pool   = gold doc(s) entered the candidate pool (retrieve+rerank)
  read   = gold doc(s) survived into the reader payload (doc/chunk selection)

Both "any" (≥1 gold doc) and "full" (ALL gold docs — matters for multihop) are
reported, with per-category and per-language breakdowns.

Usage:
  python eval_v2/retrieval_probe.py [--ids v2-001,v2-002] [--set data/eval_v2/eval_set_v2.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import agent.nodes.read as readmod
from agent.nodes import (
    analyze_intent_node,
    temporal_guard_node,
    online_ingest_node,
    retrieve_candidates_node,
    rerank_titles_node,
    read_docs_node,
)

DEFAULT_SET = "data/eval_v2/eval_set_v2.jsonl"


def _probe_one(question: str) -> tuple[list[int], list[int]]:
    """Return (candidate_pool_doc_ids, read_payload_doc_ids) for a question."""
    cap: dict[str, list] = {}

    def _stub(_q, docs, full_docs=None):
        cap["docs"] = docs
        return []

    orig = readmod.extract_evidence
    readmod.extract_evidence = _stub
    try:
        st = {"question": question, "request_id": "probe", "debug": True}
        for node in (
            analyze_intent_node,
            temporal_guard_node,
            online_ingest_node,
            retrieve_candidates_node,
            rerank_titles_node,
        ):
            out = node(st)
            if isinstance(out, dict):
                st.update(out)
        read_docs_node(st)
    finally:
        readmod.extract_evidence = orig

    cand = [c.get("document_id") for c in (st.get("candidate_docs") or [])]
    read = [d.get("document_id") for d in (cap.get("docs") or [])]
    return [c for c in cand if c is not None], [r for r in read if r is not None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.set, encoding="utf-8") if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in want]

    agg = defaultdict(lambda: {"n": 0, "pool_any": 0, "pool_full": 0, "read_any": 0, "read_full": 0})
    per_q = []
    for r in rows:
        if r.get("should_abstain"):
            continue  # OOS has no gold to retrieve
        gold = set(int(d) for d in (r.get("doc_ids") or []))
        if not gold:
            continue
        cand, read = _probe_one(r["question"])
        cand_s, read_s = set(cand), set(read)
        pa = bool(gold & cand_s); pf = gold <= cand_s
        ra = bool(gold & read_s); rf = gold <= read_s
        for key in ("ALL", f"cat:{r['category']}", f"lang:{r['language']}"):
            a = agg[key]
            a["n"] += 1
            a["pool_any"] += pa; a["pool_full"] += pf
            a["read_any"] += ra; a["read_full"] += rf
        per_q.append({"id": r["id"], "category": r["category"], "language": r["language"],
                      "gold": sorted(gold), "pool_any": pa, "pool_full": pf,
                      "read_any": ra, "read_full": rf})
        miss = "" if pf else ("  <-- POOL MISS" if not pa else "  <-- partial/read")
        print(f"[{r['id']}] {r['category']:10} pool={'F' if pf else ('A' if pa else '-')} "
              f"read={'F' if rf else ('A' if ra else '-')}{miss}", flush=True)

    def line(key):
        a = agg[key]; n = a["n"] or 1
        return (f"{key:16} n={a['n']:3}  pool_any={a['pool_any']/n:.2f} pool_full={a['pool_full']/n:.2f}"
                f"  read_any={a['read_any']/n:.2f} read_full={a['read_full']/n:.2f}")
    print("\n=== PIPELINE RECALL (answerable) ===")
    print(line("ALL"))
    for k in sorted(k for k in agg if k.startswith("cat:")):
        print(" ", line(k))
    for k in sorted(k for k in agg if k.startswith("lang:")):
        print(" ", line(k))

    if args.out:
        json.dump({"per_q": per_q, "agg": {k: dict(v) for k, v in agg.items()}},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
