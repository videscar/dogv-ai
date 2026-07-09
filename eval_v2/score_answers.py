"""Aggregate answer-quality metrics from human/Claude judgments + collected answers.

Judgments file (judgments.jsonl), one object per id:
  {"id","correctness":0|1|2,"faithful":0|1,"critical_error":true|false,
   "abstained":true|false,"note":"..."}
  - correctness: 2=fully correct, 1=partial/incomplete, 0=wrong or missing (answerable only)
  - faithful: 1 if every claim is supported by retrieved evidence/citations, else 0
  - critical_error: material factual error (wrong number/date/ref/name) -> HARD GATE to 0
  - abstained: for out_of_scope items, did it correctly decline

Per-question gated score:
  out_of_scope -> 1.0 if abstained else 0.0
  answerable   -> 0.0 if critical_error else correctness/2

Usage: python eval_v2/score_answers.py judgments.jsonl answers_raw.jsonl [out.json]
"""

from __future__ import annotations

import json
from collections import defaultdict


def load_jsonl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate answer-quality metrics.")
    ap.add_argument("judgments")
    ap.add_argument("answers")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument(
        "--holdout",
        default=None,
        help="holdout_ids.json — also report the tuned slice vs the frozen holdout separately",
    )
    args = ap.parse_args()

    judg = {j["id"]: j for j in load_jsonl(args.judgments)}
    raw = {r["id"]: r for r in load_jsonl(args.answers)}
    out_path = args.out
    holdout_ids: set[str] = set()
    if args.holdout:
        holdout_ids = set(json.load(open(args.holdout, encoding="utf-8")).get("holdout_ids") or [])

    rows = []
    for qid, r in raw.items():
        j = judg.get(qid)
        if not j:
            print(f"WARN no judgment for {qid}")
            continue
        oos = bool(r["should_abstain"])
        if oos:
            gated = 1.0 if j.get("abstained") else 0.0
        else:
            gated = 0.0 if j.get("critical_error") else (int(j.get("correctness", 0)) / 2.0)
        rows.append(
            {
                "id": qid,
                "category": r["category"],
                "language": r["language"],
                "out_of_scope": oos,
                "correctness": j.get("correctness"),
                "faithful": j.get("faithful"),
                "critical_error": bool(j.get("critical_error")),
                "abstained": j.get("abstained"),
                "gated": gated,
                "n_citations": r.get("n_citations"),
            }
        )

    def agg(items):
        ans = [x for x in items if not x["out_of_scope"]]
        oos = [x for x in items if x["out_of_scope"]]
        d = {"n": len(items), "n_answerable": len(ans), "n_oos": len(oos)}
        if ans:
            d["correctness_avg"] = round(sum(x["correctness"] for x in ans) / (2 * len(ans)), 3)
            d["faithfulness_rate"] = round(sum(x["faithful"] for x in ans) / len(ans), 3)
            d["critical_error_rate"] = round(sum(x["critical_error"] for x in ans) / len(ans), 3)
            d["gated_score"] = round(sum(x["gated"] for x in ans) / len(ans), 3)
        if oos:
            d["abstention_rate"] = round(sum(1 for x in oos if x["abstained"]) / len(oos), 3)
        return d

    groups = defaultdict(list)
    for x in rows:
        groups["ALL"].append(x)
        groups[f"cat:{x['category']}"].append(x)
        groups[f"lang:{x['language']}"].append(x)
    summary = {g: agg(v) for g, v in groups.items()}

    def line(d):
        s = (
            f"n={d['n']:3d} corr={d.get('correctness_avg','-')} faith={d.get('faithfulness_rate','-')} "
            f"crit_err={d.get('critical_error_rate','-')} gated={d.get('gated_score','-')}"
        )
        if "abstention_rate" in d:
            s += f" abstain={d['abstention_rate']}"
        return s

    print("\n=== ANSWER QUALITY ===")
    print("ALL:", line(summary["ALL"]))
    print("\n-- by category --")
    for g in sorted(k for k in summary if k.startswith("cat:")):
        print(f"  {g[4:]:13s}: {line(summary[g])}")
    print("\n-- by language --")
    for g in sorted(k for k in summary if k.startswith("lang:")):
        print(f"  {g[5:]:13s}: {line(summary[g])}")

    split_summary = None
    if holdout_ids:
        tuned_rows = [x for x in rows if x["id"] not in holdout_ids]
        hold_rows = [x for x in rows if x["id"] in holdout_ids]
        split_summary = {"tuned": agg(tuned_rows), "holdout": agg(hold_rows)}
        print("\n-- tuned vs frozen holdout --")
        print(f"  tuned   : {line(split_summary['tuned'])}")
        print(f"  holdout : {line(split_summary['holdout'])}")

    if out_path:
        if split_summary:
            summary = {
                **summary,
                "split:tuned": split_summary["tuned"],
                "split:holdout": split_summary["holdout"],
            }
        json.dump(
            {"summary": summary, "per_query": rows},
            open(out_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=1,
        )
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
