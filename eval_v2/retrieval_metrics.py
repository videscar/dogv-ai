"""Compute recall@k, MRR and nDCG@k from a run_eval.py report, with per-category
and per-language breakdowns. Primary stage = 'rerank' (the production final order).

Usage: python eval_v2/retrieval_metrics.py <run_eval_report.json> <retrieval_input.json> [out.json]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

K_VALUES = [1, 5, 10, 20]
PRIMARY_STAGE = "rerank"


def gold_sets_of(q) -> list[list[int]]:
    return [list(map(int, g)) for g in q.get("gold_sets", []) if g]


def ranked_ids(q, stage) -> list[int]:
    return [int(c["document_id"]) for c in q["candidates"][stage]]


def recall_at_k(ranked, gold_sets, k) -> int:
    topk = set(ranked[:k])
    return 1 if any(set(g).issubset(topk) for g in gold_sets) else 0


def satisfying_rank(ranked, gold_sets) -> int | None:
    """Smallest rank (1-based) at which some gold set is fully present."""
    best = None
    pos = {d: i + 1 for i, d in enumerate(ranked)}
    for g in gold_sets:
        if all(d in pos for d in g):
            r = max(pos[d] for d in g)  # need all docs of the set present
            best = r if best is None else min(best, r)
    return best


def ndcg_at_k(ranked, gold_sets, k) -> float:
    """Binary nDCG over the union of gold docs (best gold set by completeness)."""
    # relevant = union of docs across gold sets (any acceptable doc is relevant)
    relevant = set()
    for g in gold_sets:
        relevant |= set(g)
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    # ideal: as many relevant as could appear, capped by k and by |relevant|
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> int:
    report = json.load(open(sys.argv[1], encoding="utf-8"))
    inp = json.load(open(sys.argv[2], encoding="utf-8"))
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    meta = {i["id"]: i for i in inp}
    queries = report["queries"]

    # accumulators: group -> stage -> metric -> [values]
    def new_acc():
        return defaultdict(lambda: defaultdict(list))

    groups = {"ALL": new_acc()}

    per_query = []
    for q in queries:
        qid = q["id"]
        m = meta.get(qid, {})
        cat = m.get("category", "?")
        lang = m.get("language", q.get("language", "?"))
        src = m.get("text_source", "?")
        gs = gold_sets_of(q)
        bucket_keys = ["ALL", f"cat:{cat}", f"lang:{lang}", f"src:{src}"]
        for bk in bucket_keys:
            groups.setdefault(bk, new_acc())

        rec = {"id": qid, "category": cat, "language": lang, "text_source": src,
               "gold_sets": gs}
        for stage in ("hybrid", PRIMARY_STAGE):
            ranked = ranked_ids(q, stage)
            sr = satisfying_rank(ranked, gs)
            mrr = (1.0 / sr) if sr else 0.0
            vals = {"mrr": mrr, "rank": sr}
            for k in K_VALUES:
                vals[f"recall@{k}"] = recall_at_k(ranked, gs, k)
                vals[f"ndcg@{k}"] = ndcg_at_k(ranked, gs, k)
            rec[stage] = vals
            for bk in bucket_keys:
                acc = groups[bk][stage]
                acc["mrr"].append(mrr)
                for k in K_VALUES:
                    acc[f"recall@{k}"].append(vals[f"recall@{k}"])
                    acc[f"ndcg@{k}"].append(vals[f"ndcg@{k}"])
        per_query.append(rec)

    def summarize(acc):
        return {st: {mk: round(sum(v) / len(v), 4) if v else 0.0 for mk, v in md.items()}
                for st, md in acc.items()}

    summary = {bk: {"n": len(g[PRIMARY_STAGE]["mrr"]), **summarize(g)} for bk, g in groups.items()}

    # pretty print
    def fmt(d):
        return (f"R@1={d['recall@1']:.3f} R@5={d['recall@5']:.3f} R@10={d['recall@10']:.3f} "
                f"R@20={d['recall@20']:.3f} MRR={d['mrr']:.3f} "
                f"nDCG@5={d['ndcg@5']:.3f} nDCG@10={d['ndcg@10']:.3f}")

    print(f"\n=== RETRIEVAL METRICS (stage={PRIMARY_STAGE}, primary production k=10) ===")
    print(f"ALL (n={summary['ALL']['n']}): {fmt(summary['ALL'][PRIMARY_STAGE])}")
    print(f"  [hybrid pre-rerank]: {fmt(summary['ALL']['hybrid'])}")
    print("\n-- by category --")
    for bk in sorted(k for k in summary if k.startswith("cat:")):
        print(f"  {bk[4:]:13s} (n={summary[bk]['n']:2d}): {fmt(summary[bk][PRIMARY_STAGE])}")
    print("\n-- by language --")
    for bk in sorted(k for k in summary if k.startswith("lang:")):
        print(f"  {bk[5:]:13s} (n={summary[bk]['n']:2d}): {fmt(summary[bk][PRIMARY_STAGE])}")
    print("\n-- by text_source (html vs pdf/annex) --")
    for bk in sorted(k for k in summary if k.startswith("src:")):
        print(f"  {bk[4:]:13s} (n={summary[bk]['n']:2d}): {fmt(summary[bk][PRIMARY_STAGE])}")

    if out_path:
        json.dump({"summary": summary, "per_query": per_query},
                  open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
