"""Regression diff for the on-demand-fetch change.

Compares a fresh /ask collection (new code) against a baseline collection
(prior prod) on the v2 100Q answer suite, bucketed by whether the question is an
explicit reference query (the only bucket the change can affect). Normal-bucket
citation/abstention changes are flagged as potential regressions; reference-bucket
changes are expected (single-citation collapse, pre-window fetches).

Usage: python eval_v2/regress_diff.py NEW.jsonl BASELINE.jsonl
"""

import json
import sys

sys.path.insert(0, ".")
from api.query_classifiers import is_reference_query  # noqa: E402


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    return out


def cset(d):
    return set(d.get("cited_doc_ids") or [])


def abstained(d):
    # heuristic: no citations OR an explicit abstain fallback
    return (d.get("n_citations") or 0) == 0 or bool(d.get("fallback_reason"))


def main():
    new = load(sys.argv[1])
    base = load(sys.argv[2])
    common = [q for q in base if q in new]
    norm_changed, ref_changed, missing = [], [], []
    norm_total = ref_total = 0
    for qid in common:
        b, n = base[qid], new[qid]
        q = b.get("question") or n.get("question") or ""
        is_ref = is_reference_query(q)
        if is_ref:
            ref_total += 1
        else:
            norm_total += 1
        cited_changed = cset(b) != cset(n)
        abst_changed = abstained(b) != abstained(n)
        if cited_changed or abst_changed:
            rec = {
                "id": qid,
                "ref": is_ref,
                "q": q[:70],
                "base_cites": sorted(cset(b)),
                "new_cites": sorted(cset(n)),
                "base_n": b.get("n_citations"),
                "new_n": n.get("n_citations"),
                "base_abst": abstained(b),
                "new_abst": abstained(n),
            }
            (ref_changed if is_ref else norm_changed).append(rec)
    for qid in base:
        if qid not in new:
            missing.append(qid)

    print(f"compared {len(common)} questions  (normal={norm_total} reference={ref_total})")
    if missing:
        print(f"!! {len(missing)} baseline questions missing from new run: {missing}")
    print(
        f"\n=== NORMAL bucket: {len(norm_changed)} changed (EXPECTED 0 — any is a regression) ==="
    )
    for r in norm_changed:
        print(f"  [{r['id']}] {r['q']}")
        print(
            f"      cites {r['base_cites']} -> {r['new_cites']}  | abstain {r['base_abst']}->{r['new_abst']}"
        )
    print(
        f"\n=== REFERENCE bucket: {len(ref_changed)} changed (expected: collapse / new fetches) ==="
    )
    for r in ref_changed:
        print(f"  [{r['id']}] {r['q']}")
        print(
            f"      cites n {r['base_n']}->{r['new_n']} {r['base_cites']} -> {r['new_cites']}  | abstain {r['base_abst']}->{r['new_abst']}"
        )

    print("\n=== VERDICT ===")
    print(
        "CLEAN"
        if not norm_changed and not missing
        else "REVIEW NEEDED (normal-bucket changes above)"
    )


if __name__ == "__main__":
    main()
