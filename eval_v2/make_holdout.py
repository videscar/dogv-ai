"""Deterministically carve a frozen holdout out of the v2 suite.

The split is **content-blind**: an id is in the holdout iff sha1(id) mod 4 == 0.
This can't be cherry-picked to flatter (it ignores category, difficulty and the
answer), it is stable across runs, and it re-derives byte-for-byte from the ids
alone — so the committed data/eval_v2/holdout_ids.json is auditable.

Honesty note: the shipped fixes were historically validated on the *full* 100, so
this holdout is "frozen going forward" rather than never-seen; treat the tuned-vs-
holdout gap as a lower bound. From the freeze date these ids are excluded from
tuning. See docs/EVALS.md.

Usage:
    python eval_v2/make_holdout.py            # verify committed file matches the rule
    python eval_v2/make_holdout.py --write    # (re)write data/eval_v2/holdout_ids.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

INPUT = "data/eval_v2/answer_input.json"
OUT = "data/eval_v2/holdout_ids.json"
FREEZE_DATE = "2026-07-09"


def is_holdout(qid: str) -> bool:
    return int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 4 == 0


def compute() -> dict:
    items = json.load(open(INPUT, encoding="utf-8"))
    holdout = sorted(d["id"] for d in items if is_holdout(d["id"]))
    return {
        "rule": "sha1(id) % 4 == 0 (content-blind)",
        "freeze_date": FREEZE_DATE,
        "source": INPUT,
        "n_total": len(items),
        "n_holdout": len(holdout),
        "holdout_ids": holdout,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite the committed holdout file")
    args = ap.parse_args()

    computed = compute()
    if args.write:
        json.dump(computed, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"wrote {OUT}: {computed['n_holdout']}/{computed['n_total']} ids")
        return 0

    try:
        committed = json.load(open(OUT, encoding="utf-8"))
    except FileNotFoundError:
        print(f"{OUT} missing — run with --write")
        return 1
    if committed.get("holdout_ids") != computed["holdout_ids"]:
        print("MISMATCH: committed holdout_ids != rule output. Re-run with --write.")
        return 1
    print(f"OK: {OUT} matches the rule ({computed['n_holdout']}/{computed['n_total']} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
