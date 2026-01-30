from __future__ import annotations

import argparse
import json
import random


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_set_v1.json")
    parser.add_argument("--output", default="data/eval_audit_sample.json")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20250122)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    if not entries:
        raise SystemExit("Eval set is empty")

    count = min(args.count, len(entries))
    rng = random.Random(args.seed)
    sampled = rng.sample(entries, count)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(sampled, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(sampled)} audit samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
