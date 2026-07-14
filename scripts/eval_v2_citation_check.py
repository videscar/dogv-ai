#!/usr/bin/env python3
"""Citation + abstention regression check for the eval_v2 answer set.

A fast, deterministic regression signal over data/eval_v2/eval_set_v2.jsonl run
against a live /ask API (default prod :8088). NOT the LLM-judge answer-quality
score (that is a separate, heavier, non-deterministic run) — this measures the
two things a retrieval-layer change can break:

  * non-abstain cases  -> GOLD-CITED: does a gold doc_id appear in the citations?
  * should_abstain cases -> ABSTAINED: does the answer decline (no consta / ...)?

    python scripts/eval_v2_citation_check.py
    python scripts/eval_v2_citation_check.py --api http://127.0.0.1:8088
    python scripts/eval_v2_citation_check.py --input data/eval_v2/eval_set_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

_ABSTAIN = re.compile(r"no consta|no puedo confirmar|no hay publicaciones|no se encontr", re.I)


def _gold_ids(case: dict) -> set[int]:
    ids: set[int] = set()
    for key in ("doc_ids", "gold_sets"):
        val = case.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, list):
                    ids.update(int(x) for x in item)
                elif isinstance(item, int | str):
                    try:
                        ids.add(int(item))
                    except ValueError:
                        pass
    return ids


def ask(api: str, question: str, timeout: int) -> tuple[list[int], str]:
    data = json.dumps({"question": question, "debug": False}).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/ask", data=data, headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    cites = [int(c["document_id"]) for c in resp.get("citations", [])]
    return cites, str(resp.get("answer") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8088")
    ap.add_argument("--input", default="data/eval_v2/eval_set_v2.jsonl")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    cases = [
        json.loads(line)
        for line in Path(args.input).read_text("utf-8").splitlines()
        if line.strip()
    ]
    by_cat: dict[str, list[bool]] = defaultdict(list)
    abstain_total = abstain_ok = 0
    cite_total = cite_ok = 0
    failures: list[str] = []

    print(f"Running {len(cases)} eval_v2 cases against {args.api}\n")
    for c in cases:
        cid = c.get("id", "?")
        should_abstain = bool(c.get("should_abstain"))
        try:
            cites, answer = ask(args.api, c["question"], args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"  {cid:8} ERROR: {exc}")
            by_cat[c.get("category", "?")].append(False)
            failures.append(f"{cid} (error)")
            continue

        if should_abstain:
            ok = bool(_ABSTAIN.search(answer))
            abstain_total += 1
            abstain_ok += int(ok)
            kind = "ABSTAIN"
        else:
            gold = _gold_ids(c)
            ok = bool(gold & set(cites))
            cite_total += 1
            cite_ok += int(ok)
            kind = "CITE"
        by_cat[c.get("category", "?")].append(ok)
        if not ok:
            failures.append(f"{cid} ({kind})")
        mark = "OK  " if ok else "MISS"
        print(f"  {cid:8} {kind:8} {mark} cited={cites[:4]}")
        sys.stdout.flush()

    print("\nBy category:")
    for cat, res in sorted(by_cat.items()):
        print(f"  {cat:22} {sum(res)}/{len(res)}")
    total_ok = cite_ok + abstain_ok
    total = cite_total + abstain_total
    print("\nSummary:")
    print(f"  gold-cited (non-abstain): {cite_ok}/{cite_total}")
    print(f"  abstained (should_abstain): {abstain_ok}/{abstain_total}")
    print(f"  TOTAL: {total_ok}/{total}")
    if failures:
        print(f"\nmisses: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
