#!/usr/bin/env python3
"""Run the identifier probe set against a live /ask API and report per-class results.

Retrieval-level harness for the identifier-layer work (docs/identifier_layer_design.md).
GOLD_HIT = any of a probe's gold_doc_ids appears in the citations. Prod is deterministic,
so a single pass suffices. Not an eval-gate: the identifier layer fires 0/100 on eval_v2,
so this probe set is its validation harness.

    python scripts/oneoff/run_identifier_probes.py
    python scripts/oneoff/run_identifier_probes.py --class code
    python scripts/oneoff/run_identifier_probes.py --api http://127.0.0.1:8090
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

PROBES = Path(__file__).resolve().parents[2] / "data" / "probes" / "identifier_probes.jsonl"


def load_probes(id_class: str | None) -> list[dict]:
    probes = []
    for line in PROBES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        if id_class and p.get("id_class") != id_class:
            continue
        probes.append(p)
    return probes


def ask(api: str, question: str, timeout: int) -> list[int]:
    data = json.dumps({"question": question, "debug": False}).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/ask", data=data, headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    return [int(c["document_id"]) for c in resp.get("citations", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8088")
    ap.add_argument("--class", dest="id_class", default=None)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    probes = load_probes(args.id_class)
    if not probes:
        print("no probes matched")
        return 2

    by_class: dict[str, list[bool]] = defaultdict(list)
    regressions: list[str] = []
    print(f"Running {len(probes)} probe(s) against {args.api}\n")
    for p in probes:
        try:
            cited = ask(args.api, p["question"], args.timeout)
            hit = any(g in cited for g in p["gold_doc_ids"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {p['id']:5} {p['id_class']:7} ERROR: {exc}")
            by_class[p["id_class"]].append(False)
            continue
        by_class[p["id_class"]].append(hit)
        mark = "OK  " if hit else "MISS"
        traps = ",".join(p.get("traps") or []) or "-"
        print(
            f"  {p['id']:5} {p['id_class']:7} {mark} gold={p['gold_doc_ids']} "
            f"cited={cited[:4]} [{traps}] baseline={p.get('baseline')}"
        )
        if p.get("baseline") == "pass" and not hit:
            regressions.append(p["id"])
        sys.stdout.flush()

    print("\nPer-class:")
    total_hit = total = 0
    for cls, results in sorted(by_class.items()):
        h, n = sum(results), len(results)
        total_hit += h
        total += n
        print(f"  {cls:7} {h}/{n}")
    print(f"  {'TOTAL':7} {total_hit}/{total}")

    if regressions:
        print(f"\nREGRESSIONS (baseline=pass now MISS): {', '.join(regressions)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
