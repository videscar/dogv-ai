"""Drive the live /ask through multi-turn conversations and score the final turn.

For each scenario in eval_multiturn.jsonl, sends each user turn in order with the
accumulating conversation history (exactly as the Chainlit UI does), capturing the
citations + answer at every turn. The LAST turn is scored against the scenario's
alt-gold (see eval_v2/altgold.py). The point: show that a named follow-up recovers
the flagship that the vague turn-1 misses.

Usage: python eval_v2/collect_multiturn.py [--base-url http://127.0.0.1:8088] \
           [--set data/eval_v2/eval_multiturn.jsonl] [--out data/eval_v2/answers_multiturn.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

from eval_v2.altgold import accept_set, citation_any_hit, citation_full_hit

DEFAULT_SET = "data/eval_v2/eval_multiturn.jsonl"
DEFAULT_OUT = "data/eval_v2/answers_multiturn.jsonl"


def _ask(base: str, question: str, history: list[dict], timeout: float) -> dict:
    payload = {"question": question, "debug": True, "history": history}
    with httpx.Client(base_url=base, timeout=timeout) as cl:
        r = cl.post("/ask", json=payload)
    r.raise_for_status()
    return r.json()


def run_scenario(base: str, sc: dict, timeout: float) -> dict:
    history: list[dict] = []
    turns_out: list[dict] = []
    user_turns = [t for t in sc["turns"] if t.get("role") == "user"]
    for ti, turn in enumerate(user_turns):
        q = turn["content"]
        started = time.perf_counter()
        d = _ask(base, q, list(history), timeout)
        answer = str(d.get("answer") or "")
        cited = [c.get("document_id") for c in (d.get("citations") or [])]
        prof = (d.get("debug") or {}).get("profile") or {}
        ctx = prof.get("contextualize") or {}
        turns_out.append({
            "i": ti, "question": q, "answer": answer, "cited_doc_ids": cited,
            "contextualized": ctx.get("applied"),
            "rewritten_question": ctx.get("rewritten_question"),
            "latency_s": round(time.perf_counter() - started, 1),
        })
        # Mirror the UI: append the user turn + the plain answer to history.
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

    accept = accept_set(sc)
    final = turns_out[-1]
    first = turns_out[0]
    return {
        "id": sc["id"], "language": sc.get("language"), "seed": sc.get("seed"),
        "gold_doc_ids": sc.get("gold_doc_ids"), "accept_doc_ids": sorted(accept),
        "turn1_any_hit": citation_any_hit(first["cited_doc_ids"], accept),
        "final_any_hit": citation_any_hit(final["cited_doc_ids"], accept),
        "final_full_hit": citation_full_hit(final["cited_doc_ids"], sc.get("gold_sets")),
        "turns": turns_out, "note": sc.get("note"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8088")
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--ids", default="")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    scenarios = [json.loads(l) for l in open(args.set, encoding="utf-8") if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        scenarios = [s for s in scenarios if s["id"] in want]

    results = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for sc in scenarios:
            try:
                res = run_scenario(base, sc, args.timeout)
            except Exception as exc:
                res = {"id": sc["id"], "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{sc['id']}] ERROR {res['error']}", flush=True)
                fout.write(json.dumps(res, ensure_ascii=False) + "\n"); fout.flush()
                results.append(res); continue
            fout.write(json.dumps(res, ensure_ascii=False) + "\n"); fout.flush()
            results.append(res)
            print(f"[{res['id']}] {res['language']} turn1={'HIT' if res['turn1_any_hit'] else 'miss'} "
                  f"-> final={'HIT' if res['final_any_hit'] else 'MISS'} "
                  f"(ctx={res['turns'][-1].get('contextualized')}) gold={res['gold_doc_ids']} "
                  f"final_cites={res['turns'][-1]['cited_doc_ids']}", flush=True)

    ok = [r for r in results if "error" not in r]
    if ok:
        t1 = sum(r["turn1_any_hit"] for r in ok)
        tf = sum(r["final_any_hit"] for r in ok)
        print(f"\n=== MULTI-TURN RECALL (n={len(ok)}) ===", flush=True)
        print(f"turn-1 any_hit: {t1}/{len(ok)}  ->  final-turn any_hit: {tf}/{len(ok)}", flush=True)
        print(f"lift from follow-up: +{tf - t1}", flush=True)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
