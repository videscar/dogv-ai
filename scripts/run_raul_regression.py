"""Run Raul's combined gold-tester question set against live prod and report regressions.

Reads data/raul_regression/questions.json (the unified, de-duplicated set extracted
from both of Raul's documents), asks each question against the running API, and
checks two regression signals:
  - answered: the model did not abstain ("no consta" / "no puedo confirmar ...");
  - cited_norm: when the question names a specific disposition (N/YYYY) or Raul gave
    a gold disposition, at least one citation's title carries that N/YYYY.

Writes a timestamped markdown report and refreshes the living doc's "last run"
column. This is the regression harness we re-run before asking Raul to continue.

Usage:
    python scripts/run_raul_regression.py [--api http://localhost:8088] [--out PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
QFILE = ROOT / "data" / "raul_regression" / "questions.json"
_ABSTAIN = re.compile(r"no consta|no puedo confirmar|no hay publicaciones|no se encontr", re.I)
_NY = re.compile(r"\b(\d+/\d{4})\b")


def _expected_ny(item: dict) -> str | None:
    """The N/YYYY the answer should cite — from the named norm or Raul's gold note."""
    for field in ("named_norm", "gold_disposition"):
        m = _NY.search(item.get(field) or "")
        if m:
            return m.group(1)
    return None


def ask(api: str, question: str, timeout: int = 240) -> dict:
    r = requests.post(f"{api}/ask", json={"question": question}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run(api: str) -> list[dict]:
    items = json.loads(QFILE.read_text())
    results = []
    for it in items:
        exp_ny = _expected_ny(it)
        row = {**it, "expected_ny": exp_ny}
        try:
            t0 = time.time()
            d = ask(api, it["question"])
            row["seconds"] = round(time.time() - t0, 1)
            answer = d.get("answer") or ""
            cites = d.get("citations") or []
            row["answer"] = answer
            row["cites"] = [{"document_id": c.get("document_id"), "ref": c.get("ref"),
                             "title": c.get("title")} for c in cites]
            row["answered"] = not bool(_ABSTAIN.search(answer))
            if exp_ny:
                row["cited_norm"] = any(exp_ny in (c.get("title") or "") for c in cites)
            else:
                row["cited_norm"] = None  # no specific norm to check
            row["error"] = None
        except Exception as exc:  # noqa: BLE001
            row.update(answer="", cites=[], answered=False, cited_norm=None,
                       seconds=None, error=str(exc))
        # A regression = abstained, OR named a norm it failed to cite, OR errored
        row["regression"] = bool(
            row["error"]
            or not row["answered"]
            or (row["cited_norm"] is False)
        )
        results.append(row)
        flag = "REGRESSION" if row["regression"] else "ok"
        print(f"#{it['id']:2} {flag:11} answered={row['answered']} "
              f"cited_norm={row['cited_norm']} cites={[c['ref'] for c in row['cites']]}")
    return results


def write_report(results: list[dict], out: Path) -> None:
    regs = [r for r in results if r["regression"]]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Raul regression run — {ts}", ""]
    lines.append(f"**{len(results) - len(regs)}/{len(results)} clean**, "
                 f"{len(regs)} regression(s).\n")
    if regs:
        lines.append("## Regressions")
        for r in regs:
            why = ("error: " + r["error"]) if r["error"] else (
                "abstained" if not r["answered"] else f"missing norm cite ({r['expected_ny']})")
            lines.append(f"- **#{r['id']} [{r['source']}]** ({why}) — {r['question']}")
        lines.append("")
    lines.append("## All results")
    lines.append("| # | src | answered | norm cite | cites | secs | question |")
    lines.append("|---|-----|----------|-----------|-------|------|----------|")
    for r in results:
        cn = {True: "✓", False: "✗", None: "—"}[r["cited_norm"]]
        cites = ", ".join(c["ref"] or "?" for c in r["cites"]) or "—"
        q = r["question"].replace("|", "/")[:70]
        lines.append(f"| {r['id']} | {r['source']} | {'✓' if r['answered'] else '✗'} "
                     f"| {cn} | {cites} | {r['seconds']} | {q} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {out}  ({len(results) - len(regs)}/{len(results)} clean)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8088")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = run(args.api)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    out = Path(args.out) if args.out else QFILE.parent / f"run_{stamp}.md"
    write_report(results, out)
    return 1 if any(r["regression"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
