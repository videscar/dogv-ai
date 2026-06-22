"""Resolver-precision harness for the on-demand historical-fetch feature.

Runs api.dogv_resolver.resolve_question over the tester's gold set and scores
whether the resolved document's title carries the expected (gold) disposition.
Non-circular: gold_title_contains comes from the tester's manual DOGV lookups.

Usage: python scripts/eval_ondemand_resolver.py
(Needs outbound HTTPS to dogv.gva.es.)
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.dogv_resolver import parse_reference, resolve_question  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "eval_ondemand_gold.json"
OUT = ROOT / "data" / "eval_reports" / "ondemand_resolver_last.txt"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return " ".join(s.split())


def main() -> int:
    cases = json.loads(GOLD.read_text(encoding="utf-8"))["cases"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = open(OUT, "w", buffering=1, encoding="utf-8")  # line-buffered: survives a kill

    def emit(*a: object) -> None:
        line = " ".join(str(x) for x in a)
        print(line)
        print(line, file=out, flush=True)

    def run_case(c: dict) -> dict:
        q = c["question"]
        ref = parse_reference(q)
        ref_ok = ref is not None and ref.num_year == c["ref"]
        resolved = resolve_question(q) if ref_ok else None
        want = _norm(c["gold_title_contains"])
        got_title = resolved.titulo if resolved else ""
        title_ok = bool(resolved) and want in _norm(got_title)
        id_ok = ("gold_id" not in c) or (resolved is not None and resolved.disposicion_id == c["gold_id"])
        return {"c": c, "ref": ref, "ref_ok": ref_ok, "resolved": resolved,
                "title_ok": title_ok, "id_ok": id_ok, "got_title": got_title}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run_case, cases))

    passed = 0
    failed: list[str] = []
    for r in results:
        c, ref, resolved = r["c"], r["ref"], r["resolved"]
        ref_ok, title_ok, id_ok, got_title = r["ref_ok"], r["title_ok"], r["id_ok"], r["got_title"]
        ok = ref_ok and title_ok and id_ok
        passed += int(ok)
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failed.append(c["id"])
        rid = resolved.disposicion_id if resolved else "-"
        fpub = resolved.fecha_publicacion if resolved else "-"
        emit(f"[{flag}] {c['id']:4} ref={c['ref']:9} id={rid} pub={fpub}")
        emit(f"        got   {got_title[:100]!r}")
        if not ref_ok:
            emit(f"        !! parse_reference mismatch: {ref}")
        if resolved and not id_ok:
            emit(f"        !! id mismatch: got {resolved.disposicion_id} want {c['gold_id']}")
    n = len(cases)
    emit(f"Resolver precision: {passed}/{n} = {passed / n:.2%}")
    if failed:
        emit("Failed: " + ", ".join(failed))
    out.close()
    return 0 if passed == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
