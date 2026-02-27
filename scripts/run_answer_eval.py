from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import unicodedata

import httpx

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Answer eval input must be a JSON array")
    return payload


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize_eval_text(term.strip())
    if not normalized_term:
        return False
    return normalized_term in _normalize_eval_text(text)


def _normalize_eval_text(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(ascii_text.lower().split())


def _citation_count(citations: Any) -> int:
    if not isinstance(citations, list):
        return 0
    return len(citations)


def _score_case(
    case: dict[str, Any],
    answer: str,
    citations: Any,
) -> tuple[dict[str, Any], dict[str, float], bool]:
    must_include = [str(x).strip() for x in (case.get("must_include") or []) if str(x).strip()]
    must_not_include = [str(x).strip() for x in (case.get("must_not_include") or []) if str(x).strip()]

    include_hits = [term for term in must_include if _contains_term(answer, term)]
    missing_include = [term for term in must_include if term not in include_hits]
    must_not_hits = [term for term in must_not_include if _contains_term(answer, term)]

    lexical_recall = (len(include_hits) / len(must_include)) if must_include else 1.0

    required_citations = max(0, int(case.get("required_citations") or 0))
    found_citations = _citation_count(citations)
    if required_citations == 0:
        citation_faithfulness = 1.0
    else:
        citation_faithfulness = min(1.0, found_citations / required_citations)

    hard_faithfulness = citation_faithfulness
    if must_not_hits:
        hard_faithfulness = 0.0

    total_score = hard_faithfulness
    passed = hard_faithfulness >= 0.85

    checks = {
        "must_include_hits": include_hits,
        "missing_must_include": missing_include,
        "must_not_hits": must_not_hits,
        "required_citations": required_citations,
        "found_citations": found_citations,
    }
    scores = {
        "citation_faithfulness": round(citation_faithfulness, 4),
        "lexical_recall": round(lexical_recall, 4),
        "hard_faithfulness": round(hard_faithfulness, 4),
        "total": round(total_score, 4),
    }
    return checks, scores, passed


def _classify_failure(
    *,
    error: str | None,
    checks: dict[str, Any],
    passed: bool,
) -> str | None:
    if passed:
        return None
    if error:
        return "runtime_error"

    required_citations = int(checks.get("required_citations") or 0)
    found_citations = int(checks.get("found_citations") or 0)
    must_include_hits = checks.get("must_include_hits") or []
    missing_include = checks.get("missing_must_include") or []
    must_not_hits = checks.get("must_not_hits") or []

    has_citation_gap = required_citations > found_citations
    if has_citation_gap:
        return "citation_gap"

    # Heuristic: answer is likely semantically valid but failed strict term matching.
    if not must_not_hits and missing_include and must_include_hits:
        return "lexical_mismatch"

    return "content_failure"


def _is_lexical_mismatch(checks: dict[str, Any]) -> bool:
    must_include_hits = checks.get("must_include_hits") or []
    missing_include = checks.get("missing_must_include") or []
    must_not_hits = checks.get("must_not_hits") or []
    return bool(missing_include and not must_not_hits and must_include_hits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_answer_demo_v1.json")
    parser.add_argument("--output-dir", default="data/eval_reports")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--rescue-timeout",
        type=float,
        default=900.0,
        help="Final timeout for one rescue attempt after timeout retries are exhausted.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or _now_run_id()

    cases = _load_cases(input_path)
    if not cases:
        raise SystemExit("Answer eval set is empty")

    results: list[dict[str, Any]] = []
    api_errors = 0

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        total = len(cases)
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or "")
            question = str(case.get("question") or "").strip()
            language = str(case.get("language") or "")
            critical = bool(case.get("critical"))
            if not question:
                raise SystemExit(f"Invalid case {case_id or '<missing id>'}: question is required")

            error: str | None = None
            answer = ""
            citations: Any = []
            started = time.perf_counter()
            payload: dict[str, Any] = {}
            last_exc: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    response = client.post("/ask", json={"question": question, "debug": False})
                    response.raise_for_status()
                    payload = response.json()
                    answer = str(payload.get("answer") or "")
                    citations = payload.get("citations") or []
                    last_exc = None
                    break
                except httpx.TimeoutException as exc:
                    last_exc = exc
                    if attempt < args.retries:
                        time.sleep(max(0.0, args.retry_delay))
                        continue
                except Exception as exc:
                    last_exc = exc
                    break

            if isinstance(last_exc, httpx.TimeoutException) and args.rescue_timeout > args.timeout:
                try:
                    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.rescue_timeout) as rescue_client:
                        response = rescue_client.post("/ask", json={"question": question, "debug": False})
                    response.raise_for_status()
                    payload = response.json()
                    answer = str(payload.get("answer") or "")
                    citations = payload.get("citations") or []
                    last_exc = None
                except Exception as exc:
                    last_exc = exc

            if last_exc is not None:
                error = str(last_exc)
                api_errors += 1
            latency_seconds = round(time.perf_counter() - started, 3)

            checks, scores, passed = _score_case(case, answer, citations)
            if error:
                passed = False
                scores = {
                    "citation_faithfulness": 0.0,
                    "lexical_recall": 0.0,
                    "hard_faithfulness": 0.0,
                    "total": 0.0,
                }

            results.append(
                {
                    "id": case_id,
                    "language": language,
                    "critical": critical,
                    "question": question,
                    "checks": checks,
                    "scores": scores,
                    "passed": passed,
                    "latency_seconds": latency_seconds,
                    "error": error,
                    "response": {
                        "answer": answer,
                        "citations": citations,
                    },
                }
            )
            status = "pass" if passed else "fail"
            err_tag = " error" if error else ""
            print(
                f"[progress] {idx}/{total} case={case_id or '<no-id>'} status={status} "
                f"latency={latency_seconds:.3f}s{err_tag}",
                flush=True,
            )

    total_cases = len(results)
    aggregate_hard_score = sum(item["scores"]["total"] for item in results) / max(1, total_cases)
    lexical_recall_avg = sum(float(item["scores"].get("lexical_recall") or 0.0) for item in results) / max(1, total_cases)
    passed_cases = sum(1 for item in results if item["passed"])
    critical_cases = [item for item in results if item["critical"]]
    critical_failed = [item["id"] for item in critical_cases if not item["passed"]]
    failure_categories = ["runtime_error", "citation_gap", "lexical_mismatch", "content_failure"]
    failure_breakdown = {name: 0 for name in failure_categories}
    lexical_mismatches = 0
    for item in results:
        category = _classify_failure(
            error=item.get("error"),
            checks=item.get("checks") or {},
            passed=bool(item.get("passed")),
        )
        lexical_flag = _is_lexical_mismatch(item.get("checks") or {})
        item["failure_category"] = category
        item["lexical_mismatch"] = lexical_flag
        if lexical_flag:
            lexical_mismatches += 1
        if category in failure_breakdown:
            failure_breakdown[category] += 1

    report = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": str(input_path),
        "base_url": args.base_url,
        "size": total_cases,
        "weights": {
            "hard_faithfulness": 1.0,
            "lexical_recall": 0.0,
        },
        "summary": {
            "aggregate_hard_score": round(aggregate_hard_score, 4),
            "aggregate_score": round(aggregate_hard_score, 4),
            "lexical_recall_avg": round(lexical_recall_avg, 4),
            "pass_rate": round(passed_cases / max(1, total_cases), 4),
            "passed_cases": passed_cases,
            "critical_total": len(critical_cases),
            "critical_passed": len(critical_cases) - len(critical_failed),
            "critical_failed_ids": critical_failed,
            "api_errors": api_errors,
            "lexical_mismatches": lexical_mismatches,
            "failure_breakdown": failure_breakdown,
        },
        "cases": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"answer_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Answer eval report written to {report_path}")
    print(
        "aggregate_hard_score={:.4f} pass_rate={:.4f} critical_failed={} api_errors={} "
        "runtime_errors={} citation_gaps={} lexical_mismatches={} content_failures={}".format(
            report["summary"]["aggregate_hard_score"],
            report["summary"]["pass_rate"],
            len(critical_failed),
            api_errors,
            failure_breakdown["runtime_error"],
            failure_breakdown["citation_gap"],
            lexical_mismatches,
            failure_breakdown["content_failure"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
