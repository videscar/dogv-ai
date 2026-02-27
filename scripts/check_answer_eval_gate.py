from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("Answer eval report must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    report = _load_report(Path(args.report))
    summary = report.get("summary") or {}

    aggregate_hard_score = summary.get("aggregate_hard_score")
    if aggregate_hard_score is None:
        aggregate_hard_score = summary.get("aggregate_score")
    aggregate_hard_score = float(aggregate_hard_score or 0.0)
    critical_failed = summary.get("critical_failed_ids") or []
    api_errors = int(summary.get("api_errors") or 0)
    failure_breakdown = summary.get("failure_breakdown") or {}

    errors: list[str] = []
    if aggregate_hard_score < args.threshold:
        errors.append(
            f"aggregate_hard_score {aggregate_hard_score:.4f} < threshold {args.threshold:.4f}"
        )
    if critical_failed:
        errors.append(f"critical_failed_ids: {critical_failed}")
    if api_errors > 0:
        errors.append(f"api_errors: {api_errors}")

    if errors:
        print("Answer eval gate failed:")
        for item in errors:
            print(f"- {item}")
        if failure_breakdown:
            print("- failure_breakdown:", failure_breakdown)
        return 1

    print("Answer eval gate passed.")
    print(f"- aggregate_hard_score: {aggregate_hard_score:.4f}")
    print("- critical failures: 0")
    print("- api errors: 0")
    if failure_breakdown:
        print(f"- failure_breakdown: {failure_breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
