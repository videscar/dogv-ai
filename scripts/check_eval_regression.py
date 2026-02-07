from __future__ import annotations

import argparse
import json
import os


def _load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_recall(report: dict) -> dict[str, dict[str, float]]:
    summary = report.get("summary") or {}
    recall = summary.get("recall") or {}
    return {stage: metrics for stage, metrics in recall.items() if isinstance(metrics, dict)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/eval_baseline.json")
    parser.add_argument("--report", required=True)
    parser.add_argument("--stages", default="hybrid,rerank")
    parser.add_argument("--k", default="5,10")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="Copy --report into --baseline and exit.",
    )
    args = parser.parse_args()

    if args.set_baseline:
        report = _load_report(args.report)
        os.makedirs(os.path.dirname(args.baseline) or ".", exist_ok=True)
        with open(args.baseline, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Baseline updated at {args.baseline}")
        return 0

    if not os.path.exists(args.baseline):
        raise SystemExit(
            f"Baseline not found: {args.baseline}. "
            f"Run with --set-baseline once to initialize it."
        )

    baseline = _load_report(args.baseline)
    report = _load_report(args.report)

    baseline_recall = _extract_recall(baseline)
    report_recall = _extract_recall(report)

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    k_values = [k.strip() for k in args.k.split(",") if k.strip()]

    regressions: list[str] = []
    for stage in stages:
        base_metrics = baseline_recall.get(stage) or {}
        new_metrics = report_recall.get(stage) or {}
        for k in k_values:
            key = f"recall@{k}"
            base_value = float(base_metrics.get(key, 0.0))
            new_value = float(new_metrics.get(key, 0.0))
            delta = new_value - base_value
            if delta < -args.tolerance:
                regressions.append(
                    f"{stage} {key}: {new_value:.4f} (baseline {base_value:.4f})"
                )

    if regressions:
        print("Recall regressions detected:")
        for item in regressions:
            print(f"- {item}")
        return 1

    print("No recall regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
