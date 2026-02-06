from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.taxonomy import canonical_doc_kind


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_ollama_ready(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/api/version", timeout=2) as resp:
                if resp.status == 200:
                    return
        except URLError as exc:
            last_err = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            last_err = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Ollama at {base_url} not ready within {timeout_s}s (last error: {last_err})")


def _build_run_eval_cmd(
    input_path: Path,
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--k-values",
        args.k_values,
        "--max-candidates",
        str(args.max_candidates),
        "--rerank-top-n",
        str(args.rerank_top_n),
    ]
    if args.include_nofilter:
        cmd.append("--include-nofilter")
    if args.write_csv:
        cmd.append("--write-csv")
    if args.fast_intent:
        cmd.append("--fast-intent")
    if args.skip_rerank_llm:
        cmd.append("--skip-rerank-llm")
    return cmd


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _split_eval_set(eval_set: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left: list[dict[str, Any]] = []
    right: list[dict[str, Any]] = []
    for idx, row in enumerate(eval_set):
        if idx % 2 == 0:
            left.append(row)
        else:
            right.append(row)
    return left, right


def _compute_summary(
    queries: list[dict[str, Any]],
    k_values: list[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not queries:
        return {}, {}, {}

    max_k = str(max(k_values))
    stages = sorted({stage for q in queries for stage in (q.get("hits") or {}).keys()})
    recall: dict[str, dict[str, float]] = {}
    misses: dict[str, list[dict[str, Any]]] = {}

    for stage in stages:
        counts = {str(k): 0 for k in k_values}
        total = 0
        misses[stage] = []
        for q in queries:
            stage_hits = (q.get("hits") or {}).get(stage)
            if not stage_hits:
                continue
            total += 1
            for k in k_values:
                if stage_hits.get(str(k)):
                    counts[str(k)] += 1
            if not stage_hits.get(max_k, False) and len(misses[stage]) < 20:
                misses[stage].append(
                    {
                        "id": q.get("id"),
                        "question": q.get("question"),
                        "gold_sets": q.get("gold_sets"),
                        "doc_kind": q.get("doc_kind"),
                        "doc_subkind": q.get("doc_subkind"),
                    }
                )
        denom = total or 1
        recall[stage] = {f"recall@{k}": counts[str(k)] / denom for k in k_values}

    by_kind_totals: dict[str, int] = {}
    by_kind_counts: dict[str, dict[str, dict[str, int]]] = {}
    for q in queries:
        kind_raw = q.get("doc_kind")
        if not kind_raw:
            continue
        kind = canonical_doc_kind(kind_raw) or str(kind_raw)
        by_kind_totals[kind] = by_kind_totals.get(kind, 0) + 1
        q_hits = q.get("hits") or {}
        by_kind_counts.setdefault(kind, {})
        for stage, stage_hits in q_hits.items():
            by_kind_counts[kind].setdefault(stage, {str(k): 0 for k in k_values})
            for k in k_values:
                if stage_hits.get(str(k), False):
                    by_kind_counts[kind][stage][str(k)] += 1

    by_kind_summary: dict[str, dict[str, dict[str, float]]] = {}
    for kind, stage_counts in by_kind_counts.items():
        denom = by_kind_totals.get(kind, 0) or 1
        by_kind_summary[kind] = {}
        for stage, counts in stage_counts.items():
            by_kind_summary[kind][stage] = {f"recall@{k}": counts[str(k)] / denom for k in k_values}

    timing_keys = sorted({key for q in queries for key in (q.get("timings") or {}).keys()})
    avg_latency: dict[str, float] = {}
    total_queries = len(queries) or 1
    for key in timing_keys:
        avg_latency[key] = sum(float((q.get("timings") or {}).get(key, 0.0) or 0.0) for q in queries) / total_queries

    return recall, by_kind_summary, {"misses": misses, "avg_latency": avg_latency}


def _merge_csv(parts: list[Path], output_csv: Path) -> None:
    header = ["query_id", "stage", "rank", "document_id", "score", "in_gold"]
    with output_csv.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header)
        writer.writeheader()
        for part in parts:
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    writer.writerow(row)


def _terminate_process(proc: subprocess.Popen[Any], name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait(timeout=5)
    print(f"[cleanup] terminated {name} (pid={proc.pid})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval_set.json")
    parser.add_argument("--output-dir", default="data/eval_reports")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--k-values", default="1,3,5,10,20,50")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=5)
    parser.add_argument("--include-nofilter", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--fast-intent", action="store_true")
    parser.add_argument("--skip-rerank-llm", action="store_true")
    parser.add_argument("--gpu0-id", default="0")
    parser.add_argument("--gpu1-id", default="1")
    parser.add_argument("--gpu0-port", type=int, default=11436)
    parser.add_argument("--gpu1-port", type=int, default=11437)
    parser.add_argument("--ollama-models", default="/usr/share/ollama/.ollama/models")
    parser.add_argument("--ollama-llm-library", default="cuda_v12")
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or _now_run_id()
    temp_dir = output_dir / f".parallel_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    eval_set = _load_json(input_path)
    if not isinstance(eval_set, list) or not eval_set:
        raise SystemExit("Input eval set is empty or invalid")
    if len(eval_set) < 2:
        raise SystemExit("Need at least 2 queries for parallel split")

    if not _port_is_free(args.gpu0_port):
        raise SystemExit(f"Port {args.gpu0_port} is busy; choose another with --gpu0-port")
    if not _port_is_free(args.gpu1_port):
        raise SystemExit(f"Port {args.gpu1_port} is busy; choose another with --gpu1-port")

    left, right = _split_eval_set(eval_set)
    if not left or not right:
        raise SystemExit("Split produced an empty shard; cannot continue")

    left_path = temp_dir / "eval_left.json"
    right_path = temp_dir / "eval_right.json"
    _write_json(left_path, left)
    _write_json(right_path, right)
    print(f"[split] left={len(left)} right={len(right)}", flush=True)

    ollama0_log = temp_dir / "ollama_gpu0.log"
    ollama1_log = temp_dir / "ollama_gpu1.log"
    worker0_log = temp_dir / "worker0.log"
    worker1_log = temp_dir / "worker1.log"

    base_env = os.environ.copy()
    ollama0_env = base_env.copy()
    ollama1_env = base_env.copy()
    ollama0_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu0_id),
            "OLLAMA_HOST": f"127.0.0.1:{args.gpu0_port}",
            "OLLAMA_LLM_LIBRARY": args.ollama_llm_library,
            "OLLAMA_MODELS": args.ollama_models,
            "OLLAMA_DEBUG": "INFO",
        }
    )
    ollama1_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu1_id),
            "OLLAMA_HOST": f"127.0.0.1:{args.gpu1_port}",
            "OLLAMA_LLM_LIBRARY": args.ollama_llm_library,
            "OLLAMA_MODELS": args.ollama_models,
            "OLLAMA_DEBUG": "INFO",
        }
    )

    ollama0 = None
    ollama1 = None
    worker0 = None
    worker1 = None

    try:
        with ollama0_log.open("w", encoding="utf-8") as fh0, ollama1_log.open("w", encoding="utf-8") as fh1:
            ollama0 = subprocess.Popen(
                ["/usr/local/bin/ollama", "serve"],
                env=ollama0_env,
                stdout=fh0,
                stderr=subprocess.STDOUT,
                text=True,
            )
            ollama1 = subprocess.Popen(
                ["/usr/local/bin/ollama", "serve"],
                env=ollama1_env,
                stdout=fh1,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"[start] ollama gpu0 pid={ollama0.pid} port={args.gpu0_port}", flush=True)
            print(f"[start] ollama gpu1 pid={ollama1.pid} port={args.gpu1_port}", flush=True)

            _wait_ollama_ready(f"http://127.0.0.1:{args.gpu0_port}", args.startup_timeout)
            _wait_ollama_ready(f"http://127.0.0.1:{args.gpu1_port}", args.startup_timeout)
            print("[ready] both ollama servers are healthy", flush=True)

        left_run_id = f"{run_id}_gpu0"
        right_run_id = f"{run_id}_gpu1"

        worker0_env = base_env.copy()
        worker1_env = base_env.copy()
        worker0_env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{args.gpu0_port}"
        worker1_env["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{args.gpu1_port}"

        cmd0 = _build_run_eval_cmd(left_path, temp_dir, left_run_id, args)
        cmd1 = _build_run_eval_cmd(right_path, temp_dir, right_run_id, args)
        print(f"[start] worker0 queries={len(left)}", flush=True)
        print(f"[start] worker1 queries={len(right)}", flush=True)

        with worker0_log.open("w", encoding="utf-8") as wh0, worker1_log.open("w", encoding="utf-8") as wh1:
            worker0 = subprocess.Popen(
                cmd0,
                env=worker0_env,
                stdout=wh0,
                stderr=subprocess.STDOUT,
                text=True,
            )
            worker1 = subprocess.Popen(
                cmd1,
                env=worker1_env,
                stdout=wh1,
                stderr=subprocess.STDOUT,
                text=True,
            )

            while True:
                rc0 = worker0.poll()
                rc1 = worker1.poll()
                if rc0 is not None and rc1 is not None:
                    break
                time.sleep(2)

        if worker0.returncode != 0:
            raise RuntimeError(f"worker0 failed (exit={worker0.returncode}). log: {worker0_log}")
        if worker1.returncode != 0:
            raise RuntimeError(f"worker1 failed (exit={worker1.returncode}). log: {worker1_log}")
        print("[done] both worker runs completed", flush=True)

        left_report_path = temp_dir / f"{left_run_id}.json"
        right_report_path = temp_dir / f"{right_run_id}.json"
        if not left_report_path.exists() or not right_report_path.exists():
            raise RuntimeError("Missing partial report(s) after workers finished")

        left_report = _load_json(left_report_path)
        right_report = _load_json(right_report_path)
        queries = list(left_report.get("queries") or []) + list(right_report.get("queries") or [])
        queries.sort(key=lambda row: int(row.get("id") or 0))

        if not queries:
            raise RuntimeError("No queries found in partial reports")

        k_values = left_report.get("k_values") or right_report.get("k_values") or []
        recall, by_kind_summary, extras = _compute_summary(queries, [int(x) for x in k_values])

        merged = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eval_set": str(input_path),
            "size": len(queries),
            "k_values": k_values,
            "settings": dict(left_report.get("settings") or {}),
            "summary": {
                "recall": recall,
                "recall_by_doc_kind": by_kind_summary,
                "misses": extras["misses"],
                "avg_latency": extras["avg_latency"],
            },
            "queries": queries,
        }

        report_path = output_dir / f"{run_id}.json"
        _write_json(report_path, merged)
        print(f"[merge] wrote report: {report_path}", flush=True)

        if args.write_csv:
            left_csv = temp_dir / f"{left_run_id}.csv"
            right_csv = temp_dir / f"{right_run_id}.csv"
            merged_csv = output_dir / f"{run_id}.csv"
            _merge_csv([left_csv, right_csv], merged_csv)
            print(f"[merge] wrote csv: {merged_csv}", flush=True)

    finally:
        if worker0 is not None and worker0.poll() is None:
            _terminate_process(worker0, "worker0")
        if worker1 is not None and worker1.poll() is None:
            _terminate_process(worker1, "worker1")
        if ollama0 is not None and ollama0.poll() is None:
            _terminate_process(ollama0, "ollama-gpu0")
        if ollama1 is not None and ollama1.poll() is None:
            _terminate_process(ollama1, "ollama-gpu1")
        if not args.keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
