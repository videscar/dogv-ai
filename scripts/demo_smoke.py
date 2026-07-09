from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


def _now_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log_line(lines: list[str], message: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    line = f"[{timestamp}] {message}"
    lines.append(line)
    print(line)


def _check_health(base_url: str) -> dict:
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        response = client.get("/health")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError("/health returned unexpected payload")
    return payload


def _check_ready(base_url: str) -> dict:
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        response = client.get("/ready")
    response.raise_for_status()
    payload = response.json()
    required = {"ready", "status", "reason", "freshness", "startup_sync"}
    if not isinstance(payload, dict) or not required.issubset(payload.keys()):
        raise RuntimeError("/ready payload is missing required keys")
    return payload


def _check_ask_happy_path(base_url: str, question: str, timeout_seconds: float) -> dict:
    with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
        response = client.post("/ask", json={"question": question, "debug": False})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("/ask happy path returned non-object JSON")
    if not isinstance(payload.get("answer"), str) or not payload.get("answer").strip():
        raise RuntimeError("/ask happy path missing non-empty answer")
    if not isinstance(payload.get("citations"), list):
        raise RuntimeError("/ask happy path missing citations list")
    return payload


def _check_ask_timeout_path(base_url: str, question: str, timeout_seconds: float) -> None:
    try:
        with httpx.Client(base_url=base_url, timeout=timeout_seconds) as client:
            client.post("/ask", json={"question": question, "debug": False})
    except httpx.TimeoutException:
        return
    raise RuntimeError(
        "timeout path did not trigger timeout; lower --timeout-seconds or use a heavier prompt"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument(
        "--question",
        default="zzzzzz qqqqq no-match-token-123456789",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.5,
        help="Timeout used for deterministic timeout-path validation.",
    )
    parser.add_argument(
        "--ask-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout for /ask happy-path validation.",
    )
    parser.add_argument("--log-path", default="")
    args = parser.parse_args()

    log_path = (
        Path(args.log_path) if args.log_path else Path("logs/demo") / f"smoke_{_now_ts()}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    exit_code = 0
    results: dict[str, str] = {}

    _log_line(lines, f"smoke.start base_url={args.base_url}")

    checks = [
        ("health", lambda: _check_health(args.base_url)),
        ("ready", lambda: _check_ready(args.base_url)),
        (
            "ask_happy",
            lambda: _check_ask_happy_path(
                args.base_url,
                args.question,
                args.ask_timeout_seconds,
            ),
        ),
        (
            "ask_timeout",
            lambda: _check_ask_timeout_path(args.base_url, args.question, args.timeout_seconds),
        ),
    ]

    for name, fn in checks:
        try:
            payload = fn()
            if payload is None:
                _log_line(lines, f"check.pass name={name}")
            else:
                summary = json.dumps(payload, ensure_ascii=True)[:300]
                _log_line(lines, f"check.pass name={name} payload={summary}")
            results[name] = "pass"
        except Exception as exc:
            _log_line(lines, f"check.fail name={name} error={exc}")
            results[name] = "fail"
            exit_code = 1

    _log_line(lines, f"smoke.done status={'pass' if exit_code == 0 else 'fail'}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"log_path={log_path}")
    print("results=" + json.dumps(results, ensure_ascii=True, sort_keys=True))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
