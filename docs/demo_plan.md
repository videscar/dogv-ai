# DOGV AI Demo Readiness Plan (Chainlit) for February 26, 2026

## Summary
This plan targets a single-host demo stack (Postgres + Ollama + FastAPI + Chainlit), with scripted single-turn QA and a hard dual release gate (retrieval + answer quality).

Baseline date: February 8, 2026.
Demo deadline: February 26, 2026.

Current repo baseline:
- Backend endpoints are only `GET /health`, `GET /issues`, `GET /issues/{issue_id}/documents`, `POST /ask`.
- No Chainlit integration present.
- No `/ready` endpoint and `/ask` can run while startup sync is warming.
- Startup gap repair does one-call checks without retry and iterates all candidate date/language pairs.
- Eval gate is retrieval-only.
- No automated tests in repo.
- `scripts/run_eval_parallel.py` hardcodes `/usr/local/bin/ollama`.
- Latest eval baseline report: `data/eval_reports/20260206T132627Z.json`.

## Locked Decisions
- UI: Chainlit only.
- Deployment: single-host stack.
- Demo interaction target: scripted single-turn QA (3-5 prompts).
- Release gate: hard dual gate.
- Persistent plan path: `docs/demo_plan.md`.

## Demo Non-Goals
- No guaranteed multi-turn memory continuity between questions.
- No OpenAI-compatible API adapter for this demo cycle.
- No multi-host deployment orchestration for the February 26 demo.

## Public API and Interface Changes
1. Add `GET /ready` in `api/main.py`.
- Response shape:
  - `ready: bool`
  - `status: "ready" | "warming" | "degraded"`
  - `reason: str | null`
  - `freshness: { today, min_issue_date, max_issue_date, lag_days }`
  - `startup_sync: {...}`
- Rules:
  - `ready=false` when startup sync state is `running`.
  - `ready=false` when startup sync finished with `success=false`.
  - `ready=true` otherwise.

2. Gate `POST /ask` in `api/main.py`.
- If `ready=false` and `DEMO_ENFORCE_READY_GATE=true`, return `503` with readiness payload.

3. Add Chainlit interface in `ui/chainlit_app.py`.
- User message -> backend `/ask`.
- Render citations as clickable DOGV links using `html_url` fallback to `pdf_url`.
- Handle warming, timeout, backend unavailable, no-results states.

4. Add settings in `api/config.py` and `.env.example`.
- `demo_enforce_ready_gate: bool = True`
- `demo_request_timeout_seconds: int = 60`
- `auto_ingest_gap_check_retries: int = 3`
- `auto_ingest_gap_check_backoff_seconds: float = 1.5`
- `auto_ingest_gap_repair_scan_max_days: int = 45`
- `chainlit_backend_url: str = "http://127.0.0.1:8000"`

## Phase Plan

## Execution Order (Dependency/Risk Priority)
1. Scope freeze and planning assets.
2. Backend readiness contract (`/ready`, `/ask` readiness gate, related settings).
3. Minimal end-to-end Chainlit slice.
4. Core automated tests and smoke checks.
5. Ingestion hardening (gap repair retries/persistence and portability fixes).
6. Dual quality gate (retrieval + answer quality).
7. Operations packaging and runbook.
8. Rehearsal and freeze.

## Phase 1 (Feb 8-9): Scope Freeze and Planning Assets
Status: Done on February 9, 2026.
1. Persist this plan in `docs/demo_plan.md`.
2. Create `docs/demo_script.md` with 5 scripted prompts and expected citation behavior.
3. Freeze non-goals for demo:
- no multi-turn memory guarantee
- no OpenAI-compatible API work

## Phase 2 (Feb 9-11): Backend Readiness Contract
Status: Done on February 9, 2026.
1. Implement `GET /ready`.
2. Implement `/ask` readiness gating (`503` while warming when enabled).
3. Add readiness-related settings in `api/config.py` and `.env.example`:
- `demo_enforce_ready_gate`
- `demo_request_timeout_seconds`

## Phase 3 (Feb 10-13): Minimal End-to-End Chainlit Slice
Status: Done on February 9, 2026.
1. Add dependencies in `requirements.txt`: `chainlit`, `httpx`.
2. Create `ui/chainlit_app.py`.
3. Create `ui/backend_client.py` for typed HTTP calls to `/ask` and `/ready`.
4. Required UI behavior:
- readiness banner if backend warming
- deterministic timeout message
- deterministic backend unavailable message
- answer + citations rendering with clickable links
5. Launch command:
- `chainlit run ui/chainlit_app.py --host 0.0.0.0 --port 8501`

## Phase 4 (Feb 11-14): Core Tests and Smoke
1. Add test deps: `pytest`, `pytest-cov`.
2. Add tests:
- `tests/test_health_ready.py`
- `tests/test_ask_readiness_gate.py`
- `tests/test_chainlit_citation_render.py`
3. Add smoke runner `scripts/demo_smoke.py`:
- `/health`
- `/ready`
- one `/ask` happy path
- one timeout path
4. Pre-demo check commands:
- `pytest -q`
- `python scripts/demo_smoke.py`

## Phase 5 (Feb 13-18): Ingestion Hardening and Portability
1. Improve gap source checks in `api/auto_ingest.py`:
- retry/backoff around source checks
- cap startup scan with `auto_ingest_gap_repair_scan_max_days`
2. Persist failed checks:
- new migration `sql/2026-04-gap-source-failures.sql`
- table `ingest_gap_source_failures` with `issue_date`, `language`, `attempts`, `last_error`, `last_checked_at`, `next_retry_at`, `resolved_at`
3. Add `scripts/retry_gap_source_checks.py`.
4. Add ingestion reliability tests:
- `tests/test_auto_ingest_gap_retry.py`
5. Replace hardcoded Ollama path in `scripts/run_eval_parallel.py`:
- use `OLLAMA_BIN` env var or `shutil.which("ollama")`

## Phase 6 (Feb 16-20): Dual Quality Gate
1. Keep retrieval gate:
- `scripts/run_eval.py`
- `scripts/check_eval_regression.py`
2. Add answer-quality eval set:
- `data/eval_answer_demo_v1.json` (30 hard queries)
3. Add `scripts/run_answer_eval.py`:
- call `/ask`
- evaluate citation faithfulness and key-content expectations
4. Add `scripts/check_answer_eval_gate.py`:
- pass threshold `>= 0.85`
- no critical misses on scripted prompts
5. Add unified gate runner:
- `scripts/demo_release_gate.sh`

## Phase 7 (Feb 20-24): Operations Packaging and Runbook
1. Add `scripts/demo_ctl.sh` with `start|stop|status|logs|smoke`.
2. Store pids/logs under `logs/demo/`.
3. Add model warmup script `scripts/warm_models.py`.
4. Write `docs/demo_runbook.md`:
- prerequisites
- startup steps
- health checks
- smoke prompts
- rollback and restart actions
- known failures and mitigations

## Phase 8 (Feb 24-26): Rehearsal and Freeze
1. Feb 24: full rehearsal on target machine/network.
2. Feb 25: second rehearsal + restart/failover drill.
3. Feb 25 end-of-day: code freeze except blocker fixes.
4. Feb 26: run pre-demo checklist and warm stack 30-45 minutes before demo.

## Test Scenarios
1. Readiness:
- startup running -> `/ready.ready=false`, `/ask` gated
- startup successful -> `/ready.ready=true`, `/ask` succeeds
- startup failed -> `/ready.status="degraded"`, `/ask` gated
2. Chainlit:
- warming banner shown
- timeout fallback shown
- citation links clickable and prefer `html_url`
3. Gap-repair reliability:
- transient source error retries
- failed checks persisted and later recovered
4. Gates:
- retrieval regression fails on significant drop
- answer gate fails below threshold
5. Smoke:
- one-command smoke validates core demo path

## Acceptance Criteria
1. Chainlit answers all 5 scripted prompts with citations and stable UX states.
2. `/ready` and `/ask` gating prevent warmup partial-data responses.
3. Retrieval regression gate passes vs baseline.
4. Answer eval pass rate >= 85% on demo answer set.
5. `pytest -q` and `python scripts/demo_smoke.py` pass on demo host.
6. Runbook and one-command operations validated in two rehearsals.

## Assumptions and Defaults
- Supported languages remain ES/VA.
- Multi-turn memory is not required for demo readiness.
- Single-host deployment is acceptable.
- OpenAI-compatible API is out of scope for this Chainlit demo.
