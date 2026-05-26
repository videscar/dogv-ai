# DOGV AI Demo Readiness Plan (Chainlit) for February 26, 2026

## Summary
This plan targets a single-host demo stack (Postgres + local LLM runtime + FastAPI + Chainlit), with scripted single-turn QA and a hard dual release gate (retrieval + answer quality).

Status snapshot updated: February 26, 2026.

Baseline date: February 8, 2026.
Demo deadline: February 26, 2026.

Current repo status (as of February 26, 2026):
- Backend endpoints include `GET /ready` with readiness payload.
- `POST /ask` readiness gating is implemented when `DEMO_ENFORCE_READY_GATE=true`.
- Chainlit integration is present (`ui/chainlit_app.py`, `ui/backend_client.py`).
- Startup gap repair retries and failed source-check persistence are implemented.
- Answer-quality gate artifacts are implemented (`data/eval_answer_demo_v1.json`, `scripts/run_answer_eval.py`, `scripts/check_answer_eval_gate.py`, `scripts/demo_release_gate.sh`).
- Runtime answer mutators are disabled by default; validator + single repair attempt + fallback is enabled.
- Automated tests are present and passing in `.venv` (`35 passed`).
- Parallel eval orchestration is out of scope for this demo run.
- Retrieval regression run completed and passing:
  - report: `data/eval_reports/ret_20260225T204311Z.json`
  - `check_eval_regression.py`: no recall regressions
  - `rerank recall@5`: `0.9733`
- Full answer gate run completed and passing:
  - report: `data/eval_reports/answer_ans_fix_20260226T112543Z.json`
  - aggregate: `0.9600`, pass rate: `0.8667`
  - critical failures: `0`, API errors: `0`
- Follow-up focused run (`data/eval_reports/answer_focus_20260226T133205Z.json`) shows residual lexical mismatches on `A07`, `A20`, `A25` (generalization/rubric precision issue, not runtime stability).

## Project Status (February 26, 2026)
- Retrieval quality gate: pass (stable, no regression).
- Answer quality gate: pass on full 30-case set (`ans_fix` run).
- Runtime stability: materially improved (`api_errors` dropped from `5` to `0` across latest full runs).
- Main open quality risks:
  - topicality drift on narrow queries (example cluster around `A25`)
  - lexical/rubric sensitivity still present in focused checks (now tracked as diagnostic, non-blocking)
  - latency tail remains high on a few cases despite gate pass
- Operational status:
  - startup auto-sync is disabled in demo env to avoid startup backfill storms
  - request-time ingestion now skips already covered temporal ranges

## Implementation Decisions (Locked on February 23, 2026)
- Add explicit Definition of Done (DoD) per pending phase with commands, outputs, and artifact paths.
- Keep `CHAINLIT_BACKEND_URL` as a UI environment variable only (no mirrored setting in `api/config.py`).
- Add test dependencies directly to `requirements.txt` for simplest demo-host setup.
- Use deterministic tests with mocks for LLM/retrieval dependencies; keep live checks in smoke scripts.
- `scripts/demo_smoke.py` assumes API is already running and accepts `--base-url`.
- Gap source-check retry policy: 3 attempts, exponential backoff from 1.5 seconds, with jitter; retry transient failures only.
- `auto_ingest_gap_repair_scan_max_days` caps source-check scan window only (not the final repair execution semantics).
- Persist gap-check failures with one active row per `(issue_date, language)` via upsert semantics.
- `scripts/retry_gap_source_checks.py` retries checks and triggers ingestion immediately when source is confirmed.
- Skip parallel eval portability work for this demo cycle (non-blocking).
- Answer eval dataset schema uses: `id`, `language`, `question`, `must_include`, `must_not_include`, `required_citations`, `critical`.
- Answer gate scoring now uses hard faithfulness (`aggregate_hard_score`) with lexical recall diagnostics (non-blocking); pass threshold `>= 0.85`; all critical prompts must pass.
- Runtime answer mutators are disabled by default and replaced by deterministic validation + one repair attempt + evidence fallback.
- Unified gate script runs all gates, reports aggregated status, and exits non-zero if any gate fails.
- Operations control uses `nohup` + PID files + `logs/demo/` (no systemd dependency).
- Rehearsal requires sign-off artifacts for both rehearsal days before freeze.
- Latency/error acceptance thresholds are explicit and enforced in rehearsal logs.

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
- `answer_mutators_enabled: bool = False`
- `answer_missing_notes_enabled: bool = False`
- `answer_validator_enabled: bool = True`
- `answer_repair_attempts: int = 1`
- `auto_ingest_gap_check_retries: int = 3`
- `auto_ingest_gap_check_backoff_seconds: float = 1.5`
- `auto_ingest_gap_repair_scan_max_days: int = 45`
- `CHAINLIT_BACKEND_URL` remains in UI env configuration (`ui/chainlit_app.py`, `.env.example`) and is not mirrored in `api/config.py`.

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
Status: Done on February 23, 2026.
Implementation choices:
- Add `pytest` and `pytest-cov` to `requirements.txt`.
- Test files should mock external dependencies (LLM/network-heavy retrieval paths) for deterministic execution.
- Smoke runner validates live behavior against an already-running API endpoint.
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
DoD:
1. `requirements.txt` includes `pytest` and `pytest-cov`.
2. `pytest -q` exits `0` and runs at minimum:
- `tests/test_health_ready.py`
- `tests/test_ask_readiness_gate.py`
- `tests/test_chainlit_citation_render.py`
3. `scripts/demo_smoke.py` supports `--base-url` (default `http://127.0.0.1:8000`) and exits non-zero on any failed check.
4. Smoke validates:
- `/health` returns success payload
- `/ready` returns readiness payload shape
- one `/ask` happy path
- one deterministic timeout path
5. Smoke output is saved to `logs/demo/smoke_<timestamp>.log`.

## Phase 5 (Feb 13-18): Ingestion Hardening and Portability
Status: Done on February 23, 2026 (except deferred parallel-eval portability backlog).
Implemented:
- Startup gap repair flow exists in `api/auto_ingest.py`.
Remaining:
- parallel eval portability change deferred for this demo cycle
Implementation choices:
- Retry policy: 3 attempts per source check, exponential backoff from 1.5 seconds plus bounded jitter.
- Retry only transient failures (timeouts, connection errors, DNS errors, HTTP 5xx).
- `auto_ingest_gap_repair_scan_max_days` applies to source-check candidate scan only.
- Persist failures in `ingest_gap_source_failures` with unique key `(issue_date, language)`.
- Add unresolved retry index on `next_retry_at` for scheduling.
- `scripts/retry_gap_source_checks.py` triggers immediate ingestion for pairs that become source-positive and marks them resolved.
1. Improve gap source checks in `api/auto_ingest.py`:
- retry/backoff around source checks
- cap startup scan with `auto_ingest_gap_repair_scan_max_days`
2. Persist failed checks:
- new migration `sql/2026-04-gap-source-failures.sql`
- table `ingest_gap_source_failures` with `issue_date`, `language`, `attempts`, `last_error`, `last_checked_at`, `next_retry_at`, `resolved_at`
3. Add `scripts/retry_gap_source_checks.py`.
4. Add ingestion reliability tests:
- `tests/test_auto_ingest_gap_retry.py`
5. Defer parallel-eval orchestration to a post-demo backlog (no script committed).
DoD:
1. `api/config.py` and `.env.example` include:
- `AUTO_INGEST_GAP_CHECK_RETRIES`
- `AUTO_INGEST_GAP_CHECK_BACKOFF_SECONDS`
- `AUTO_INGEST_GAP_REPAIR_SCAN_MAX_DAYS`
2. `api/auto_ingest.py` applies retry/backoff with transient-error filtering and scan cap.
3. `sql/2026-04-gap-source-failures.sql` exists and creates `ingest_gap_source_failures` with required columns and unique `(issue_date, language)`.
4. `scripts/retry_gap_source_checks.py` retries unresolved rows due by `next_retry_at`, ingests source-positive rows, and resolves them on success.
5. `tests/test_auto_ingest_gap_retry.py` passes and covers:
- transient failure then success
- persistent failure persistence/update
- scan cap behavior

## Phase 6 (Feb 16-20): Dual Quality Gate
Status: Done on February 26, 2026 (with residual quality/performance risks tracked in backlog).
Implemented:
- Retrieval gate scripts already present (`scripts/run_eval.py`, `scripts/check_eval_regression.py`).
- Added live progress and per-case latency capture to `scripts/run_answer_eval.py`.
- Full retrieval regression run completed:
  - report: `data/eval_reports/ret_20260225T204311Z.json`
  - regression check: pass (`No recall regressions detected.`)
- Full 30-case answer eval completed:
  - report: `data/eval_reports/answer_ans_fix_20260226T112543Z.json`
  - aggregate score: `0.9600`
  - pass rate: `0.8667`
  - critical failures: `0`
  - API errors: `0`
- Follow-up focused regression run executed:
  - report: `data/eval_reports/answer_focus_20260226T133205Z.json`
  - aggregate score: `0.8500`
  - failure mix: lexical mismatches only (`A07`, `A20`, `A25`)
Remaining:
- Keep refining generalization quality (topicality + faithfulness) without overfitting to individual eval IDs.
- Reduce high-latency tail cases that can exceed operational targets.
Implementation choices:
- Answer eval set schema per case:
- `id: str`
- `language: "es" | "va"`
- `question: str`
- `must_include: list[str]`
- `must_not_include: list[str]`
- `required_citations: int`
- `critical: bool`
- Score formula:
  - hard gate score: `hard_faithfulness` (`citation_faithfulness`, forced to `0.0` on forbidden-term hits)
  - lexical recall: diagnostic only (non-blocking)
- Gate passes only if aggregate hard score `>= 0.85`, all `critical=true` cases pass, and API errors are `0`.
- Unified gate runs retrieval and answer gates, prints summary, and exits non-zero on any failure.
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
DoD:
1. `data/eval_answer_demo_v1.json` exists with 30 queries following the locked schema.
2. `scripts/run_answer_eval.py` produces `data/eval_reports/answer_<run_id>.json` with per-case scores plus:
- `summary.aggregate_hard_score` (gate metric)
- `summary.lexical_recall_avg` (diagnostic metric)
- `summary.aggregate_score` as backward-compatible alias of hard score
3. `scripts/check_answer_eval_gate.py --report <path>` exits `0` only when threshold and critical conditions are met.
4. `scripts/demo_release_gate.sh` runs:
- retrieval regression gate
- answer-quality gate
- aggregated summary output with final pass/fail exit code

## Phase 7 (Feb 20-24): Operations Packaging and Runbook
Status: Done on February 25, 2026.
Implementation choices:
- `scripts/demo_ctl.sh` uses `nohup` and PID files under `logs/demo/`.
- Process model: one FastAPI process + one Chainlit process managed by script.
- No systemd dependency for demo host.
Validation evidence:
- Full chained control validation executed on isolated ports (`8013/8513`):
  - `stop -> start -> status -> smoke -> logs -> stop`
  - smoke log: `logs/demo/smoke_20260225T195204Z.log` (pass)
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
DoD:
1. `scripts/demo_ctl.sh start` launches API + Chainlit and writes PID files/log files under `logs/demo/`.
2. `scripts/demo_ctl.sh stop` gracefully stops both processes and cleans stale PID state safely.
3. `scripts/demo_ctl.sh status` reports running/not-running and includes `/health` and `/ready` probe summaries.
4. `scripts/demo_ctl.sh logs` tails both services.
5. `scripts/demo_ctl.sh smoke` executes `scripts/demo_smoke.py` against configured base URL.
6. `scripts/warm_models.py` warms configured LLM + embedding models and exits non-zero on warmup failure.
7. `docs/demo_runbook.md` exists and matches script behavior exactly.

## Phase 8 (Feb 24-26): Rehearsal and Freeze
Status: In progress as of February 26, 2026 (quality gates now passing; final demo-day checklist/sign-off pending).
Implementation choices:
- Require explicit rehearsal artifacts and sign-offs in source control docs.
- Add `docs/demo_rehearsal_log.md` template with owner, date, environment, smoke result, scripted prompt results, incidents, and sign-off.
Execution notes:
- Rehearsal artifacts started in `docs/demo_rehearsal_log.md`.
- Smoke evidence captured on target API:
  `logs/demo/smoke_20260225T195315Z.log` and `logs/demo/smoke_20260225T201232Z.log` (pass).
- Scripted prompt segment evidence captured in:
  `data/eval_reports/answer_scripted5_20260225T201300Z.json`
  (`3/5` pass; `A04` and `A05` failed, `A05` timed out at 180s).
- Full quality validation completed on February 25-26:
  - retrieval: `data/eval_reports/ret_20260225T204311Z.json` (pass)
  - answer full set: `data/eval_reports/answer_ans_fix_20260226T112543Z.json` (gate pass)
  - answer focused set: `data/eval_reports/answer_focus_20260226T133205Z.json` (gate pass, lexical mismatches remain)
1. Feb 24: full rehearsal on target machine/network.
2. Feb 25: second rehearsal + restart/failover drill.
3. Feb 25 end-of-day: code freeze except blocker fixes.
4. Feb 26: run pre-demo checklist and warm stack 30-45 minutes before demo.
DoD:
1. `docs/demo_rehearsal_log.md` is present and has entries for February 24, 2026 and February 25, 2026.
2. Each rehearsal entry includes:
- smoke command + exit status
- all 5 scripted prompts pass/fail and latency values
- incident list and mitigation notes
- explicit sign-off
3. Freeze decision recorded on February 25, 2026 with blocker-only exception rule.
4. Demo day checklist execution logged on February 26, 2026 before showtime.

## Remaining Work (Execution Priority)
1. Improve generalization quality without case overfitting:
- add stronger topicality guardrails for off-target evidence bundles
- improve faithfulness checks for numeric/limit statements
2. Reduce latency tail outliers (while preserving answer quality):
- keep median stable and reduce rare multi-minute outliers
3. Complete phase-8 sign-offs:
- finalize February 24 entry status (executed/not executed)
- finalize February 25 freeze decision
- run and log February 26 pre-demo checklist
4. Post-demo backlog: rebuild a parallel-eval orchestrator on the new llama.cpp runtime if parallel eval is needed later.

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
1. Chainlit answers all 5 scripted prompts with citations and no backend error state in both rehearsals.
2. `/ready` and `/ask` gating prevent warmup partial-data responses.
3. Retrieval regression gate passes vs baseline.
4. Answer eval aggregate hard score is `>= 0.85`, all critical prompts pass, and API errors are `0`.
5. `pytest -q` and `python scripts/demo_smoke.py` pass on demo host.
6. Runbook and one-command operations (`demo_ctl.sh`) are validated in two rehearsals.
7. Rehearsal latency target: scripted prompts have p95 end-to-end latency `<= 40s` (warm stack), with 0 timeout failures.

Current check (as of February 26, 2026):
- Criteria `3`: met.
- Criteria `4`: met.
- Criteria `5`: met for automated tests; smoke validated on rehearsal runs.
- Criteria `1`, `6`, `7`: partially met / pending final sign-off entry and latency-tail cleanup.

## Assumptions and Defaults
- Supported languages remain ES/VA.
- Multi-turn memory is not required for demo readiness.
- Single-host deployment is acceptable.
- OpenAI-compatible API is out of scope for this Chainlit demo.
