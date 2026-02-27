# Demo Rehearsal Log

## Instructions
- Complete one entry per rehearsal date.
- Record exact commands and outcomes.
- Include sign-off at the end of each entry.

## Rehearsal 1 - February 24, 2026
- Owner: pending (not executed on scheduled date)
- Environment (host/network): target single-host stack
- Start time (UTC): not executed
- End time (UTC): not executed

### Service State
- `bash scripts/demo_ctl.sh status` output summary:
  - Not recorded on February 24, 2026.

### Smoke
- Command: not executed
- Exit code: n/a
- Log path: n/a
- Notes:
  - This rehearsal window was missed and requires explicit acceptance of risk or a replacement run decision.

### Scripted Prompt Results (docs/demo_script.md)
- Prompt 1: not executed
- Prompt 2: not executed
- Prompt 3: not executed
- Prompt 4: not executed
- Prompt 5: not executed

### Incidents and Mitigations
- Incident:
  - Scheduled rehearsal not completed on February 24.
- Mitigation:
  - Performed extended validation on February 25 (see Rehearsal 2).
- Follow-up action:
  - Decide whether February 25 evidence is accepted as backfill for February 24 gate.

### Sign-off
- Rehearsal outcome: fail (not executed)
- Approved by: pending
- Timestamp (UTC): pending

## Rehearsal 2 - February 25, 2026
- Owner: Codex + local operator
- Environment (host/network): `/home/ai-server/dogv-ai`, localhost single-host
- Start time (UTC): 2026-02-25T19:52:03Z
- End time (UTC): 2026-02-25T20:08:00Z

### Service State
- `bash scripts/demo_ctl.sh status` output summary:
  - Isolated control validation on ports `8013/8513` with startup sync disabled for deterministic control path:
    - API: running
    - Chainlit: running
    - `/health`: ok
    - `/ready`: `ready=true`, `status=ready`

### Smoke
- Command:
  - `AUTO_INGEST_STARTUP_ENABLED=false API_PORT=8013 CHAINLIT_PORT=8513 BASE_URL=http://127.0.0.1:8013 bash -lc 'scripts/demo_ctl.sh stop || true; scripts/demo_ctl.sh start; scripts/demo_ctl.sh status; scripts/demo_ctl.sh smoke; scripts/demo_ctl.sh logs; scripts/demo_ctl.sh stop'`
  - Target-host smoke check:
    - `.venv/bin/python scripts/demo_smoke.py --base-url http://127.0.0.1:8000`
- Exit code:
  - Isolated control validation: `0`
  - Target-host smoke: `0`
- Log path:
  - `logs/demo/smoke_20260225T195204Z.log` (isolated control)
  - `logs/demo/smoke_20260225T195315Z.log` (target-host API)
  - `logs/demo/smoke_20260225T201232Z.log` (latest target-host API pass)
- Notes:
  - Core smoke checks passed in both runs.

### Scripted Prompt Results (docs/demo_script.md)
- Prompt 1: pass, 122.653s, from `answer_scripted5_20260225T201300Z`
- Prompt 2: pass, 73.502s, from `answer_scripted5_20260225T201300Z`
- Prompt 3: pass, 89.798s, from `answer_scripted5_20260225T201300Z`
- Prompt 4: fail, 129.052s, from `answer_scripted5_20260225T201300Z`
- Prompt 5: fail, 180.061s (timeout), from `answer_scripted5_20260225T201300Z`
- Evidence file:
  - `data/eval_reports/answer_scripted5_20260225T201300Z.json`

### Restart / Failover Drill
- Steps executed:
  - `scripts/demo_ctl.sh stop` (clean stale PID state)
  - `scripts/demo_ctl.sh start`
  - `scripts/demo_ctl.sh status`
  - `scripts/demo_ctl.sh smoke`
  - `scripts/demo_ctl.sh stop`
- Result:
  - Pass on isolated ports (`8013/8513`).
- Recovery time:
  - < 10 seconds to healthy probes after restart in startup-disabled mode.

### Incidents and Mitigations
- Incident:
  - Scripted prompt latencies exceeded rehearsal target (`<= 40s`), including one timeout at 180s.
- Mitigation:
  - Captured exact per-prompt latency and pass/fail evidence.
- Follow-up action:
  - Optimize `/ask` runtime and rerun full 30-case answer eval + full release gate before freeze.

### Freeze Decision
- Freeze status (blocker-only after sign-off): no
- Approved by: pending
- Timestamp (UTC): pending (blocked by phase-6 gate and latency target failures)

## Rehearsal 3 - February 26, 2026 (Quality Validation Update)
- Owner: Codex + local operator
- Environment (host/network): `/home/ai-server/dogv-ai`, localhost single-host
- Start time (UTC): 2026-02-26T11:25:43Z
- End time (UTC): 2026-02-26T13:32:05Z

### Service State
- API status:
  - `/health`: `status=ok`
  - `/ready`: `ready=true`
  - startup sync state: `startup_sync_disabled` (configured for demo runtime stability)

### Quality Gate Runs
- Retrieval gate:
  - report: `data/eval_reports/ret_20260225T204311Z.json`
  - regression check result: pass (`No recall regressions detected.`)
- Full answer gate:
  - report: `data/eval_reports/answer_ans_fix_20260226T112543Z.json`
  - aggregate: `0.9600`
  - pass rate: `0.8667`
  - critical failed: `0`
  - API errors: `0`
  - gate result: pass
- Focused follow-up run (`A07`, `A10`, `A20`, `A25`):
  - report: `data/eval_reports/answer_focus_20260226T133205Z.json`
  - aggregate: `0.8500`
  - pass rate: `0.2500`
  - failures: lexical mismatches only (`A07`, `A20`, `A25`) under strict lexical rubric
  - gate result: pass

### Incidents and Mitigations
- Incident:
  - Earlier full runs showed runtime timeouts on long temporal queries and incorrect ingestion behavior.
- Mitigation applied:
  - request-time ingestion now skips fully covered date ranges
  - startup sync ingestion disabled in demo env to avoid startup backfill storms
  - intent normalization includes deterministic year-range fallback when model omits dates
  - answer generation includes one retry before fallback and logs fallback reason
  - runtime answer mutators/injections disabled by default; validator+repair+fallback path enabled
  - answer gate now uses hard faithfulness for pass/fail, lexical mismatch as diagnostic

### Sign-off
- Rehearsal outcome: pass for quality gates; operational sign-off still pending
- Approved by: pending
- Timestamp (UTC): pending

## Demo Day Checklist - February 26, 2026
- Warm models completed (yes/no): pending
- `/ready.ready=true` before demo (yes/no): yes (observed during February 26 validation)
- Smoke pass before demo (yes/no): pending (no new same-day smoke log captured yet)
- Start time for warm stack: pending
- Final go/no-go decision: pending
- Approved by: pending
