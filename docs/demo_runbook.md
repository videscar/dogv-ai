# Demo Runbook (Chainlit + FastAPI)

## Prerequisites
- Repo cloned and `.venv` available.
- `.env` configured from `.env.example`.
- PostgreSQL reachable with DOGV data loaded.
- Ollama running with configured LLM and embedding models.

## Startup Steps
1. Ensure previous demo processes are stopped:
   - `bash scripts/demo_ctl.sh stop`
2. Warm models:
   - `.venv/bin/python scripts/warm_models.py`
3. Confirm demo runtime ingest settings (recommended for stable startup):
   - `AUTO_INGEST_STARTUP_ENABLED=false`
   - `AUTO_INGEST_STARTUP_REPAIR_GAPS=false`
   - `AUTO_INGEST_STARTUP_PURGE_OLD=false`
4. Start demo services:
   - `bash scripts/demo_ctl.sh start`
5. Check status:
   - `bash scripts/demo_ctl.sh status`

## Health Checks
- API health:
  - `curl -sS http://127.0.0.1:8000/health`
- API readiness:
  - `curl -sS http://127.0.0.1:8000/ready`
- Smoke run:
  - `bash scripts/demo_ctl.sh smoke`
- Latest smoke log:
  - `ls -1t logs/demo/smoke_*.log | head -n 1`

## Smoke Prompts
Use the scripted prompts from:
- `docs/demo_script.md`

Rules:
- One new chat per prompt.
- Keep prompts unchanged.
- Log pass/fail and latency.

## Logs and Monitoring
- Snapshot logs:
  - `bash scripts/demo_ctl.sh logs`
- Follow logs live:
  - `bash scripts/demo_ctl.sh logs --follow`
- Log files:
  - `logs/demo/api.log`
  - `logs/demo/chainlit.log`
  - `logs/demo/smoke_<timestamp>.log`

## Rollback and Restart
- Restart full stack:
  1. `bash scripts/demo_ctl.sh stop`
  2. `bash scripts/demo_ctl.sh start`
  3. `bash scripts/demo_ctl.sh status`
- If readiness remains warming/degraded for too long:
  1. Check API logs for startup sync errors.
  2. Validate DB connectivity and DOGV source reachability.
  3. Re-run smoke after recovery.

## Known Failures and Mitigations
1. `/ready` stays `warming` for long startup windows (when startup sync is enabled):
- Mitigation: for demo runs keep startup sync disabled in `.env`; otherwise wait for ingest completion and monitor logs.

2. `/ask` returns `503` while warming:
- Mitigation: expected when readiness gate is enabled; wait for `/ready.ready=true`.

3. Ollama timeout or no response:
- Mitigation: run `scripts/warm_models.py`, check Ollama service health, reduce concurrent load.

4. Smoke timeout check unexpectedly passes:
- Mitigation: reduce `--timeout-seconds` in `scripts/demo_smoke.py` command.

5. High latency outliers on some `/ask` prompts:
- Mitigation: keep stack warm, avoid concurrent heavy jobs, and verify no request-time ingestion is triggered for already-covered date ranges.
