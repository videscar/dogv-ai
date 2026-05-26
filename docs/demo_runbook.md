# Demo Runbook (Chainlit + FastAPI + llama.cpp)

## Prerequisites
- Repo cloned and `.venv` available.
- `.env` configured from `.env.example`.
- PostgreSQL reachable with DOGV data loaded.
- llama.cpp built with CUDA at `~/llama.cpp/build/bin/llama-server`.
- Chat GGUF available at `~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf` (or `CHAT_LLM_MODEL` override).
- Embedding GGUF available at `~/models/bge-m3/bge-m3-f16.gguf` (or `EMBED_LLM_MODEL` override).

## Port Layout (defaults)
- Chat llama-server: `127.0.0.1:8000` (shared; reused if already running)
- Embed llama-server: `127.0.0.1:8001` (managed by `demo_ctl.sh`)
- FastAPI: `0.0.0.0:8088` (managed by `demo_ctl.sh`)
- Chainlit: `0.0.0.0:8501` (managed by `demo_ctl.sh`)

## Startup Steps
1. Ensure previous demo processes are stopped:
   - `bash scripts/demo_ctl.sh stop`
2. Confirm demo runtime ingest settings (recommended for stable startup):
   - `AUTO_INGEST_STARTUP_ENABLED=false`
   - `AUTO_INGEST_STARTUP_REPAIR_GAPS=false`
   - `AUTO_INGEST_STARTUP_PURGE_OLD=false`
3. Start demo services:
   - `bash scripts/demo_ctl.sh start`
   - Starts the embed llama-server if absent.
   - Starts the chat llama-server only if no instance is already healthy on `:8000` (so other projects sharing the chat model are not disturbed).
   - Starts FastAPI on `:8088` and Chainlit on `:8501`.
4. Warm models (single request to each server):
   - `.venv/bin/python scripts/warm_models.py`
5. Check status:
   - `bash scripts/demo_ctl.sh status`

## Health Checks
- Chat llama-server: `curl -sS http://127.0.0.1:8000/health`
- Embed llama-server: `curl -sS http://127.0.0.1:8001/health`
- API health: `curl -sS http://127.0.0.1:8088/health`
- API readiness: `curl -sS http://127.0.0.1:8088/ready`
- Smoke run: `bash scripts/demo_ctl.sh smoke`
- Latest smoke log: `ls -1t logs/demo/smoke_*.log | head -n 1`

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
  - `logs/demo/chat_llm.log`
  - `logs/demo/embed_llm.log`
  - `logs/demo/api.log`
  - `logs/demo/chainlit.log`
  - `logs/demo/smoke_<timestamp>.log`

## Rollback and Restart
- Restart full stack:
  1. `bash scripts/demo_ctl.sh stop`
  2. `bash scripts/demo_ctl.sh start`
  3. `bash scripts/demo_ctl.sh status`
- `stop` only kills processes whose PID files this script owns. If the chat
  llama-server was already running when `start` was called, `stop` leaves it
  alone (other projects may depend on it).
- If readiness remains warming/degraded for too long:
  1. Check API logs for startup sync errors.
  2. Validate DB connectivity and DOGV source reachability.
  3. Re-run smoke after recovery.

## Known Failures and Mitigations
1. `/ready` stays `warming` for long startup windows (when startup sync is enabled):
- Mitigation: for demo runs keep startup sync disabled in `.env`; otherwise wait for ingest completion and monitor logs.

2. `/ask` returns `503` while warming:
- Mitigation: expected when readiness gate is enabled; wait for `/ready.ready=true`.

3. LLM timeout or no response:
- Mitigation: run `scripts/warm_models.py`, verify `curl http://127.0.0.1:8000/v1/models` and `curl http://127.0.0.1:8001/health`, check GPU pressure with `nvidia-smi`, reduce concurrent load.

4. Embed llama-server fails to start:
- Mitigation: confirm `EMBED_LLM_MODEL` GGUF path exists and is readable; check `logs/demo/embed_llm.log` for VRAM/OOM errors; bge-m3 needs ~600MB on GPU.

5. Smoke timeout check unexpectedly passes:
- Mitigation: reduce `--timeout-seconds` in `scripts/demo_smoke.py` command.

6. High latency outliers on some `/ask` prompts:
- Mitigation: keep stack warm, avoid concurrent heavy jobs, and verify no request-time ingestion is triggered for already-covered date ranges.
