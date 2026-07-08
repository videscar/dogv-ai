# Ops units

## Always-on stack (systemd)

Four native units run the full DOGV stack hands-off, restart on failure, and
start at boot. They mirror the launch logic in `scripts/demo_ctl.sh` (which
stays as the dev/manual convenience) but use systemd for lifecycle:

| Unit | Service | Port | GPU |
|------|---------|------|-----|
| `dogv-chat.service`     | vLLM Qwen3.6-27B int4-AutoRound, TP2, MTP | 8000 | 0,1 |
| `dogv-embed.service`    | llama.cpp bge-m3 embeddings               | 8001 | 1   |
| `dogv-api.service`      | FastAPI/uvicorn                           | 8088 | —   |
| `dogv-chainlit.service` | Chainlit UI                               | 8501 | —   |

Ordering: chat + embed come up first and each holds "activating" via an
`ExecStartPost` health poll until `/health` answers, so the API only starts once
its backends are serving; Chainlit starts after the API. Coupling is `Wants=`
(not `Requires=`), so a momentary backend crash+restart doesn't cascade — the API
already degrades gracefully (503 → UI message). `dogv.target` groups all four.

Key gotchas baked into the units (see also `demo_ctl.sh` comments):
- **chat**: `vllm-023-env/bin` on `PATH` for flashinfer's `ninja` JIT; `KillMode=control-group`
  reaps TP workers (else they orphan and pin ~10GB/GPU); `TimeoutStartSec=420`
  for the ~2.5min cold start; an `ExecStartPost` runs `scripts/warm_chat_longctx.py`
  so the first real query doesn't pay the one-time long-prefill kernel compile.
- **embed**: `--verbosity 0` (default logging crashed it ~every 40min under load).

### Install (one-time, requires sudo)

```bash
sudo install -m 0644 ops/dogv-chat.service     /etc/systemd/system/
sudo install -m 0644 ops/dogv-embed.service    /etc/systemd/system/
sudo install -m 0644 ops/dogv-api.service      /etc/systemd/system/
sudo install -m 0644 ops/dogv-chainlit.service /etc/systemd/system/
sudo install -m 0644 ops/dogv.target           /etc/systemd/system/
sudo systemctl daemon-reload
# Enable boot-start for all four, then bring the stack up:
sudo systemctl enable dogv-chat dogv-embed dogv-api dogv-chainlit
sudo systemctl start dogv.target
```

Make sure the manual `demo_ctl.sh` stack is stopped first
(`scripts/demo_ctl.sh stop`) so the ports are free.

### Verify

```bash
systemctl status dogv-chat dogv-embed dogv-api dogv-chainlit
curl -fsS http://127.0.0.1:8088/health   # API up
curl -fsS http://127.0.0.1:8088/ready    # index ready
journalctl -u dogv-chat -f               # watch the slow cold start
```

### Operate

```bash
sudo systemctl restart dogv-api          # restart one service
sudo systemctl stop dogv.target          # stop the whole stack
sudo systemctl start dogv.target         # bring it back up
```

### Reboot test (required before declaring beta-ready)

```bash
sudo reboot
# after it comes back (give chat ~3-4min for cold start):
systemctl status dogv-chat dogv-embed dogv-api dogv-chainlit
curl -fsS http://127.0.0.1:8088/ready
```

## Daily ingest timer (systemd)

Installs a `dogv-daily-ingest` oneshot service triggered Mon–Fri at
09:30 Europe/Madrid by a matching timer. The service runs
`scripts/maintain_indices.py --daily` under user `ai-server`, with the
project venv on PATH and `.env` loaded as `EnvironmentFile`.

The 09:30 schedule is a hedge against DOGV's undocumented daily
publication window. After a week of real runs, check
`journalctl -u dogv-daily-ingest -n 200` — if most runs pick up that
day's issue immediately, keep 09:30; if many runs are catching the
issue only on the *next* day, edit the timer to
`OnCalendar=Mon..Fri *-*-* 11:00:00 Europe/Madrid` and
`systemctl daemon-reload && systemctl restart dogv-daily-ingest.timer`.

### Install (one-time, requires sudo)

```bash
sudo install -m 0644 ops/dogv-daily-ingest.service /etc/systemd/system/
sudo install -m 0644 ops/dogv-daily-ingest.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dogv-daily-ingest.timer
```

### Verify

```bash
systemctl status dogv-daily-ingest.timer
systemctl list-timers --all | grep dogv-daily-ingest
# trigger an immediate run to confirm the service works:
sudo systemctl start dogv-daily-ingest.service
journalctl -u dogv-daily-ingest -f
```

### Prerequisites

- The ingest catch-up has completed successfully (see plan Step 1c).
- `AUTO_INGEST_ENABLED` can stay `false` in `.env` — the timer is the
  scheduler. If you also want runtime gap-repair from the FastAPI
  background loop, flip it to `true` after the timer is verified.
- The chat + embed llama-servers must be reachable by the time the
  daily ingest runs (classify_documents calls the LLM). Either keep
  the demo_ctl stack up 24/7, or add an `ExecStartPre` to the service
  that brings the stack up first.
