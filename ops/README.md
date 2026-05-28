# Ops units

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
