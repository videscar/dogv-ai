# One-off scripts

Archived scripts that performed a **single historical corpus-repair operation**.
They are kept for provenance (how the corpus reached its current state), not for
routine use — the behaviours they backfilled are now handled inline by the daily
ingest pipeline. Run from the repository root.

| Script | What it did (one-off) |
|---|---|
| `recover_bis_editions.py` | Recovered ~6.7k dispositions dropped because `date-sumario` only returned the *bis* edition. Prevention is now wired into the ingest pipeline via `api/bis_editions.py`. |
| `reprocess_html_backfill.py` | Re-extracted clean HTML-first text over a date range after the HTML-first ingestion switch (corpus reached ~88% HTML, 0 empty). |
| `retry_gap_source_checks.py` | Retried source-availability checks for gap rows and triggered immediate ingestion once a source became confirmed (auto-ingest machinery, off by default). |

`_path.py` puts the repository root on `sys.path` so these can be launched directly.
