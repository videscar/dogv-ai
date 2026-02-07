# AGENTS.md

Project: DOGV AI local assistant (FastAPI + LangGraph + PostgreSQL + Ollama).
This file guides coding agents working in this repo.

## Environment
- Python deps: `pip install -r requirements.txt` (prefer existing `.venv`).
- Copy `.env.example` to `.env` and set `DOGV_DB_DSN`, `DATABASE_URL`, `OLLAMA_BASE_URL`, model names, etc.
- PostgreSQL must have `pgvector` available.

## Database
- Create tables: `python scripts/init_db.py`.
- SQL migrations and index changes live in `sql/` (apply with `psql` in order when needed).
- Reset tables: `python scripts/reset_db.py [--with-cache]`.

## Run the API
- Start server: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- Endpoints: `GET /health`, `GET /issues`, `GET /issues/{issue_id}/documents`, `POST /ask`.

## Ingest and indexing
- Daily or bootstrap: `python scripts/maintain_indices.py --daily` or `--bootstrap`.
- Full pipeline for a date range: `python scripts/ingest_pipeline.py YYYY-MM-DD YYYY-MM-DD es_es,va_va`.
- Chunking/embeddings: `python scripts/build_chunks.py` (long-running).
- Rebuild BM25: `python scripts/rebuild_tsv.py --language va_va --ts-config catalan --batch-size 5000`.

## Evaluation
- Build eval set: `python scripts/build_eval_set.py --size 200 --output data/eval_set_v1.json`.
- Run eval: `python scripts/run_eval.py --input data/eval_set_v1.json --include-nofilter`.
- Regression gate (default KPI: `hybrid,rerank` at `k=5,10`): `python scripts/check_eval_regression.py --report data/eval_reports/<run_id>.json`.

## Code style
- Follow Python best practices: clear naming, small functions, type hints where helpful, and minimal side effects.
- Prefer explicit, readable code over cleverness; keep functions cohesive and testable.
- Keep error handling robust and logs informative but not noisy.
- Maintain consistent formatting (PEP 8); avoid introducing new linting/formatting tools unless requested.
- Update documentation or comments when behavior changes.

## Agent tips
- Most scripts assume repo root as CWD (they import `scripts/_path.py`).
- Ingest/build/eval jobs are heavy; confirm before running large ranges.
- Core retrieval/LLM logic lives in `agent/graph.py`; config in `api/config.py` and `.env`.
