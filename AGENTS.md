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

### Keep it simple (no over-engineering)
- Prefer the simplest correct solution. Don’t add abstractions, frameworks, factories, plugin systems, or “future-proof” layers unless they solve a current, concrete problem.
- Avoid clever or overly generic code. Optimize for readability and maintainability.
- Make changes minimal and local; avoid broad refactors unless required for the task.

### Clean, readable Python
- Follow PEP 8 and keep formatting consistent with the existing repo style (do not introduce new formatters/linters unless requested).
- Use clear, descriptive names (`snake_case` for functions/vars, `PascalCase` for classes).
- Use type hints for public functions and non-trivial logic where it improves clarity. Don’t over-annotate trivial code.
- Prefer small, cohesive functions and straightforward control flow (guard clauses over deep nesting).

### DRY without forcing abstractions
- Remove true duplication, but don’t extract helpers so early that the code becomes harder to read.
- Only factor out shared logic when it repeats meaningfully or reduces bug risk (“rule of three” as a guideline).

### No dead code or redundancy
- Do not leave commented-out code, unused helpers, unused imports, or unused parameters.
- Delete leftovers from experiments and temporary debugging once the change is complete.
- If something might be useful later, rely on version control instead of keeping dead code around.

### Keep files small and focused
- Avoid large “god” modules. Split by responsibility.
- If a file grows beyond ~300–500 lines or a function beyond ~40–60 lines, refactor by extracting well-named modules/helpers.
- Don’t create grab-bag `utils.py` files; prefer domain-specific modules.

### Errors, logs, and side effects
- Validate inputs at boundaries (API, DB, external services). Raise specific errors with actionable messages.
- Don’t swallow exceptions; catch only when you can add context or recover safely.
- Prefer `logging` over `print`. Keep logs informative and not noisy.
- Minimize side effects; keep core logic testable (pure functions where practical).


## Agent tips
- Most scripts assume repo root as CWD (they import `scripts/_path.py`).
- Ingest/build/eval jobs are heavy; confirm before running large ranges.
- Core retrieval/LLM logic lives in `agent/graph.py`; config in `api/config.py` and `.env`.
