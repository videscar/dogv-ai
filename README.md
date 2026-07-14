# DOGV AI

**English** · [Español](README.es.md)

[![CI](https://github.com/videscar/dogv-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/videscar/dogv-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

Self-hosted RAG assistant for the DOGV (*Diari Oficial de la Generalitat
Valenciana*, the official gazette of the Valencian regional government): public
employment, grants/subsidies/awards and scholarships. Automated daily ingestion,
multi-lane hybrid search over PostgreSQL, and answers with verifiable citations —
or an explicit abstention when there is no evidence. The whole stack runs locally
(2× consumer GPUs); no data leaves the machine.

- **API**: FastAPI (`:8088`) — `POST /ask` and `POST /ask/stream` (SSE with per-stage progress).
- **Orchestration**: LangGraph (`agent/graph.py`), one node per pipeline stage.
- **Storage**: PostgreSQL with `pgvector` (embeddings) + `tsvector` (BM25).
- **Chat LLM**: Qwen3.6-27B (int4 AutoRound) served with **vLLM 0.23** — tensor-parallel 2,
  MTP speculative decoding, fp8 KV cache, prefix caching (`ops/dogv-chat.service`).
- **Embeddings**: bge-m3 (GGUF) served with **llama.cpp** as a separate process
  (`ops/dogv-embed.service`).
- **UI**: Chainlit (`:8501`) with progress streaming.
- Languages: Spanish and Valencian (BM25 with the `catalan` text-search config and
  fallback to `spanish`).

## The pipeline at a glance

```mermaid
flowchart TD
    Q([Question + history]) --> CTX["contextualize · multi-turn rewrite"]
    CTX --> INT["intent · language, kind, entities, dates"]
    INT --> TG{"temporal_guard"}
    TG -->|invalid time frame| REJ([explicit rejection])
    TG -->|ok| OI["online_ingest · freshness (off: daily timer owns it)"]
    OI --> RET
    subgraph RET["retrieve · multi-lane hybrid retrieval"]
        V["chunk vectors"] --> RRF
        B["chunk BM25 · broad + strict + facets + PRF"] --> RRF
        TB["title BM25"] --> RRF
        TV["title vectors"] --> RRF
        H["HyDE · only when the base pool is low-confidence"] -.-> RRF
        RRF["weighted RRF fusion · semantic anchors · filter relaxation"]
    end
    RET --> DEC{"names a specific norm<br/>or empty pool?"}
    DEC -->|yes| BF["backfill · resolve on the DOGV portal<br/>and ingest on the fly"]
    BF -->|new document fetched| RET
    BF -->|already in corpus / unresolved| RR
    DEC -->|no| RR["rerank · LLM top-k + enumeration series<br/>+ stale-edition suppression"]
    RR --> RD["read · keyword-window chunks + citation floor<br/>+ quote re-grounding + full-doc"]
    RD --> ANS["answer · deterministic synthesis + numeric claim guard"]
    ANS --> OUT([cited answer])
    ANS --> ABS([explicit abstention when evidence is insufficient])
```

## Architecture

### 1) Ingestion (daily, systemd timer)
- `scripts/sumario_ingest.py`: downloads the daily gazette summary and upserts issues
  (capturing *bis* editions so no dispositions are lost on double-edition dates).
- `scripts/extract_documents.py`: creates documents (dispositions) per issue.
- `scripts/download_assets.py` / `download_html.py`: local PDF/HTML cache.
- `scripts/extract_text.py`: clean HTML-first text with PDF fallback.
- `scripts/classify_documents.py`: LLM classification into `doc_kind`/`doc_subkind`.
- `scripts/build_chunks.py`: chunking on the real bge-m3 tokenizer
  (300–500 tokens, 80 overlap) + chunk, title and document-level embeddings + `tsvector`.
- Warm window: rolling 24 months (`scripts/maintain_indices.py` purges older content).

Key tables: `dogv_issues`, `dogv_documents`, `rag_chunk` (embedding + tsv),
`rag_title`, `rag_doc` (document-level embedding). Migrations live in `sql/`.

### 2) Query pipeline (LangGraph — `agent/graph.py`)
1. **contextualize** — rewrites follow-up turns into a standalone query using the
   conversation history (the server is stateless: the client sends `history` with
   each request).
2. **intent** — the LLM extracts language, `doc_kind`/`doc_subkind`, entities and dates.
3. **temporal_guard** — validates/filters the question's time frame.
4. **online_ingest** — (optional) freshness ingest; in production freshness is owned
   by the daily timer and this path is disabled.
5. **retrieve** — multi-lane hybrid retrieval:
   - lanes: chunk vectors, chunk BM25 (broad + strict + per-facet queries
     + PRF expansion), title BM25 and title vectors;
   - **confidence-gated HyDE**: the hypothetical document is only generated when the
     base pool's RRF margin is low, and never for queries citing a specific norm;
   - **weighted RRF fusion** with deterministic tiebreaking, adaptive pool expansion
     when the margin is flat, and a filter-relaxation ladder
     (doc_kind → language → dates);
   - **semantic anchors**: a document in the top-N of a semantic lane is guaranteed a
     slot in the fused pool (prevents correlated BM25 lanes from evicting it).
6. **backfill** — *on-demand historical fetch*: if the question cites a norm outside
   the 24-month window, it is resolved against the DOGV portal search, ingested on
   the fly, and retrieval re-runs only if something new was actually fetched.
7. **rerank** — LLM re-ranking of the top candidates; enumeration queries ("list all
   the…") widen the pool with the month+category series via SQL; **stale sibling
   editions** of recurring publications (near-identical by document-embedding cosine)
   are suppressed so only the most recent edition is read.
8. **read** — per-document chunk selection with **keyword-window** truncation (not
   prefix truncation), a *citation floor* (every selected document contributes a
   usable quote), re-grounding of non-verbatim LLM quotes onto the source chunk, and
   full-document reads when the evidence demands it.
9. **answer** — deterministic synthesis (thinking off, temperature 0) + a validator
   with a **unit-aware numeric claim guard** (every monetary/percentage figure must
   exist in the cited source), a conditional repair retry, and forced citation of the
   target norm when the question names one. Insufficient evidence → explicit abstention.

### 3) Serving (systemd)
Four units + a timer, with health-check-ordered startup (see `ops/README.md`):
`dogv-chat` (vLLM :8000) → `dogv-embed` (llama.cpp :8001) → `dogv-api` (:8088) →
`dogv-chainlit` (:8501), grouped under `dogv.target`; `dogv-daily-ingest.timer`
keeps the corpus current. `scripts/demo_ctl.sh` reproduces the same stack manually
for development.

## Demo

<!-- To restore the demo: add assets/demo.gif and replace the line below with: ![Chainlit UI streaming per-stage progress](assets/demo.gif) -->
_A screen recording of the Chainlit UI streaming per-stage `/ask/stream` progress is coming soon._

## Configuration

Use `.env.example` as the template. The ~15 variables that actually matter:

| Variable | What it controls |
|---|---|
| `DATABASE_URL` / `DOGV_DB_DSN` | PostgreSQL (SQLAlchemy / CLI) |
| `LLM_BASE_URL`, `LLM_MODEL` | OpenAI-compatible chat server (vLLM) |
| `EMBED_BASE_URL`, `EMBED_MODEL` | Embedding server (llama.cpp) |
| `ASK_LANES` | Active retrieval lanes (`vector,bm25,title`) |
| `ASK_MAX_DOCS`, `ASK_READ_MAX_DOCS` | Fused pool size / read-set size |
| `ASK_HYDE_ENABLED` | Confidence-gated HyDE |
| `ASK_SEMANTIC_ANCHOR_ENABLED` | Guaranteed slots for semantic anchors |
| `ASK_EDITION_RECENCY_ENABLED` | Stale sibling-edition suppression |
| `ANSWER_CLAIM_GUARD_MODE` | Numeric claim guard (`unit_aware_strict` in production) |
| `ASK_CONTEXTUALIZE_ENABLED` | Multi-turn rewriting |
| `BACKFILL_ENABLED` | On-demand historical fetch |
| `AUTO_INGEST_ENABLED` | Auto-ingest from the API (OFF; the timer owns freshness) |
| `WARM_INDEX_MONTHS` | Rolling corpus window (24) |

The full variable reference, with defaults and rationale, lives in
[`docs/CONFIG.md`](docs/CONFIG.md).

## Quick start

```bash
# Index bootstrap (24 months) or daily ingest
.venv/bin/python scripts/maintain_indices.py --bootstrap   # or --daily

# API
uvicorn api.main:app --host 0.0.0.0 --port 8088

# UI (separate terminal)
chainlit run ui/chainlit_app.py --host 0.0.0.0 --port 8501

# Full manual stack (vLLM chat + llama.cpp embed + API + UI)
bash scripts/demo_ctl.sh start
```

## Run without local GPUs

The only service you must host yourself is **PostgreSQL + pgvector**; a
`docker-compose.yml` is provided for it (it also enables `unaccent` and defines
the `catalan` text-search config the Valencian lane needs):

```bash
docker compose up -d db
.venv/bin/python scripts/init_db.py                       # create tables
DB="postgresql://dogv_ai:dogv_ai@localhost:5432/dogv_ai"
for f in sql/*.sql; do psql "$DB" -f "$f"; done            # indexes + pgvector DDL
```

The chat and embedding models are reached over plain **OpenAI-compatible HTTP**,
so you can point them at any endpoint instead of the local 2×GPU stack — a hosted
API, another machine, or a small local runtime. Set the host root (the client
appends `/v1/chat/completions` and `/v1/embeddings`):

```bash
export LLM_BASE_URL=https://your-chat-host          # -> /v1/chat/completions
export LLM_MODEL=your-chat-model
export EMBED_BASE_URL=https://your-embed-host        # -> /v1/embeddings
export EMBED_MODEL=bge-m3
```

Honest caveats about this mode:
- The corpus vectors are **bge-m3, 1024-dim** (`EMBEDDING_DIM=1024`, baked into the
  `rag_chunk`/`rag_doc` schema). For a like-for-like run the embedding endpoint must
  also be bge-m3; a different embedder means re-embedding the corpus and changing the
  vector dimension in `sql/`.
- The published answer-quality numbers were measured with **Qwen3.6-27B** as the chat
  model — a weaker endpoint will score lower.
- You still need to ingest a corpus (`scripts/maintain_indices.py --bootstrap`), which
  calls the embedding endpoint for every chunk, so a remote embedder makes the initial
  bootstrap slower.

## Endpoints

- `GET /health` — status + index freshness + the exact commit/config being served.
- `GET /ready` — readiness gate for traffic.
- `GET /issues`, `GET /issues/{issue_id}/documents` — corpus browsing.
- `POST /ask` — `{question, history?, debug?}` → `{answer, citations, debug?}`.
- `POST /ask/stream` — SSE variant: one `stage` event per graph node, then `result`.

## Evaluation

> 📊 **The full evaluation story** — every shipped fix with before/after gated
> numbers, the thinking-ON experiment that was *rejected for reproducibility*, the
> retrieval ceiling, and the tuned-vs-holdout generalization gap — is in
> **[docs/EVALS.md](docs/EVALS.md)**.

The hard suite (`data/eval_v2/`, 100 questions, 50/50 Valencian/Spanish: clean,
vague, colloquial, wrong-reference, multi-hop, annex and out-of-scope) scores
**retrieval** and **answer quality** separately, with a hard gate that zeroes any
answer containing a material factual error. Every run is tied to the exact commit
that produced it (a `.meta.json` sidecar + `/health`). Details:
`data/eval_v2/README.md` and the reports in `data/eval_v2/*.md`.

Latest results (100Q, full re-run on master `03ab7db` — clean tree — 2026-07-09,
production config):

| Metric | Value |
|---|---|
| Overall score (with factual gate) | **0.706** (June baseline on same config: 0.622) |
| Faithfulness to evidence | 0.978 |
| Critical error rate | 1.1% (June: 6.7%) |
| Out-of-scope abstention | 10/10 |
| Frozen holdout (29Q incl. out-of-scope, never tuned against) | 0.690 |
| Retrieval (rerank) R@10 | 0.744 |
| MRR (rerank) | 0.582 |
| External tester regression set (30Q) | 30/30 |

**Known limitations (measured, not theoretical):** ~25% of the hard questions never
retrieve the gold document (an embeddings ceiling, dominated by vague queries —
R@10 0.44 — and by Valencian, which trails Spanish by ~7 points); median `/ask`
latency is ~50–60 s (a multi-stage pipeline on a local 27B); no OCR for scanned PDFs.

### One command: the full regression sweep

`scripts/run_all_regressions.sh` runs **every** regression set in sequence against a
live API (default prod `:8088`) and writes all outputs under
`data/regression_reports/<timestamp>/`:

```bash
scripts/run_all_regressions.sh                       # prod :8088
scripts/run_all_regressions.sh http://127.0.0.1:8090 # a dev API
```

| # | Suite | Size | What it checks |
|---|---|---|---|
| 1 | Identifier probes | 12 | code/BDNS/ref exact-match retrieval (gold-cited) |
| 2 | Tester / Raul regression | 30 | answered (not abstained) + correct norm citation |
| 3 | Retrieval eval | 90 | recall@k per stage (in-process) |
| 4 | eval_v2 citation + abstention | 100 | gold-cited / correctly-abstained (fast signal, **not** the LLM-judge score) |

**Latest full sweep — prod, master `81b1352`, 2026-07-14:** identifier probes **12/12** ·
tester/Raul **30/30** · retrieval rerank **R@10 0.733** (at the documented ~0.74 ceiling) ·
eval_v2 citation **67/100 raw** (misses concentrate in the known-hard vague/wrong-ref
categories — the embeddings ceiling; the abstain regex undercounts, the LLM-judge measures
out-of-scope abstention at 10/10). The identifier layer fires on 0/100 eval_v2 questions, so
it neither helped nor regressed that set.

Individual suites / the authoritative answer-quality pipeline:

```bash
# All suites at once (the sweep above)
scripts/run_all_regressions.sh http://127.0.0.1:8088

# Identifier-layer probe set (code/BDNS/ref exact-match retrieval)
.venv/bin/python scripts/oneoff/run_identifier_probes.py --api http://127.0.0.1:8088

# Authoritative end-to-end answer quality (LLM judge) + aggregation
.venv/bin/python eval_v2/collect_answers.py --base-url http://127.0.0.1:8088
.venv/bin/python eval_v2/score_answers.py <judgments.jsonl> <answers.jsonl> data/eval_v2/reports/answer_metrics.json

# Retrieval (recall/MRR/nDCG per stage) + regression gate
.venv/bin/python scripts/run_eval.py --input data/eval_v2/retrieval_input.json
.venv/bin/python eval_v2/retrieval_metrics.py data/eval_reports/<run_id>.json
.venv/bin/python scripts/check_eval_regression.py --report data/eval_reports/<run_id>.json

# eval_v2 citation + abstention (fast retrieval-into-answer regression signal)
.venv/bin/python scripts/eval_v2_citation_check.py --api http://127.0.0.1:8088

# External tester regression set (30Q against production)
.venv/bin/python scripts/run_tester_regression.py --api http://localhost:8088
```

## License

[MIT](LICENSE).
