# On-demand historical fetch — implementation plan

**Status:** proposed (2026-06-22). Awaiting review before implementation.

## Problem

The corpus only covers **2025-02-26 → today** (0 docs before 2025). Tester Raúl's 50-question
set asked almost entirely about pre-2025 foundational dispositions (Ley 1/2022, Decreto 185/2018,
Ley 5/2018, Ley 10/2014, Orden 3/2017…). For these the system returns "no consta", tangential
same-number 2025 docs, or ungrounded recited prose, and **never cites the actual disposition —
because it is not in the database.** Root cause is a coverage gap, not model quality.

A full historical backfill (2014→2025, ~150–200k docs) is judged not worth it. Instead: keep the
recent rolling window and **fetch the specific historical disposition on demand when a query names
one that is missing**, then cache it permanently.

## Decisions (locked)

- **Trigger:** explicit references only — fire only when the question names a specific
  `tipo numero/año` (e.g. "Decreto 185/2018"). High precision, matches the tester's failure cases.
- **UX:** synchronous, with an SSE progress event ("buscando en el archivo histórico…"). First ask
  of a given doc is slower; the doc is cached afterwards so the second ask is instant.
- **Rolling-window eviction** (extend toward 24 months) is explicitly **out of scope here** — we
  only hold ~16 months today, nothing to evict yet. Tracked separately.

## What already exists (reuse, not rebuild)

- **Detector:** `api/query_classifiers.py:is_reference_query()` already recognizes
  `\b\d{1,4}/\d{2,4}\b` + norm-word ("Decret 74/2026"). Used today to gate HyDE.
- **Graph loop:** `agent/graph.py` already has `backfill_node` with a conditional
  `retrieve_candidates → (backfill | rerank_titles)` and an edge `backfill → retrieve_candidates`,
  guarded by `backfill_attempted`. The current `backfill_node` (`agent/nodes/backfill.py`) does a
  blunt date-walk (ingest the month *before* corpus min-date) — useless for a 2018 doc. **This node
  is the insertion point.**
- **Ingest path:** `api.auto_ingest.ensure_range_ingested(start, end, langs)` → `run_pipeline` →
  full `ingest→extract(HTML body)→classify→chunk+embed`. The pipeline ingests **by issue-date**, so
  giving it a disposition's publication date ingests that whole issue-day end-to-end with zero new
  ingest code. `online_ingest_node` already calls this for date-filtered queries.
- **Body fetch:** `/dogv-portal/disposicion/{id}?lang=` (see reference memory) returns clean HTML
  body + dates; already wired into `extract_text`.

## What is genuinely new

Only the **resolver** (ref → DOGV internal id → publication date) and the rewiring of `backfill_node`.

### Resolver — verified working live (2026-06-22)

`POST https://dogv.gva.es/dogv-portal/dogv/search?lang=es_es&page=0&size=10&sort=fecha,desc`
body `{"texto": "<ref + topic terms>", "seccion": []}`
→ `{totalElements, content:[{id, titulo, seccion, organismo, ...}]}`, exact match ranked first.

- **`size>=10` is mandatory** — `size=3` crashes the server (`total/size` BigDecimal bug).
- Ley/Decreto refs are ~unique per year → top-1 is gold (verified: Decreto 185/2018→id 332469,
  Decreto 65/2022→1 hit, Ley 5/2018→ranked first, Ley 10/2014→ranked first).
- **Orden numbers repeat across consellerias** (Orden 3/2017 → 11 hits) → append the question's
  topic/conselleria terms to `texto` to disambiguate.
- Then `GET /dogv-portal/disposicion/{id}?lang=` → `fechaPublicacion` (the issue-date to ingest)
  and `fechaDisposicion`.

## Build steps

### 1. `api/dogv_resolver.py` (new)
- `parse_reference(question) -> Reference | None` — extract `(tipo, numero, año, topic_terms)`.
  Build on `is_reference_query`; normalize tipo synonyms (Ley/Llei, Decreto/Decret, Orden/Ordre,
  Decreto Ley, Resolución…).
- `resolve(ref) -> ResolvedDoc | None` — POST search (`texto` = `"{tipo} {numero}/{año} {topic}"`,
  `size=10`, `sort=fecha,desc`); pick best hit (top-1 for Ley/Decreto; for Orden, prefer the hit
  whose `organismo`/`titulo` best matches the topic terms). Return `disposicion_id`,
  `fecha_publicacion`, `titulo`.
- `fetch_meta(disposicion_id, lang) -> dates/title` via the disposicion endpoint (confirm pub date).
- Timeouts + graceful `None` on any failure (network, 0 hits, ambiguous). Unit-testable against
  recorded fixtures; network calls behind a thin client so tests mock it.

### 2. Rewire `agent/nodes/backfill.py`
- New gating in `_should_backfill` (or inside the node): fire when retrieval returned no/low
  candidates **and** `parse_reference(question)` is not None **and** the ref is not already in DB.
- In the node: `resolve()` → if hit, `ensure_range_ingested(fecha_pub, fecha_pub, DEFAULT_LANGS)`
  → set `backfill_attempted=True` → loop back to `retrieve_candidates` (existing edge).
- Replace the blunt month-walk with this targeted path (keep behind a setting; see config).
- Emit an SSE progress event for the "buscando en archivo histórico" step (extend the
  `_PROGRESS` map in `api/main.py` and the node's profile emission).

### 3. Config (`api/config.py`)
- `ondemand_fetch_enabled: bool` (new; default decided at rollout), reuse/repoint
  `backfill_enabled`. Resolver base URL from existing `dogv_base_url`. Search `size`, timeout.

### 4. Pin flag (light, partly deferrable)
- Add `source`/`pinned` marker to `dogv_documents` (or reuse `text_source`) so a future
  rolling-window eviction never drops an on-demand-fetched historical doc. Not load-bearing until
  eviction exists — can land as a column now, enforced later.

### 5. Single strong citation (the tester's request; independent)
- When `parse_reference` matched and the resolved/ingested doc is now top-ranked, collapse the
  citation list to that one disposition for explicit single-disposition queries. Implement in the
  reader/answer assembly, gated on `is_reference_query`.

## Eval (the tester's set is the gold)

- Extract the 15 answered questions + their gold signatura links from
  `📌 Lote de 50 preguntas.docx` into `data/eval_ondemand_gold.json` (question → expected
  disposicion_id / signatura).
- **Resolver-precision harness:** for each, run `resolve()` and score top-1 == gold. Run this
  *before* graph wiring (the spike).
- **End-to-end:** after wiring, assert the answer cites the gold disposition (extend the existing
  answer-eval gold-cited check).

## Latency note

`ensure_range_ingested(pubdate, pubdate)` ingests the **whole issue-day** (~30–50 docs → embeds),
not just the one disposition. Simpler and reuses the tested path, but adds seconds to the first ask.
Measure in the spike; if too slow for sync UX, add a single-doc fast path (insert only the resolved
`disposicion_id`: one row + extract + classify + chunk) as a later optimization. v1 = whole-day reuse.

## Rollout

1. Spike: `dogv_resolver` + resolver-precision harness over the 15 gold cases. No graph changes.
2. Single-doc vs whole-day decision based on measured latency.
3. Wire `backfill_node`, config flag (default off), SSE progress.
4. E2E eval on the tester's set; enable; notify Raúl.

## Open questions for review

- Default the feature **on** once eval passes, or ship behind a flag and dark-launch first?
- For ambiguous Orden refs with no clear topical winner: pick top-1 silently, or surface a
  "¿te refieres a…?" disambiguation? (v1 leans top-1 + topic-term ranking.)
- Pin column now or defer entirely until eviction is built?
