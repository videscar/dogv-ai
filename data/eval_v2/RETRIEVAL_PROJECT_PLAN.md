# Retrieval & Evidence-Coverage Project — Plan & Scope

_Drafted 2026-06-02, on top of baseline `answers_t4vllm_full` (commit e3a130b):
gated **0.794**, critical-error rate **0.000**, faithfulness **1.000**, OOS abstention **1.000**._

## 1. Problem & current state (measured)

The accuracy campaign closed out the *synthesis-side, confident-wrong* failures
(premise-correction + figure-pin → crit-rate 0.033 → 0). What remains is mostly
**the answer content not reliably reaching the synthesizer**, plus a hard tail of
reasoning-quality misses.

Measured on the 90 answerable questions (from the stored run):

| Metric | Value |
|---|---|
| Pool-recall — *any* gold doc in candidate pool | **0.82** |
| Pool-recall — *all* gold docs (multihop-relevant) | **0.80** |

The 28 answerable failures (18.5 gated points) segment as:

| Bucket | Count | Meaning | In scope? |
|---|---|---|---|
| **A — pool miss** | 11 | gold doc never enters the candidate pool (embedding/recency miss) | ✅ yes |
| **B — read, not evidenced** | 4 | gold doc read into payload but its answer chunk dropped from the 8-quote evidence | ✅ yes |
| **C — in evidence, wrong answer** | 13 | gold content reaches synthesis but the answer is wrong/incomplete | ❌ **out** (reasoning-quality, separate project) |

Bucket A by category: **vague 7**, clean 2, annex 1, wrong_ref 1.
Bucket B: multihop 2, vague 1, wrong_ref 1.

**Addressable surface for *this* project = A + B ≈ 15 questions (~9–10 gated points).**
Bucket C (13 Q — multihop disambiguation, granularity, completeness) is a
synthesis/reasoning project, explicitly out of scope here.

## 2. Goal

Raise pool-recall and evidence-coverage so the gold answer content reliably reaches
synthesis, recovering a meaningful share of the 15 A+B questions **without regressing
the 0.794 / crit-0 / faithfulness-1.0 baseline**. Realistic target: recover ~half →
gated **~0.83–0.85**, crit-rate held at 0.

## 3. Workstreams

### WS0 — Retrieval measurement harness (prerequisite, ~0.5 day)
Synthesis is slow (~1h45m/full-100) and vLLM-variant. We need a **fast, deterministic
retrieval-level loop** to iterate without synthesis:
- per question: pool-recall@k (gold doc in candidates), evidence-recall (gold chunk in
  the 8 quotes), and **fact-recall** (does the literal answer string survive into the
  selected chunks).
- per-category / per-language breakdowns; reuse `eval_v2/retrieval_metrics.py`.
- **Deliverable:** one script, run in seconds, with committed baseline numbers. This is
  the iteration signal for WS1–WS3.

### WS1 — Vague-query recall & recency (targets A-vague: ~7 Q, ~1–2 days)
**Root:** vague queries ("¿hay ayudas para X?") retrieve many topical siblings but
miss or under-rank the specific / most-recent gold (e.g. an older sibling outranks the
newest convocatoria).
- Levers: (a) **recency-aware ranking** when the query is vague/temporal; (b) light
  **query expansion** (vague → concrete program terms); (c) confirm hybrid keyword+vector
  fusion is pulling its weight.
- Fast-loop measurable on the vague subset. **Highest concentration, cleanest target.**
- Risk: recency boost must not hurt specific/dated queries — guard + measure.

### WS2 — Multi-part retrieval & chunk coverage (targets B + multihop: ~4–6 Q, ~2–3 days)
**Root (verified):** for multihop failures both gold docs are *read*, but the
answer-bearing chunk (a person's name, a figure) isn't lexically findable from the
question, so keyword scoring picks the title chunk and the LLM reader drops one hop.
- Levers: (a) **query decomposition** — split a multi-part question into sub-queries,
  retrieve + read + extract evidence per sub-query, union the result (each hop gets its
  own budget); (b) **chunk-level reranking** (cross-encoder or cheap LLM pass) so the
  answer chunk surfaces even without lexical overlap.
- Most architecturally significant lever; also helps some A misses.
- Risk: **latency** (multiple retrievals/rerank per query) and complexity.

### WS3 — Recall ceiling (targets A-other: ~4 Q, time-boxed ~2–3 days, open-ended)
**Root:** gold simply isn't in the top-k by bge-m3 similarity (the known embeddings
ceiling; recall plateaus ~0.77–0.80; va lags es).
- Levers: (a) **cross-encoder reranker** over a larger candidate pool; (b) hybrid
  **BM25 + vector**; (c) bigger pool + rerank.
- ⚠️ Diminishing returns — the embeddings ceiling is real. **Time-box it**; don't let it
  become a rabbit hole.

### WS4 — Integration & full-100 validation (~1 day)
Combine the shippable workstreams; run the authoritative full-100; **guard the
0.794 / crit-0 baseline** (no regression); blast-test before the full run; commit clean
SHA. Re-judge changed answers, re-running movers 3× for vLLM variance.

## 4. Validation methodology
- **Fast loop (WS0):** retrieval metrics — deterministic, seconds. Primary iteration signal.
- **Slow loop:** full-100 answer eval — only at integration (WS4).
- **Guardrail:** every change must hold gated ≥ 0.794 and crit-rate = 0. Blast-test the
  affected subset + controls before any full-100 (established discipline).

## 5. Risks & tensions
- **Latency vs. beta-readiness (the big one):** WS2 (decomposition) and WS3 (reranking)
  add per-query retrieval/LLM passes → *worse* interactive latency, which directly
  conflicts with the beta-readiness latency goal. If interactive latency is a beta
  blocker, these need a latency budget, async, or cached design — decide up front.
- **Embeddings ceiling (WS3):** may yield little; time-box.
- **Regression:** retrieval changes affect all 100 questions; the fast loop + the
  baseline guard contain this.
- **vLLM variance:** answer-level measurement is noisy; rely on the deterministic
  retrieval metrics to iterate, synthesis only to confirm.

## 6. Out of scope
- Bucket C (13 Q reasoning/synthesis quality) — separate project.
- Re-training/replacing embeddings; wholesale re-chunking (HTML backfill just landed).
- The synthesis-prompt work (done: premise-correction, figure-pin).

## 7. Sequencing & effort
WS0 (0.5d) → **WS1 vague/recency (1–2d, cleanest, highest concentration)** →
WS2 decomposition/chunk-rerank (2–3d, biggest lever) → WS3 recall (time-boxed 2–3d) →
WS4 integration (1d). **Total ≈ 1.5–2.5 weeks**, WS3 open-ended.

**Recommended order rationale:** WS1 first — it's the highest-concentration bucket
(7 vague pool-misses), measurable on the fast loop, and lower-risk than decomposition.
Decide the **latency tension** before committing to WS2/WS3.
