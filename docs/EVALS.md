# Evaluation: how DOGV AI is measured, and what it actually scores

This is the story of the evaluation, not a spec. It explains why the headline
number is **0.700** and not **1.0**, what each shipped fix bought, which idea was
rejected *because* it couldn't be measured reproducibly, and where the system still
fails. Every number here comes from a committed report or run — nothing is rounded
up for effect.

> **Shipped config, authoritative full re-run (100Q, 2026-07-08, production
> settings, thinking OFF, temperature 0):**
>
> | Metric | Value |
> |---|---|
> | **Overall gated score** (hard factual gate) | **0.700** |
> | Faithfulness to evidence | 0.989 |
> | Critical-error rate | 1.1% |
> | Out-of-scope abstention | 10/10 |
> | Retrieval R@10 (rerank) / MRR | 0.744 / 0.582 |
> | Frozen holdout (29Q) gated | 0.667 |
> | External tester regression set (30Q) | 30/30 |
>
> Lineage: the deterministic thinking-OFF config went **0.622 → 0.700** over the
> campaign below — overtaking the thinking-ON config (0.694) that was *rejected* for
> being non-reproducible.

Sources: [`data/eval_v2/reports/answer_metrics.json`](../data/eval_v2/reports/answer_metrics.json),
[`data/eval_v2/reports/retrieval_metrics.json`](../data/eval_v2/reports/retrieval_metrics.json),
and the dated reports in [`data/eval_v2/`](../data/eval_v2/). Every answer run carries
a `.meta.json` sidecar stamping the exact `git_sha` + serving config (also exposed at
`/health`), so a score can never drift from the code that produced it.

---

## 1. The first suite measured nothing

The original eval set was **saturated**: retrieval recall@10 ≈ 0.96 and the answer
metric read 1.0. But that answer metric only checked for lexical term overlap and a
non-zero citation count — it could not tell a correct answer from a fluent wrong one.
A perfect score that survives a wrong answer is not a metric; it's a thermometer that
always reads 37°C. So the real numbers below are *lower* than the old ones by design:
we deliberately built something that could fail.

## 2. A deliberately hard 100-question suite

[`data/eval_v2/`](../data/eval_v2/) is 100 questions, grounded in real DOGV facts and
built to break the pipeline in specific, diagnosable ways:

- **Strict 50 Valencian / 50 Spanish.** 90 answerable + 10 out-of-scope (must abstain).
- **Categories:** clean 25, vague 16, colloquial 16 (typos / no accents), annex 14
  (content only in PDF-fallback tables), wrong_ref 10 (question cites the wrong
  number/date), out_of_scope 10, multihop 9.
- **Source mix:** 47 HTML / 43 PDF-annex, so the annex questions stress the
  ingestion path, not just retrieval.
- Gold is scored at the **document** level (what the pipeline actually ranks); the
  gold chunk is recorded for traceability.

Retrieval and answer quality are scored **separately**, and answer quality is split
into two independent axes behind a hard gate:

- **correctness** 0/1/2 (wrong-or-missing / partial / fully correct),
- **faithfulness** 0/1 (is every claim supported by the retrieved evidence?),
- **critical_error** → a **hard gate that zeroes the question** on any material
  factual error (wrong figure, date, reference, name).

The gate is the point: a confidently-wrong answer scores **0**, not partial credit.
That is why 0.700 is a load-bearing number.

## 3. The idea that was rejected *because* it couldn't be measured

The first clean A/B was thinking-ON vs thinking-OFF synthesis
([`COMPARISON_REPORT.md`](../data/eval_v2/COMPARISON_REPORT.md), 2026-05-31):

| | Baseline | Think-OFF | **Think-ON** |
|---|---|---|---|
| Gated score | 0.667 | 0.622 | **0.694** |
| Critical-error rate | — | 0.067 | 0.044 |
| Mean latency / question | — | **70.9 s** | 142.9 s |

On paper thinking-ON won (+0.072). It was **shipped OFF anyway**, for two reasons the
score alone hid: (1) the reasoning trace is non-deterministic, so the same question
could score differently run-to-run — you cannot regression-test a moving target; and
(2) it is **2× slower**. (A later investigation confirmed the run-to-run answer
variance was a non-deterministic RRF tiebreak, since fixed with stable tiebreakers —
so today's OFF config is bit-reproducible.) The decision was to make the *deterministic*
config good enough to win outright, and drive **that** number up instead.

## 4. The accuracy campaign (each lever, measured before → after)

Each fix below was gated on the full 100Q against a byte-identical control run, so
the delta isolates the change. These were measured on the **corpus of the day**; as
the corpus grew (§5) the absolute numbers shifted, but each lever earned its place:

| Lever | What it fixes | Gated Δ (as measured) | Report |
|---|---|---|---|
| **Unit-aware claim-guard** | The validator was policing grounded figures (source writes `1.366,74`, answer writes `1.366,74 euros`) and *dumping correct answers*. Now it checks figure **presence** among all source numbers. | 0.694 → **0.728**; wrong_ref 0.35 → **0.60** | [`W3_VALIDATOR_REPORT.md`](../data/eval_v2/W3_VALIDATOR_REPORT.md) |
| **Premise-correction** | When the question cites a norm by the wrong number/date but the evidence clearly *is* that norm, correct the reference instead of abstaining — without substituting a different norm on a related topic. | 0.750 → **0.783** | commit `dfebca3`/`0c77ac1` |
| **Within-doc figure pin** | The reader LLM dropped figures that sit deep in a long annex chunk (subject-amount, total granted). | → **0.794**, critical 0.011 | commit `8146ffd`/`4e17a22` |
| **Confidence-gated HyDE + citation floor** | Fire the HyDE lane only when the baseline pool is low-confidence (`rrf_margin < 0.22`), and guarantee every reader-payload doc contributes a citable quote. | **+6 gold-cited** vs off, OOS abstention 10/10 intact | [`HYDE_GATE_FLOOR_REPORT.md`](../data/eval_v2/HYDE_GATE_FLOOR_REPORT.md) |
| **Deterministic norm-pin** | An in-corpus norm named by the question is always cited when it reaches the read set (fixes "read it but didn't cite it"). | 30/30 tester set, 0 regressions | commit `89a0d97` |
| **RC1 — sibling-edition recency** | Recurring publications (near-identical by doc-embedding cosine) were answered from a **stale year's** edition; suppress all but the most recent. | 0 regressions (12Q + 30Q) | commit `55ebd82` (RC1) |
| **RC3 — evidence extraction** | Keyword-**window** chunk truncation (not prefix) + re-grounding of non-verbatim LLM quotes onto the source chunk. | 14/14 + tester 30/30 citation-diff clean | commit `d11a8c2` (RC3) |
| **RC4 — semantic anchor** | A doc in the top-N of a semantic lane is guaranteed a fused-pool slot, so correlated BM25 lanes can't evict the gold on paraphrase/annex queries. | 0 regressions | commit (RC4) |

## 5. Re-baselined on the real shipping corpus: 0.700

The campaign above ran while the corpus was ~29k documents. It then grew ~3× to a
full rolling 24-month window (~50.9k docs) after recovering the silently-dropped
*bis* editions and back-filling history. A bigger corpus means **more distractors**,
so the suite got harder in absolute terms. The honest, apples-to-apples number is the
**full re-run on today's corpus and frozen production config**:

- **Gated 0.700**, faithfulness **0.989**, critical-error **1.1%**, out-of-scope
  abstention **10/10**.
- This is the deterministic thinking-OFF config, now **above** the rejected
  thinking-ON 0.694 — and still ~2× faster.

Retrieval on the same run (`rerank` stage, 90 answerable):

| k | Recall@k | nDCG@k |
|---|---|---|
| 1 | 0.489 | 0.567 |
| 5 | 0.700 | 0.644 |
| **10** | **0.744** | 0.658 |
| 20 | 0.756 | 0.660 |

MRR 0.582. The LLM reranker does real work: pre-rerank hybrid fusion is R@1 0.333 /
MRR 0.458, so reranking lifts R@1 by ~16 points.

## 6. Where it still fails (measured, not hand-waved)

- **Recall plateaus.** R@10 = 0.744 and R@20 = 0.756 — going from 10 to 20 candidates
  buys almost nothing. So **~25% of answerable questions never retrieve the gold
  document at all.** This is an embeddings/query-understanding ceiling, not a ranking
  bug: passing more context cannot fix it.
- **Vague queries are the floor** and **Valencian trails Spanish** (es rerank R@10
  0.778 vs va lower); Valencian dominates the unrecoverable misses — a real bge-m3
  asymmetry on Valencian/Catalan. The weakest answer category is `wrong_ref` (0.50).
- **Latency:** median `/ask` ~50–60 s — a multi-stage pipeline on a local 27B.
- **No OCR** for scanned PDFs.

None of these are hidden by the gate; they're why the number is 0.700.

## 7. Generalization: tuned vs. a frozen holdout

Every shipped fix was validated on the full 100, so to make the tuning-vs-generalization
gap visible, [`eval_v2/make_holdout.py`](../eval_v2/make_holdout.py) carves a
**content-blind** holdout — `sha1(id) % 4 == 0`, which ignores category, difficulty and
answer, so it cannot be cherry-picked. That is **29 of 100** questions, frozen
2026-07-09 and excluded from tuning from now on.

On the 2026-07-08 run:

| Slice | n (answerable) | Gated | Critical-error |
|---|---|---|---|
| Tuned | 63 | **0.714** | 1.6% |
| **Frozen holdout** | 27 | **0.667** | **0.0%** |

The holdout sits ~4.7 points below the tuned slice on gated correctness but has **zero**
critical errors — no sign of brittle over-fitting to specific questions.

**Honest caveat:** because the shipped fixes were historically validated against the
full 100, these 29 were *seen* during development — this is a **frozen-forward** holdout,
so treat the 0.667 as a **lower bound** on the true generalization gap. Its value is
going forward: it is now a genuinely untuned slice for the next round of changes.

## 8. Reproduce it yourself

```bash
# 1. Collect answers from a live API (stamps git_sha + config into a .meta.json)
python eval_v2/collect_answers.py --base-url http://127.0.0.1:8088 \
    --out data/eval_v2/answers_run.jsonl

# 2. Judge them with the fixed-rubric LLM-as-judge (use a DIFFERENT model than the
#    one being judged — Qwen3.6-27B answers, so judge with something else)
export JUDGE_BASE_URL=https://your-judge-host JUDGE_MODEL=your-judge-model
python eval_v2/judge.py --answers data/eval_v2/answers_run.jsonl \
    --out data/eval_v2/judgments_run.jsonl

# 3. Score, with the tuned-vs-holdout split
python eval_v2/score_answers.py data/eval_v2/judgments_run.jsonl \
    data/eval_v2/answers_run.jsonl data/eval_v2/reports/answer_metrics.json \
    --holdout data/eval_v2/holdout_ids.json

# 4. Retrieval metrics + regression gate
python scripts/run_eval.py --input data/eval_v2/retrieval_input.json
python eval_v2/retrieval_metrics.py data/eval_reports/<run_id>.json

# Verify the committed holdout matches the content-blind rule
python eval_v2/make_holdout.py
```

The judging rubric is fixed in code ([`eval_v2/judge.py`](../eval_v2/judge.py)) so a
third party can regenerate the judgments; the holdout is a committed, rule-derived list
([`data/eval_v2/holdout_ids.json`](../data/eval_v2/holdout_ids.json)).
