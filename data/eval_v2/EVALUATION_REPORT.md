# DOGV AI — Hard Eval Suite v2: Evaluation Report

**Date:** 2026-05-30 · **Corpus:** post HTML-first backfill (28,926 docs / 290,150 chunks, issues 2025-02-26 → 2026-05-29) · **Models:** chat `qwen3.6-27b`, embed `bge-m3` (llama.cpp).

The previous eval sets were saturated (retrieval recall@10 ≈ 0.96, answer aggregate 1.0 — but the old answer metric only checked lexical terms + citation count, so it measured nothing). This suite is grounded in real corpus facts, deliberately hard, and scores retrieval and answer quality **separately**, with answer quality split into independent **correctness** and **faithfulness** axes behind a **hard factual-error gate**.

## The suite (100 questions)

- Strict **50 va / 50 es**. 90 answerable + 10 out-of-scope (abstain).
- Categories: clean 25, vague 16, colloquial 16 (typos/no-accents), annex 14 (PDF-fallback content), wrong_ref 10, out_of_scope 10, multihop 9.
- Answerable items: html 47 / **pdf-annex 43** (stresses the new ingestion path).
- Gold scored at **document level** (the pipeline ranks documents); `gold_chunk` recorded for traceability.

## Production fan-out (measured, not assumed)

Via `/ask?debug=true`, the pipeline passes **~10–12 documents / ~40–47 chunks / 8 evidence quotes** to the synthesis LLM (governed by `ask_rerank_max_candidates=10` → `read_max_docs`, + coverage/amount/eligibility extras). **Primary retrieval k = 10.**

---

## 1. RETRIEVAL (90 answerable items, production `rerank` order)

| k | Recall@k | nDCG@k |
|---|----------|--------|
| 1 | **0.489** | 0.489 |
| 5 | **0.767** | 0.676 |
| **10 (prod)** | **0.767** | 0.676 |
| 20 (ceiling) | **0.767** | 0.676 |

**MRR = 0.600.** Pre-rerank (hybrid fusion) is markedly worse — R@1 0.344, MRR 0.475 — so the LLM reranker is doing real work on ranking quality.

**The headline: recall plateaus at k=5.** R@5 = R@10 = R@20 = 0.767. Going from 10 to 20 candidates buys *nothing*. So **23% of answerable questions never surface the gold document at all**, and this is a recall/embeddings ceiling, not a ranking problem — passing more context cannot fix it.

Decomposing the 21 misses@20:
- **10 are absent from *every* lane** (vector + BM25 + title): a true embeddings/chunking/query-understanding ceiling. **7 of these 10 are Valencian.**
- 11 are present in some lane but lost in fusion/reranking (recoverable with tuning).

### Retrieval by category

| Category | n | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|----------|---|-----|-----|------|-----|---------|
| clean | 25 | 0.68 | 0.88 | 0.88 | 0.78 | 0.81 |
| annex (pdf) | 14 | 0.71 | 0.86 | 0.86 | 0.79 | 0.80 |
| colloquial | 16 | 0.63 | 0.81 | 0.81 | 0.67 | 0.71 |
| multihop | 9 | 0.00 | 0.78 | 0.78 | 0.32 | 0.82 |
| wrong_ref | 10 | 0.30 | 0.70 | 0.70 | 0.42 | 0.49 |
| **vague** | 16 | 0.25 | **0.50** | **0.50** | 0.35 | 0.37 |

### Retrieval by language / source

| Group | n | R@1 | R@10 | MRR | nDCG@10 |
|-------|---|-----|------|-----|---------|
| es | 45 | 0.60 | **0.82** | 0.69 | 0.76 |
| **va** | 45 | 0.38 | **0.71** | 0.51 | 0.59 |
| html | 47 | 0.55 | 0.79 | 0.64 | 0.72 |
| pdf/annex | 43 | 0.42 | 0.74 | 0.56 | 0.63 |

**Reads:** (1) **Vague queries are the floor** (R@10 0.50) — under-specified phrasing can't pin a single document. (2) **Valencian lags Spanish by ~11 pts R@10** and dominates the unrecoverable misses — a real embeddings asymmetry. (3) **Annex/PDF content is broadly retrievable** (R@10 0.86 for annex-detail questions) — good news for the HTML-first + PDF-fallback ingestion; the gap is at the *reader*, below. (4) multihop R@1=0 is expected (needs ≥2 docs); its low MRR (0.32) reflects how deep you must read to get *both* docs, though each individually ranks well (nDCG 0.82).

---

## 2. ANSWER QUALITY (live `/ask`, judged on 3 axes + hard gate)

Each answerable item: **correctness** (0=wrong/missing, 1=partial, 2=full → /2), **faithfulness** (every claim supported by retrieved evidence), and a **hard gate** (material factual error → question scores 0). Out-of-scope items: **abstention** (binary).

| Metric (answerable, n=90) | Value |
|---|---|
| **Gated score** (hard-gate combined) | **0.667** |
| Correctness (mean) | 0.672 |
| **Faithfulness rate** | **1.000** |
| Critical-error rate | 0.022 (2 cases) |
| **Abstention rate** (oos, n=10) | **0.90** |

### The faithfulness/correctness split is the key story

**Faithfulness is essentially perfect (1.00) while correctness is only 0.67.** The system almost never invents facts — when it lacks the answer in its retrieved chunks it says *"no consta"* or returns a candidate-document list. Its failures are **missing/incomplete information, not hallucination.** (This is consistent with the `current_strict` claim-guard / answer-validator in the pipeline.) For an official-gazette assistant this is the right failure posture — but it means the headline risk is *under-answering*, and the two cases that *did* slip the gate are the most dangerous outputs in the set (below).

### Answer quality by category

| Category | n | Correctness | Gated | Crit-err |
|----------|---|-------------|-------|----------|
| colloquial | 16 | 0.81 | **0.78** | 0.06 |
| annex | 14 | 0.71 | 0.71 | 0.00 |
| vague | 16 | 0.69 | 0.69 | 0.00 |
| multihop | 9 | 0.67 | 0.67 | 0.00 |
| clean | 25 | 0.60 | 0.60 | 0.04 |
| **wrong_ref** | 10 | 0.55 | **0.55** | 0.00 |

By language: es gated 0.644 (abstention 0.80), va gated 0.689 (abstention 1.00). Valencian answers slightly *out-score* Spanish despite weaker retrieval — small-n noise, but worth watching.

**Reads:** (1) **Colloquial phrasing barely hurts** the *answer* when retrieval succeeds (0.78) — typos and missing accents are well tolerated. (2) **wrong_ref is the weakest** (0.55): a wrong decree/order number frequently derails retrieval into a fallback list, even though, when the right doc *is* found, the system corrects the reference cleanly (v2-042, v2-044). (3) **clean (0.60) is dragged down not by retrieval but by the reader** — see below.

---

## 3. Where it fails — root causes with concrete examples

The 90 answerable failures cluster into four mechanisms. The first two are the big ones, and they are **distinct, separately-fixable problems**.

### A. Retrieval ceiling — gold never retrieved (≈8 cases, unrecoverable by reranking)
- **v2-009** (es, clean): *"¿A quién se ha nombrado gerente de la Universitat de València?"* → gold (Daniel González Serisola) **absent from all lanes**; the system finds only the *cese* doc and concludes no one was named. The nombramiento doc is a 1-chunk record — short docs embed poorly and get out-ranked.
- **v2-086** (va, clean): tribunal president of the habilitación-nacional concurso → gold not retrieved; returns three unrelated tribunals.
- **v2-022 / v2-028** (va, vague): "ajudes per als pobles" / "diners per a l'esport" → the specific gold (Fondo Cooperación Municipal / Jocs Esportius) is absent; the system returns *other* valid aids.

### B. Reader / chunk-selection failure — right doc retrieved, answer chunk dropped (≈9 cases) ⚠️
This is the most actionable finding. The gold document **is** in context, but the specific fact-bearing chunk isn't among the ~4 chunks/doc passed to the synthesis LLM, so the system says *"no consta"*:
- **v2-004** (va): audiovisual ficció dotation — gold retrieved, amount chunk not passed.
- **v2-007** (es): GV-Talent 1.600.000 € global — gold retrieved (extracto), figure chunk not passed.
- **v2-036** (va): gerent UV name — gold retrieved, the chunk naming the person not passed.
- **v2-055** (es): Plan estratégico signer — *"no consta"*… yet **v2-077** (the *vague* version of the same doc) retrieved the signing chunk and answered correctly. Same document, different chunk selected → opposite outcome.
- **v2-059** (es): Orden 18/2026 submission deadline; **v2-061, v2-063, v2-064** (annex detail): the right annex doc is retrieved but the specific line (a beneficiary, a municipality amount) isn't in the passed chunks. Note **v2-012 (Alaquàs) succeeds** while **v2-063 (Alboraia) fails on the same Fondo doc** — annex retrieval is per-query chunk-luck.

**Implication:** raising `ask_chunks_per_doc` or improving within-document chunk ranking for long/annex docs would recover a meaningful chunk of the correctness gap *without* touching retrieval.

### C. Adjacent-document confusion (≈3 cases) — including a gated critical error
- **v2-083** (es, clean) — **CRITICAL ERROR (gated 0):** asked the *convocatoria* max for the 2026 fallas prizes (207.000 €); the system pulled the *concession* resolution and confidently answered **"157.000 €"** — a real number, wrong question.
- **v2-087** (es): winner of Mejor Falla Infantil → pulled the convocatoria (which names no winner) instead of the concession → *"no consta"*.

### D. Confident wrong number (the other gated critical error)
- **v2-032** (va, colloquial) — **CRITICAL ERROR (gated 0):** A1 monthly base salary. The system led with **1.387,24 €** (which is the *alt càrrec/eventual* row, increment 1.5%) instead of the funcionario A1 base **1.366,74 €** — it even mentioned the correct figure second. A grounded-but-misattributed number is exactly the official-gazette failure the hard gate is designed to punish.

### E. Abstention failure (1 of 10 oos)
- **v2-069** (es): "horario de la Seguridad Social en Alicante" (out of scope) → the system latched onto a subvención to a *Colegio de Graduados Sociales* "Oficina Pública" and fabricated **"de 9 a 14 horas."** The only hallucination in the whole run, and it's on an unanswerable question — the most important abstention failure to fix.

The other 9 out-of-scope questions abstained correctly, several with good scope reasoning (v2-074: "el DOGV regula transport de València, Alacant i Castelló, no el metro de Madrid").

---

## 4. Summary of weaknesses the new set exposes

1. **A hard retrieval recall ceiling (~0.77).** 23% of gold docs never enter the candidate pool; passing more context (k=20) recovers none of them. ~half are unreachable in *any* lane → an embeddings/chunking problem, not ranking.
2. **Valencian is a second-class retrieval citizen** (R@10 0.71 vs 0.82; 7 of 10 unrecoverable misses). Worth investigating bge-m3 behaviour on Valencian and the va chunking.
3. **A distinct reader/chunk-selection bottleneck.** Even when retrieval succeeds, long and annex/PDF documents frequently fail because the answer-bearing chunk isn't among the few passed to the LLM (v2-055 vs v2-077 on the *same* doc). This is the cheapest win available.
4. **Vague queries underperform end-to-end** (retrieval R@10 0.50): the system can't disambiguate "ajudes per als pobles" to one document — a candidate for clarifying-question behaviour.
5. **Wrong/approximate references derail retrieval** (answer 0.55) — though the system corrects the reference well *once the right doc is found*.
6. **Confident wrong numbers are rare but real** (2/90), and both came from pulling an *adjacent* document (concession vs convocatoria) or an *adjacent* table row. These are the highest-severity outputs.

**The good news the set also confirms:** faithfulness is essentially perfect (the system abstains rather than fabricates), annex/PDF-fallback content from the new ingestion is genuinely retrievable and quotable, colloquial/typo robustness is strong, and the LLM reranker materially improves ranking over raw fusion.

---

## Reproduce

```
python eval_v2/build_set.py                 # regenerate the suite
python scripts/run_eval.py --input data/eval_v2/retrieval_input.json \
       --output-dir data/eval_v2/reports --run-id v2_retrieval --k-values 1,3,5,10,20,50
python eval_v2/retrieval_metrics.py data/eval_v2/reports/v2_retrieval.json \
       data/eval_v2/retrieval_input.json data/eval_v2/reports/retrieval_metrics.json
python eval_v2/collect_answers.py --base-url http://127.0.0.1:8088   # resumable, ~3h
python eval_v2/score_answers.py data/eval_v2/judgments.jsonl \
       data/eval_v2/answers_raw.jsonl data/eval_v2/reports/answer_metrics.json
```

Artifacts: `eval_set_v2.jsonl`, `judgments.jsonl`, `answers_raw.jsonl`, `reports/retrieval_metrics.json`, `reports/answer_metrics.json`.
