# W3 — Validator tuning (claim-guard) result

Date: 2026-05-31. Code: branch `eval-v2-weaknesses` @ `2269b58` (dirty=False).
Config: thinking-ON, temp 0.0, `answer_claim_guard_mode=unit_aware_strict` (new default).
Baseline: overnight thinking-ON run (`current_strict`), gated 0.694 — reproduced
17/17 byte-identical by a same-code `current_strict` run, so the A/B isolates the
guard alone (corpus stable, pipeline deterministic).

## Result (full 100Q)

| Metric | Baseline (current_strict) | **W3 (unit_aware+fix)** | Δ |
|---|---|---|---|
| Gated (answerable) | 0.694 | **0.728** | **+0.034** |
| Faithfulness | 0.967 | 0.967 | 0 |
| Critical-error rate | 0.044 | 0.056 | +0.012 |
| OOS abstention | 0.90 | 0.90 | 0 |
| wrong_ref category | 0.35 | **0.60** | **+0.25** |

7 of 100 answers changed; the other 93 are byte-identical (keep their judgments).

## Root cause (confirmed by instrumentation, not inferred)

`unit_aware_strict` policed the answer's currency/percent figures against the
source's *currency-adjacent* figures only. Source tables write figures bare
("A1 1.366,74 52,60 …") while answers write them with a unit ("1.366,74 euros"),
so grounded figures were classified as unsupported → repair → dump. Captured the
rejected synthesis for v2-032/089: it was the *correct* answer being thrown away.

**Fix:** answer side stays unit-aware (only currency/percent claims are policed, so
incidental dates/counts/article-numbers don't trigger dumps — this is what rescues
wrong_ref), but the source side now checks figure *presence* among all source
numbers, not currency-adjacency. Equivalent to `current_strict`'s grounding for
money/percent, without policing incidental numbers. Locked by a unit test.

## Per-question changes (the 7)

| Q | cat | was | now | note |
|---|---|---|---|---|
| v2-041 | wrong_ref | 0 dump | **2** | corrected 73≠74 + both names (gold) |
| v2-045 | wrong_ref | 0 dump | **2** | corrected 12≠11 + 3.000 € (gold) |
| v2-044 | wrong_ref | 0 dump | **1** | corrected ref; gave per-person 19k not total 22.2M |
| v2-048 | multihop | 0 dump | **1** | 2025=5.000 ✓, 2026 not found |
| v2-046 | wrong_ref | 0 dump | **0 (crit)** | corrected ref but 157k≠207k — **W2** doc-selection |
| v2-068 | out_of_scope | 1.0 abst | 1.0 abst | still abstains (no fabrication) |
| v2-070 | out_of_scope | 1.0 abst | 1.0 abst | still abstains (no fabricated nota) |

## Safeguard added (so heavy runs are never ambiguous again)

`/health` and the startup log report the running git SHA + branch + dirty flag
(captured at import; uvicorn has no --reload) plus eval settings. `collect_answers`
writes a sidecar `<out>.meta.json` per run and warns on `git_dirty`. `git_dirty`
tracks tracked-code changes only (untracked scratch files don't trip it).

Authoritative baseline for W2: `answers_w3_full.jsonl` + `judgments_w3_full.jsonl`.
