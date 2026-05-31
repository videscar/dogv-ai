# W3+W2 batch — full-100 validation (2026-05-31)

Code: branch `eval-v2-weaknesses` @ `be88f8d` (dirty=False). thinking-ON, temp 0,
`unit_aware_strict`. Answers `answers_w2_full.jsonl`, judgments `judgments_w2_full.jsonl`
(83 reused byte-identical from W3 + 17 changed re-judged vs gold). Deterministic
pipeline (overnight 0.694 reproduced 17/17 byte-identical earlier).

## Verdict

| Metric | baseline | W3 | **W3+W2** |
|---|---|---|---|
| Gated (answerable) | 0.667 | 0.728 | **0.756** |
| Critical-error rate | — | 0.056 | **0.033** |
| Faithfulness | — | 0.967 | 0.967 |
| OOS abstention | 0.90 | 0.90 | 0.90 |

Per-category vs W3: wrong_ref 0.60→**0.70**, clean 0.62→**0.66**, multihop 0.556→**0.611**;
annex 0.857, colloquial 1.0, vague 0.688, OOS 0.90 unchanged. es=va=0.756 (va caught up).
W2 = +0.028 gated on top of W3; cumulative +0.089 over baseline; crit-errors ~halved.

## Regression audit

**Zero gated regressions** (no W3-correct question dropped below 1.0). Only 17/100
answers changed (all the W2-affected amount questions); 16 ans→ans, 1 DUMP→ans (v2-090).
Crit set 6→4: removed {v2-007, v2-046, v2-083}; **added v2-090** — the one cost: it
went from a safe validation dump to a confident WRONG answer (claims the GV-Talent
global is 180k, conflating becas programs / curs). Gated 0 either way, but now a crit.
Track under W1/W5 (becas program selection + curs correction).

## Error taxonomy (31 non-perfect; all map to known weaknesses — no new classes)

- **Vague → on-topic-but-not-gold (~10, gated 0.5)** → W5 retrieval precision on
  underspecified queries: v2-021/022/023/025/026/028/029/077/098/099.
- **Within-doc / annex extraction (~8)** → W6: v2-061 (Alberic annex row), v2-007
  (600k clause vs 1.6M global, same doc), v2-051, v2-004, v2-006, v2-018, v2-092, v2-056.
- **Retrieval-miss / wrong doc (~5)** → W5: v2-009, v2-086, v2-052, v2-091, v2-047.
- **Salary twins (3)** → W2-salary residual (annual vs monthly table share `subkind=Listas`,
  so the W2 convocatoria-vs-concesión rule can't separate them): v2-001 (crit), v2-084, v2-094.
- **OOS over-reach (1)** → W4 (deferred, risky): v2-069 (crit).

Remaining crits: v2-001 (salary), v2-052 (tribunal/gerent), v2-069 (OOS), v2-090 (becas).

## Next (post-overnight)
W5 retrieval: vague-query precision is the largest single bucket; plus the hard
retrieval misses (v2-009/086/052/091). W6: within-doc/annex row extraction (v2-061/007/051).
Salary twins need a unit-aware (mensual vs anual) reader signal, separate from subkind.
