# DOGV RAG — Eval v2 Comparison: Baseline vs Sprint-2 (thinking OFF vs ON)

Date: 2026-05-31. Suite: 100 Q (50 va / 50 es; 90 answerable + 10 out-of-scope).
Judge: Claude, strict rubric (correctness 0/1/2; faithful 0/1; critical_error → hard gate to 0;
validation_failed fallback-dumps scored as non-answers = 0). All three runs judged by the same judge;
baseline judgments are from an earlier (looser) pass — treat baseline as a soft reference, OFF-vs-ON as the clean A/B.

## Headline

| Metric | Baseline (pre-fix) | Think-OFF (post-fix) | **Think-ON (post-fix)** |
|---|---|---|---|
| **Gated score** | 0.667 | 0.622 | **0.694** ✅ best |
| Correctness | 0.672 | 0.622 | **0.694** |
| Faithfulness | 1.00* | 0.967 | 0.967 |
| Critical-error rate | 0.022* | 0.067 | **0.044** |
| Abstention (OOS) | 0.90 | **1.00** | 0.90 |
| Retrieval recall@cand (answerable) | 0.778 | **0.800** | **0.800** |
| Mean latency / question | ~10-70s | 70.9s | 142.9s |

\* baseline faithfulness/crit are from the looser earlier judging; not directly comparable.

**Winner: thinking-ON.** Gated 0.694 vs OFF 0.622 (+0.072, ~+12% relative) and vs baseline 0.667.
Retrieval is now deterministic and identical across OFF/ON (recall@cand 0.800, up from baseline 0.778).
Cost: ON is 2× slower (142.9s vs 70.9s per question).

## Per-category (gated)

| Category | n | Think-OFF | **Think-ON** | Δ |
|---|---|---|---|---|
| colloquial | 16 | 0.906 | **1.000** | +0.094 |
| annex | 14 | 0.821 | **0.857** | +0.036 |
| vague | 16 | 0.531 | **0.688** | +0.157 |
| multihop | 9 | 0.389 | **0.500** | +0.111 |
| clean | 25 | 0.580 | **0.620** | +0.040 |
| wrong_ref | 10 | 0.350 | 0.350 | 0.000 |
| out_of_scope | 10 | **1.000** | 0.900 | −0.100 |

Thinking-ON improves every category **except** out_of_scope (one fabrication, see W4) and wrong_ref (tied, the weakest category in both).

## Per-language (gated)

| Lang | Think-OFF | **Think-ON** |
|---|---|---|
| es | 0.600 | 0.689 |
| va | 0.644 | 0.700 |

va ≈ es in both modes (the historical va lag is gone at the answer level; retrieval recall is identical across languages now).

## Why thinking-ON wins: it synthesizes where OFF dumped

The biggest single lever was the **validation_failed fallback dumps** (retrieval found the gold doc but the
answer layer rejected synthesis and dumped raw snippets). OFF had ~13 of these; ON recovered most:

| Q | OFF | ON | What changed |
|---|---|---|---|
| v2-049 (multihop becas) | 0 (dump) | **2** | ON synthesized 180.000 + 1.600.000 + "GV-Talent more" |
| v2-095 (multihop UNED/deportistas) | 0 (dump) | **2** | ON synthesized both 75.000 + 2.250 |
| v2-040 (CACVSA president) | 0 (dump) | **2** | ON identified the correct Acord (gold-level) |
| v2-076 (DANA habitatge) | 0 (dump) | **2** | ON identified DANA lloguer + Decret 167/2024 |
| v2-022 (vague pobles) | 0 (dump) | **1** | ON synthesized on-topic municipal aids |
| v2-027 (certs valencià) | 1 | **2** | ON added the matrícula dates |
| v2-020 (CMIART) | 1 | **2** | ON added the 50% support figure |

## Remaining weaknesses (to tackle next)

**W1 — wrong_ref is the worst category (0.35 in both modes).** When the user cites a wrong decree/order
number, the system usually fallback-dumps or misses instead of correcting + answering.
Failing: v2-041, v2-044, v2-045, v2-046, v2-090, v2-091 (6/10). Working: v2-042, v2-043, v2-089 correct.
Lever: teach the pipeline to detect "near-miss reference" and answer the corrected one.

**W2 — Two systematic wrong-DOCUMENT-selection bugs hitting es/va twins (the only critical errors that persist):**
- **Salary table**: retrieves doc 6352 (annual table, "+1.5%") over gold 3145 (monthly 1.366,74 €, "+2.5%").
  Wrong on v2-001 (es) and muddled on v2-084 (va). BUT the correct doc 3145 *is* retrieved and used correctly
  on v2-032 and v2-089 — so both docs are in the candidate set; the reader/synthesis picks the wrong table
  depending on question phrasing. This is a **reader disambiguation** problem, not retrieval.
- **Falles total**: answers 157.000 € (from concession doc 17382) instead of gold 207.000 € (convocatoria doc 8487).
  Wrong on v2-083 (es), and v2-046 (va) dumps. Gold doc for the total is 8487; system anchors on 17382.

**W3 — The answer-validator is a double-edged sword.** It correctly converts some would-be-wrong answers into
safe non-answers (ON avoided 3 of OFF's 6 critical errors this way: v2-018, v2-046, v2-084), but it also
**rejects good syntheses**: v2-048 — OFF answered it correctly (President 5.000 € both years, unchanged), but
ON's validator rejected the synthesis and dumped → 0. Net it still leaves ~7 fallback dumps in ON
(v2-041, 044, 045, 046, 048, 056, 090). Lever: tune validator threshold / give it a "repair" path instead of dump.

**W4 — Thinking-ON occasionally over-reaches (its 2 regressions vs OFF):**
- v2-069 (OOS): fabricated a "9-14h" horario for Seguridad Social Alicante pulled from an unrelated doc →
  failed abstention (drops OOS 1.0→0.9). Richer reasoning sometimes answers what it should decline.
- v2-007 (clean): confidently misquoted the global as "600.000 €" instead of 1.600.000 € — a NEW critical
  error OFF didn't make (OFF dumped). Thinking can introduce confident misquotes of specific figures.

**W5 — Retrieval ceiling: recall@cand = 0.800.** 18/90 answerable questions don't have the gold doc in the
candidate set at all (retr=N). Many are still answered correctly from a va/es twin or sibling convocatoria
(e.g. v2-034, v2-037), so effective recall is higher — but the hard misses (v2-009 gerent, v2-086 tribunal
president, v2-098/099 specific uni places) are unrecoverable by any synthesis change; they need retrieval work.

**W6 — Annex row-level extraction still weak.** When the answer is one row of a long annex/table, the reader
often can't pull it even with the gold doc retrieved: v2-061 (first DANA beneficiary in Alberic) misses in both modes.

## Suggested priority for tomorrow
1. **W3 validator tuning** — biggest, cheapest win: ~7 questions are recoverable answers being thrown away.
2. **W2 reader disambiguation** — kill the 2 systematic twin critical-error pairs (salary table, falles total).
3. **W1 wrong_ref handling** — worst category; near-miss reference correction.
4. **W4 guardrails** — keep ON's gains but stop OOS over-reach (v2-069) and figure misquotes (v2-007).
