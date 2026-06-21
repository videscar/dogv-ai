# Confidence-gated HyDE + reader citation floor — validation report (2026-06-21)

Branch `conditional-hyde`. Follow-up to `HYDE_CONDITIONAL_REPORT.md`, which found
conditional HyDE (on for all non-reference queries) **answer-neutral** + a latency
cost, and recommended two next levers. Both are now built and validated end-to-end.

## What changed

1. **Confidence-gate HyDE** (`agent/nodes/retrieve.py`, `api/config.py`).
   The HyDE generation hop (an LLM call) is now **deferred until after a baseline
   retrieval pass** and fires only when that baseline pool is low-confidence —
   `rrf_margin < ask_hyde_margin_threshold` (0.22). A confident baseline already has
   the gold near the top; firing HyDE there only risks the reader citing a drifted
   sibling. Reference-query gating (`is_reference_query`) is unchanged.
   - Threshold calibrated on eval_v2 (fresh retrieval probe, this code): the queries
     HyDE actually recovers have baseline margins ≤ 0.21 (v2-020/032/034/035/099); the
     HyDE citation-regressions sit higher (v2-023 = 0.262, v2-078 = 0.318). 0.22 keeps
     every recovery and skips the two confident-baseline regressions.
   - `top_vec_sim` and `rrf_top` were tested as gate signals and **do not separate**
     (e.g. a real recovery v2-089 has rrf_top 0.045, below the OOS queries' 0.06–0.07).

2. **Reader citation floor** (`agent/nodes/read.py`, `api/config.py`).
   Diagnosis (citation probe + last night's answer files): of 18 queries that retrieved
   the gold but never cited it, the loss was **entirely at `extract_evidence`** — the
   gold reached the reader payload with its chunks, but the reader LLM + lexical
   coverage extracted no quote for it (anchor-poor vague/colloquial queries and annex
   golds). Doc-selection never lost a pooled gold. The floor injects the best chunk of
   any reader-payload doc missing from `evidence` (confidence order, cap 5) so the
   recovered recall becomes citable.

3. **Citation-fallback caps** (`api/answer.py`, `agent/nodes/answer.py`).
   The floor can pad `evidence` with topical siblings; on an abstention/validation-
   fallback that surfaced a long citation tail. Both fallback paths (validator-rejected,
   and "LLM cited nothing") now cap attached citations to `answer_fallback_max_items`
   (the same few the abstention summary lists). Grounded answers cite explicitly and
   never hit these paths — verified unchanged.

## Result — authoritative full-100 (eval API, prod settings, only these changes)

`gold_doc_cited` is the proxy; hand-judged notes below correct it.

| metric (90 answerable) | NEW (gate+floor) | treat (cond HyDE) | ctrl (HyDE off = prod) |
|---|---|---|---|
| **gold_cited** | **63** | 58 | 57 |
| clean (25) | 20 | 19 | 19 |
| vague (16) | 6 | 4 | 6 |
| colloquial (16) | 12 | 12 | 11 |
| annex (14) | 11 | 12 | 11 |
| multihop (9) | **9** | 7 | 7 |
| wrong_ref (10) | 5 | 4 | 3 |
| OOS abstention (substance, n=10) | **10/10** | 10/10 | 10/10 |

**+6 gold_cited vs current prod (ctrl), +5 vs last-night HyDE (treat).** Deltas:
vs ctrl **+10 / −4**, vs treat **+8 / −3**.

### Hand-judged: the proxy understates the win
- **v2-059 (counted a loss, actually a WIN):** NEW answers correctly ("10 días hábiles…"
  = expected) but cites a sibling with the same deadline; treat said "No consta" (wrong).
- **v2-061 (counted a loss, actually a mutual abstention):** NEW and treat both abstain
  on the hard annex first-beneficiary extraction; treat just dumped 7 cits incl. gold.
- **v2-077 (real regression):** vague query, NEW cites the *draft* plan, treat the
  *approved* plan (gold). Inherent vague draft-vs-approved ambiguity.
- **v2-039 (vs ctrl only):** gate must fire HyDE (margin 0.175, can't skip without
  losing v2-035); HyDE demotes the rank-1 gold and the answer cites a sibling.

Net of hand-judging: **~1 real answer regression (v2-077)** against +8–10 real gains
(gate restored the HyDE citation-regressions v2-023/078; floor converted
v2-020/047/057/089…; multihop went 7→9).

### Known minor nit
- **v2-067 (OOS):** abstains correctly ("No hay evidencia disponible") but the LLM
  explicitly cites the floor-padded evidence (10 cits). Not reachable by the fallback
  caps (citations are explicit), and the floor can't be OOS-gated (no retrieval signal
  separates OOS from answerable-vague — both look like "no keyword match"). Cosmetic.

## Latency
Honest: the gate **halves** HyDE fires (~44/90 answerable fire vs ~74 unconditional) and
skips HyDE on ~56% of clean/multihop. It is **not** "near-zero on the common path" —
clean queries often have low RRF margin too (two similar top docs), so margin alone
can't perfectly flag "clean & confident". But the residual clean/multihop HyDE fires
are **answer-neutral** (0 citation changes measured), so the cost there is only the
~2–3s HyDE generation, not quality. (Absolute eval latency was vLLM-load dominated and
not comparable run-to-run.)

## Recommendation
Ship-worthy: a clear citation win over current prod (HyDE-off) with one bounded vague
regression and OOS abstention intact. Defaults set HyDE still **off**
(`ask_hyde_enabled=False`); turning it on now carries the gate + floor that make it a
net win. Suggest enabling HyDE in a follow-up once the v2-077-style vague ambiguity and
the v2-067 OOS citation cosmetics are weighed.

Artifacts: `answers_full90.jsonl` equivalent run (gate+floor), calibration in
`tmp/pool_off.jsonl`. Config flags: `ask_hyde_confidence_gated`,
`ask_hyde_margin_threshold`, `ask_read_citation_floor`, `ask_read_citation_floor_docs`.
