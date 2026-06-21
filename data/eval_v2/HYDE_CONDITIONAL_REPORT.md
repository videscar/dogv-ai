# Conditional HyDE — full validation report (2026-06-21)

Branch `conditional-hyde` @ `86eb09a`. Question that was parked: does enabling HyDE
(validated at retrieval level, 0.82→0.86) regress the end-to-end ANSWER quality?
Now answered with a controlled differential eval at **production settings**.

## What "conditional HyDE" is
HyDE on for contentful queries; OFF for reference-queries (a norm citation by
number/year "Decret 74/2026" or norm-word+month "la resolució d'abril"), because the
hypothetical drifts off the cited norm and the heavy HyDE RRF lane evicts the
correctly-retrieved gold. `is_reference_query()` in `api/query_classifiers.py`.

## Retrieval validation (in-process probe, 90 answerable)
- pool recall: baseline(off) **0.822** → conditional **0.856** → unconditional 0.867
- **v2-092 regression FIXED**: baseline rank 2 → unconditional EVICTED → conditional rank 2 ✓
- conditional gains +4 vague (v2-026/034/035/076), −1 borderline (v2-025); refs protected.
- Detector: fires 9/10 wrong_ref, 0/16 vague, never on a HyDE-recovered query.

## Answer validation (differential, controlled)
Two eval API instances, identical committed SHA + prod settings (thinking=OFF,
unit_aware_strict, validator on), ONLY HyDE differs. 100 Q each arm.

| metric (n=90 answerable) | treat (HyDE cond) | ctrl (HyDE off) |
|---|---|---|
| gold_doc_cited | 0.64 (58/90) | 0.63 (57/90) |
| num_recall (proxy) | 0.54 | 0.61 |
| OOS abstention (read, n=10) | **10/10** | **10/10** |
| latency median | **150s** | 131s (**+19s**) |

**Verdict: answer-neutral within noise, + a latency cost. NOT a clear win.**

### Hand-judged divergences (11 flagged; separating signal from noise)
- **Proxy artifacts / ties** (no real diff): v2-005 (3.000€ both; "3.000,00" vs "3.000" digit-format), v2-039 (180.000 both), v2-078 (both OK).
- **Nondeterminism** (ref-query, HyDE off both): v2-045 — treat actually BETTER (premise-corrects 11/2022→12/2022); proxy penalized it for not emitting a number.
- **Treat GAINS**: v2-020 (annex, cites gold+correct), v2-035 (colloquial, cites gold+correct).
- **Treat REGRESSIONS**:
  - v2-049 (multihop): HyDE dropped gold 86084 → missed GV-Talent hop ("no consta"). Still FAITHFUL (abstains, no wrong fact), but incomplete.
  - v2-032 (colloquial): cites gold but gives 1.366,74 (2025 base) for "este any"=2026 (correct 1.387,24). Wrong-year figure — the known two-figure cluster; ctrl got it right this run.
  - v2-023/076 (vague): retrieved tangential program; defensible but missed expected. Inherent vague ambiguity.
  - v2-095 (multihop/anexo): deportista 2.000 vs expected 2.250 (anexo has multiple per-person figures).

Net: gains (v2-020/035) ≈ losses (v2-049/032); num_recall gap is mostly artifact+noise;
OOS abstention intact. The retrieval recall gain does NOT convert to better answers —
recovered vague docs reach synthesis but get diluted by topical siblings at citation time.

## Decision: DO NOT DEPLOY
Answer-neutral + ~15–19s/query latency + 1–2 borderline regressions (v2-049 incompleteness,
v2-032 figure) = does not clear the "clear win" bar for a production legal tool. The parked
default-off decision is, on this evidence, correct. Branch kept (ready) but not merged/deployed.

## Better next design (recommended, not done here)
1. **Confidence-gate HyDE** — run it ONLY on low-retrieval-confidence / vague-detected queries,
   not all non-ref. Cuts the latency to near-zero on the common path AND avoids disrupting
   clean/multihop pools (would likely remove the v2-049/v2-032 regressions while keeping vague gains).
2. **Reader/citation dilution** — recovered vague docs enter the pool but aren't cited (v2-076);
   the bottleneck has moved downstream to evidence/citation selection. That's the next lever.

Artifacts: `answers_hyde_cond_treat.jsonl`, `answers_hyde_off_ctrl.jsonl` (100 each).
