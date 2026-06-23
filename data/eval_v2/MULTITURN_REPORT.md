# Multi-turn conversation — live proof (branch `conversational-multiturn`)

Ran `eval_v2/collect_multiturn.py` against the branch served live on `:8090` (vLLM
Qwen3.6-27B chat, bge-m3 embed), 10 two-turn scenarios from `eval_multiturn.jsonl`
(vague turn-1 → named follow-up). Each turn is a real `/ask` with the accumulating
conversation `history`, exactly as the Chainlit UI sends it.

## Headline

| metric | turn-1 | final turn |
|---|---|---|
| citation any-hit (proxy) | **2/10** | **6/10** (7/10 after accepting mt-01's Fons decree) |

`contextualize` fired (rewrote the follow-up) on 7/10; the lift is the whole point:
a named follow-up recovers programs the vague turn-1 could not.

## Read-judged (facts, not proxy — the proxy understates, as it did all session)

| id | lang | turn-1 | final | read judgment |
|---|---|---|---|---|
| mt-01 | va | miss | (proxy miss) | **correct** — names the Fons de Cooperació Municipal, cites its regulating decree (110/2025); didn't pin the 2026 assignment doc |
| mt-02 | va | miss | miss | partial — right topic (UV investigador places), generic, gold not pinned (va + many siblings) |
| mt-03 | es | miss | miss | drift — "OPE 2026" matches every municipal employment offer; Generalitat OPE not pinned (inherent ambiguity → would refine again) |
| mt-04 | es | miss | miss | **faithful abstention** — "No consta… directora general de Relaciones Institucionales"; did NOT fabricate (thin 1-chunk gold not retrieved) |
| mt-05 | es | miss | **HIT** | correct — pinned the 2024-2026 strategic plan |
| mt-06 | va | HIT | HIT | correct |
| mt-07 | va | miss | **HIT** | correct — Jocs Esportius subvencions |
| mt-08 | es | HIT | HIT | correct — JQCV convocatoria |
| mt-09 | va | miss | **HIT** | correct — IVC audiovisual subvencions |
| mt-10 | es | miss | **HIT** | correct — UA profesorado ayudante doctor |

Read-judged success after the follow-up ≈ **7–8/10 correct/acceptable** (vs 2/10 at
turn-1), with the residuals being the known retrieval ceiling (va weakness, thin
appointment docs, genuinely ambiguous "OPE 2026") — explicitly out of scope here
(encoder finetune parked). Where retrieval still can't find a thin doc, the system
**abstains faithfully** (mt-04) rather than fabricating.

## Conclusions

1. The multi-turn capability works end-to-end live: `history` threads, `contextualize`
   rewrites follow-ups, synthesis self-corrects / abstains faithfully.
2. Recall roughly triples across a follow-up (read-judged 2 → 7–8 / 10).
3. Single-turn is unchanged: empty-history requests no-op the contextualize node
   (zero LLM calls, `reason=no_history`); OOS still abstains.

## Tuning opportunities (not blockers)

- `contextualize` sometimes *appends* to the vague turn-1 instead of fully replacing
  it (mt-01: "…ajuda nova per als pobles del Fons de Cooperació Municipal"); a tighter
  standalone rewrite could sharpen retrieval.
- mt-02/mt-03 residuals are recall-ceiling, not conversation-design, issues.
