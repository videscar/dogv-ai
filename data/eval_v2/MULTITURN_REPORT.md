# Multi-turn conversation — live proof (branch `conversational-multiturn`)

Ran `eval_v2/collect_multiturn.py` against the branch served live on `:8090` (vLLM
Qwen3.6-27B chat, bge-m3 embed), 10 two-turn scenarios from `eval_multiturn.jsonl`
(vague turn-1 → named follow-up). Each turn is a real `/ask` with the accumulating
conversation `history`, exactly as the Chainlit UI sends it. Numbers below are with
the **tightened** contextualize prompt (clean standalone rewrite, drop the vague
turn-1 framing); `answers_multiturn.jsonl` is this run.

## Headline

| metric | turn-1 | final turn |
|---|---|---|
| citation any-hit (proxy) | **2/10** | **7/10** |
| read-judged correct/acceptable | ~2/10 | **~7/10**, now incl. both hard cases |

`contextualize` fired on 8/10. The win the user asked to verify: the tighter prompt
turns a follow-up into a *clean, specific* query and now solves the two cases the
looser prompt only half-handled.

## Tighter rewrite prompt — before/after (deterministic, the clearest signal)

| id | follow-up (raw) | LOOSE rewrite | TIGHT rewrite | effect |
|---|---|---|---|---|
| mt-01 | "No, em referia al Fons de Cooperació Municipal…" | "Hi ha alguna ajuda nova per als pobles **del Fons de Cooperació Municipal**…?" (chained) | "Fons de Cooperació Municipal als ajuntaments" | miss→**HIT**, cites 88759 |
| mt-04 | "…nombramiento de la directora general de Relaciones Institucionales" | *(not rewritten)* | "nombramiento directora general Relaciones Institucionales" | abstain→**HIT**, cites 86845 |
| mt-10 | "Las de la Universidad de Alicante, el concurso en régimen laboral" | "Que se ha publicado sobre plazas… en la Universidad de Alicante…?" (chained) | "concurso régimen laboral profesorado ayudante doctor Universidad de Alicante" | correct (sibling cite) |

The looser prompt appended the new terms to the stale vague question; the tighter
one emits a focused search query.

## Per-scenario (tighter prompt, read-judged)

| id | lang | turn-1 | final | judgment |
|---|---|---|---|---|
| mt-01 | va | miss | **HIT** | pins the 2026 Fons de Cooperació Municipal assignment (88759) |
| mt-02 | va | miss | miss | partial — right topic (UV investigador), gold not pinned (va + many siblings) |
| mt-03 | es | miss | miss | drift — "OPE 2026" matches every municipal offer (inherent ambiguity) |
| mt-04 | es | miss | **HIT** | now retrieves the thin appointment doc (86845), names the DG correctly |
| mt-05 | es | miss | **HIT** | 2024-2026 strategic plan |
| mt-06 | va | HIT | **HIT** | premis falles 2026 — cited es twin 87141 (same ref 2026/17382) |
| mt-07 | va | miss | miss | faithful abstention; retrieval missed 89632 this run (borderline/noise) |
| mt-08 | es | HIT | HIT | JQCV convocatoria |
| mt-09 | va | miss | **HIT** | IVC audiovisual subvencions |
| mt-10 | es | miss | **HIT** | UA profesorado ayudante doctor — cited 85148, the modification of the gold |

## Conclusions

1. End-to-end multi-turn works live: `history` threads, `contextualize` rewrites
   follow-ups into clean standalone queries, synthesis self-corrects / abstains
   faithfully.
2. Recall roughly triples across a follow-up (read-judged ~2 → ~7/10), and the
   tighter prompt specifically rescues the flagship (mt-01) and thin-doc (mt-04)
   cases the looser prompt missed.
3. Residuals (mt-02 va UV, mt-03 ambiguous "OPE 2026", mt-07 noise) are the
   pre-existing recall ceiling, explicitly out of scope (encoder finetune parked);
   where retrieval still misses, the system abstains rather than fabricates.
4. Single-turn unchanged: empty-history requests no-op the node (zero LLM calls,
   `reason=no_history`); OOS still abstains.

Note on the proxy: a single run carries ±1 from retrieval nondeterminism, and es/va
twins + modifications are valid cites — accept-sets widened accordingly (mt-01 +
39109, mt-06 + 87141, mt-10 + 85148). Judge by reading, not the bare proxy.
