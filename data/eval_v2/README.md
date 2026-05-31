# DOGV AI — Hard Evaluation Suite v2

A harder, non-saturated replacement for the old eval sets, grounded in real
corpus content (post HTML-first backfill). 100 questions, strict 50/50
Valencian/Catalan vs Spanish.

## Files

| File | What it is |
|------|------------|
| `eval_set_v2.jsonl` | The full suite — one JSON record per question (human-readable). |
| `retrieval_input.json` | The 90 answerable items in `run_eval.py` format (gold doc ids). |
| `answer_input.json` | All 100 items for the answer / abstention judge. |
| `reports/` | Generated metric reports. |

Build/regenerate with `python eval_v2/build_set.py`.

## Record schema (`eval_set_v2.jsonl`)

```
id              v2-001 .. v2-100
language        "va" | "es"            (50 / 50)
category        clean | vague | colloquial | wrong_ref | multihop | annex | out_of_scope
perturbation    none | underspecified | typo | no_accents | approx_ref | multi_doc | annex_detail
question        the user utterance (with the perturbation applied)
expected_answer concise gold answer with the critical facts, or null for abstain items
should_abstain  true for out_of_scope items
gold_sets       list of gold sets. [[id]] single; [[id1,id2]] = both required (multi-hop);
                [[id1],[id2]] = any-of acceptable. Scored at DOCUMENT level.
doc_ids         flat list of gold document ids
gold_refs       DOGV refs of the gold docs (e.g. "2026/12103")
gold_chunk      traceability note: which chunk/section/annex carries the answer
doc_kind        Ayudas | Subvenciones | Becas | Premios | Empleo Publico | Otros
text_source     html | pdf   (pdf = annex / PDF-fallback content — the new ingestion path)
```

## Category mix (100)

clean 25, vague 16, colloquial 16, annex 14, wrong_ref 10, out_of_scope 10, multihop 9.
Answerable items split html 47 / pdf-annex 43.

## Scoring

- **Retrieval** (90 answerable items): `eval_v2/retrieval_metrics.py` over the
  `run_eval.py` report → recall@k, MRR, nDCG at k=1,5,10,20. Primary stage is the
  production `rerank` order; primary k=10 (the pipeline passes ~10–12 reranked
  documents / ~40–47 chunks / 8 evidence quotes to the synthesis LLM — verified
  empirically via `/ask?debug=true`). Gold is document-level (the pipeline ranks
  documents). Out-of-scope items are excluded (no relevant doc exists).
- **Answer quality** (all 100): `eval_v2/collect_answers.py` calls the live `/ask`
  endpoint capturing the answer, citations and the evidence quotes the synthesis
  LLM saw; then each answer is judged on three independent axes — correctness,
  faithfulness/groundedness, and abstention (out-of-scope only). A **hard gate**
  zeroes any answer with a material factual error (wrong number/date/ref/name).
  `eval_v2/score_answers.py` aggregates with per-category/per-language breakdowns.
