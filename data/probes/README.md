# Identifier probes

Retrieval-level probe set for the identifier-layer work (see
`docs/identifier_layer_design.md`). Each probe anchors a query on a **structured
identifier** (code / person / norm / bdns / ref-column) and records the gold document(s)
so we can measure whether the right document is retrieved and cited.

These are **not** answer-eval cases in the `eval_v2` format: the primary check is
`gold_doc_ids` (did the identifier resolve to the right document?), with `gold_fact` as an
optional answer-side sanity check. The identifier layer fires 0/100 on `eval_v2`, so this
probe set — not the eval gate — is its validation harness.

## Schema (one JSON object per line)

| Field | Meaning |
| --- | --- |
| `id` | stable probe id (`IP01`…) |
| `id_class` | `code` \| `person` \| `norm` \| `bdns` \| `ref` |
| `identifier` | the anchoring identifier as it appears in the corpus |
| `traps` | any of `year_trap`, `near_twin_family`, `partial_rendering`, `lowercase_code`, `ref_column` |
| `lang` | `es` \| `va` — the query language |
| `question` | the natural-language query |
| `gold_doc_ids` | document(s) that correctly answer it (es and/or va edition) |
| `gold_fact` | a checkable fact from the gold doc (amount, deadline, appointee…) |
| `baseline` | prod behaviour at authoring time: `pass` \| `fail` \| `retrieved_but_hedged` \| `unknown` |
| `notes` | why the probe is here / what it isolates |

## Coverage

- **Flagship failure** (IP01/IP02/IP03): `GACUJIMA/2025/36`, both traps, es + va + partial.
- **Controls that already pass** (IP04/IP05/IP09): guard against regressions and isolate
  which trap actually causes failure (IP04 has both traps yet passes).
- **Class coverage**: code, person (incl. lowercase-code edge IP07), norm, bdns, ref-column.
- **Trap isolation**: IP12 (old code, no year_trap) vs IP01 (code + year_trap) separates
  token-loss from the year-trap.

Gold doc ids and facts were verified against the prod corpus on 2026-07-14. Before adding
probes, always check for title near-twins (`ILIKE '%<identifier>%'`) so the gold is
genuinely unique.

## Running

```
python scripts/oneoff/run_identifier_probes.py            # all probes vs prod :8088
python scripts/oneoff/run_identifier_probes.py --class code
python scripts/oneoff/run_identifier_probes.py --api http://127.0.0.1:8090
```

`GOLD_HIT` = any `gold_doc_ids` appears in the citations. The runner reports per-class
pass/fail and flags any `baseline: pass` probe that regressed.
