# Identifier layer — audit & design

**Status:** design / not implemented. Audit run 2026-07-14 against prod corpus (51,637 docs).

## Why this exists

The pipeline retrieves through two channels — lexical BM25 and semantic embeddings —
and **both are structurally incapable of matching a structured identifier**:

- BM25 tokenizes a code as one atomic token. `GACUJIMA/2025/36` becomes the single
  tsvector token `gacujima/2025/36`; standalone `gacujima` matches **0 titles**, and a
  full-question `websearch_to_tsquery` AND-match fails on long titles (they lack some
  stem) and falls to an OR-fallback that shatters the code into `gacujima`+`2025`+`36`,
  none of which match the atomic token. The single discriminating signal is lost.
- Embeddings (bge-m3) are semantically blind to codes; the 14 near-identical GACUJIMA
  extract twins all embed ~0.85 cosine to each other, so the code — the only thing that
  tells them apart — carries no vector signal.

Consequence: every time an identifier class turns out to matter, we bolt on a bespoke
deterministic "pin". There are already **three** (norm-pin, RC4 anchor, and the proposed
code-pin), each with its own regex and its own call-site. That proliferation — the same
missing abstraction rediscovered — is the smell this design removes.

This is **not** a chunking or ingestion problem. Chunking (token-window min/max/overlap)
and ingestion are sound and out of scope.

## The concrete failure (measured)

Live prod, GACUJIMA family, 2026-07-14:

| Query | Result |
| --- | --- |
| `GACUJIMA/2025/12` exact | correct doc, confident correct answer |
| `GACUJIMA/2025/36` exact | gold 90640/90686 **not retrieved**; cites wrong siblings, hedges |
| `gacujima 2025 36` (spaces) | confidently answers for the **wrong project** (`/2025/10`) |
| PI-name "Navarro Esbrí" | retrieves the right docs but synthesis still hedges |
| `ERESAR/2026/39R07/0008` (other family) | correct doc, correct answer |

Two compounding root causes, both from mis-decomposing the code:

1. **BM25 token loss** (above).
2. **Intent year-trap.** `api/intent.py:_YEAR_PATTERN` (`\b20\d{2}\b`) + `_infer_year_range`
   read the `2025` inside `GACUJIMA/2025/36` as a standalone year and build a
   `since 2025-01-01 / until 2025-12-31` filter — but the gold doc is dated **2026-06-11**,
   so it is filtered out of the pool before ranking matters.

These are **two independent removal mechanisms, either one sufficient** to lose the gold:
(a) the year-trap filters it out of the pool before ranking; (b) token-loss leaves it
un-rankable inside a near-twin family. Verified against publication dates: IP01 gold is
published **2026-06-11** (code-year 2025) and IP12 gold **2025-03-05** (code-year 2024) —
both fall outside the code-year window, so both were year-trapped. Removing that window
(step 1) rescues both. IP02 (es) and IP03 (space form) carry IP01's *same* code yet still
miss after the year-guard: there the near-twin token-loss is the operative cause and needs
the exact-match lane (step 3). IP04 (`/12`) passes throughout (shorter query, less competing
vocabulary); `ERESAR/2026` (IP05) never had a twin. The intermittency is the product of the
two overlapping causes, not one. *(Correction: an earlier draft claimed IP12 had no
year-trap and failed on token-loss alone — the dates above refute that; it was year-trapped.)*

## Corpus inventory (51,637 docs)

| Class | Where | Count | Matchable today? |
| --- | --- | --- | --- |
| DOGV publication `ref` (`2026/4148`) | column, `ix_dogv_documents_ref` | 100% (unique per ref+lang) | Indexed but **never used as a query key** |
| Title norm-ref (`DECRETO 74/2026`) | title | 3,649 | Yes — norm-pin |
| Letter-prefixed project/expedient code (`GACUJIMA/2025/36`) | title + body | 2,586 titles | **No** — token loss |
| Body project codes (`24I636`, `codi 25I656`) | body | 552 docs | No |
| BDNS subsidy IDs (`835833`) | body | 4,532 docs | No |
| Group/subgroup codes (`A1`, `C2`) | title | — | Partial — enumeration ILIKE |
| Person names (appointees / PIs) | title/body | 5,450 appointment docs | No (topic/lexical only) |

Two quantified traps:
- **28 near-twin code families** (≥3 docs sharing a code-prefix + doc_kind — the code is
  the only discriminator).
- **1,646 / 2,472 title-code docs (67%) are year-trapped** (code-year ≠ issue-year; codes
  are assigned one year, published the next).

Code uniqueness is favorable: **977 / 1,102 distinct full codes map to ≤2 docs** (es+va),
so exact-code lookup has low false-pin risk.

## Where identifiers are special-cased today (the fragmentation)

Ten modules, **~6 separate regexes re-implementing "N/YYYY" detection**, four ILIKE-title
call-sites, no shared notion of an identifier:

| Module | What it does with identifiers | Own regex |
| --- | --- | --- |
| `api/dogv_resolver.py` | hub: `parse_reference(s)`, `corpus_like_patterns`, `reference_matches_title`, `parse_named_norm_target`, `infer_reference_from_corpus`, `resolve`/`search_dogv` (on-demand fetch) | `_NUMBER_YEAR_RE`, `_REF_IN_TITLE_RE` |
| `agent/nodes/second_hop.py` | norm-pin: `_direct_title_lookup` → ILIKE title via `corpus_like_patterns` | (reuses resolver) |
| `api/edition_recency.py` | extracts process codes from titles to **veto** false edition-families (RC1) | `_CODE_RE` / `_code_tokens` |
| `api/intent.py` | **year-trap**: years (incl. code-embedded) → date filter | `_YEAR_PATTERN` |
| `api/enumeration.py` | group/subgroup-code ILIKE augmentation | `_GROUP_CODE_RE` |
| `api/doc_references.py` | **already extracts N/YYYY refs from bodies at ingest into `doc_reference`** | num-year regex |
| `api/bis_editions.py` | uses `ref` for bis-edition recovery at ingest | — |
| `agent/nodes/retrieve_pool.py` | RC4 semantic anchors (pin mechanism, not identifier-based) | — |
| `api/reader.py` | `_PINNED_DETAILS` (detail-type pinning, not identifier-based) | — |
| `api/field_anchor.py` | identity-vs-field separation (keeps code in identity string implicitly) | — |

## Does one identifier layer subsume these?

**Yes for the retrieval/pin sites; no for the semantic/coverage ones.** Honest split:

**Subsumed — collapse into the layer:**
- Norm-pin title lookup + `corpus_like_patterns` corpus-presence check → one exact-match query.
- The proposed code-pin → *becomes "add the code class to the extractor,"* not pin #4.
- Enumeration group-code ILIKE → same lane.
- The ~6 detection regexes → one shared query-side detector.
- The year-trap → the *same* detector lets intent mask code-embedded years.

**Not subsumed — stay by design:**
- `parse_named_norm_target` / `infer_principal_ref` — topic-based, no exact identifier to
  match. Stays LLM/topic.
- On-demand external fetch (`resolve`/`search_dogv`) — a coverage problem, not a matching
  one. The layer only replaces its "already in corpus?" check.
- RC1 edition-recency, field-anchor field/identity separation, RC4 anchors, reader
  pinning — different concerns; they may *consume* extracted codes but their logic remains.

The layer removes real duplicated machinery and prevents future pin-proliferation without
pretending to absorb the semantic disambiguation that legitimately exists.

## Design (bounded, additive)

1. **Ingest.** Extend the existing `doc_references`-style extractor to emit a typed
   `doc_identifier` table:
   `(document_id, id_kind ∈ {ref, norm, code, bdns, person}, id_key_normalized, raw)`.
   Same pattern and same home as the existing `doc_reference` table (precedent proven).
2. **Query.** One `extract_identifiers(question)` detector replacing the ~6 regexes;
   normalizes separators (spaces/dashes → canonical) and case.
3. **Retrieval.** One exact-match identifier lane feeding the pool with a hard pin, and
   **bypassing the intent date filter** (mirrors norm-pin's existing override). Guarded by
   match-count cap so a bare family prefix (`gacujima 2025`, 12 docs) does not pin.
4. **Migrate** norm-pin, enumeration group-code, and the corpus-presence check onto the
   layer; delete their bespoke paths (net code deletion).
5. **Independent small fix** (shippable first): intent year-guard — do not derive a
   date window from a year that is flanked by `/digits` (i.e. part of a code token).

### Key existing infrastructure to reuse
- `doc_reference` table already does ingest-time identifier extraction (the pattern, the
  home, and `extract_references` code all exist).
- `ref` column is already unique per (ref, lang) and indexed — a first-class publication
  key we simply do not query on.

## Scope, risk, and the eval gap

Medium effort, **debt-reducing** (net deletion at 3 call-sites). Low regression risk:
additive lane. **But it fires 0/100 on `eval_v2`** (no eval question carries a code), so it
**cannot be eval-gated**. The validation harness is the accompanying identifier-probe set
(`data/probes/identifier_probes.jsonl`, runner `scripts/oneoff/run_identifier_probes.py`) —
that probe set, not the code, is the main new work.

## Baseline (prod :8088, 2026-07-14)

`scripts/oneoff/run_identifier_probes.py` → **6/12 GOLD_HIT**, and the misses land exactly
on the classes the layer targets:

| Class | Result | Reading |
| --- | --- | --- |
| code | 2/6 | IP01/02/03 (`/36`) + IP12 (`/2024/26`) miss on token-loss; IP04/IP05 pass |
| bdns | 0/1 | body-only identifier, never matched |
| ref | 0/1 | `ref` column never queried on |
| norm | 1/1 | norm-pin already works (must be preserved) |
| person | 3/3 | names are lexically matchable in titles (IP08 retrieves but hedges) |

Takeaways: (a) the failure surface is precisely code + bdns + ref; (b) norm and person
already pass, so the layer must **preserve** them, not just add.

### After step 1 (intent year-guard) — prod-equivalent worktree API :8090, 2026-07-14

**8/12 GOLD_HIT, +2, zero regressions** (every `baseline=pass` probe still passes; result
stable across two runs):

| Class | 6/12 → 8/12 | What moved |
| --- | --- | --- |
| code | 2/6 → 4/6 | IP01 (`/2025/36` va) and IP12 (`/2024/26`) flip: both were year-trapped (gold published a year after the code-year), the guard removes the spurious window and the gold re-enters the pool and ranks |
| bdns / ref | 0/1, 0/1 | unchanged — body-only / column-only, untouched by step 1 |
| norm / person | 1/1, 3/3 | preserved |

IP02 (es) and IP03 (space form) share IP01's code but stay MISS — token-loss inside the
near-twin family, the target of step 3. So the year-guard is a real standalone win on the
year-trapped population, not merely preparatory.

## Staging (each step independently shippable, gated on the probe set)

1. Build the identifier-probe set + baseline it on current prod. *(done — 6/12, above)*
2. Intent year-guard (smallest, independent win). *(done — 8/12, `api/intent.py` `_complex_code_spans`; not yet deployed to prod)*
3. Ingest extractor + `doc_identifier` table. *(done — table + `api/identifiers.py` + backfill; see below)*
4. Query pin lane for code / bdns / ref. *(done — 12/12, 0 eval_v2 collision; see below)*
5. Norm-pin migration onto the layer (separately eval-gated — deferred, see below).

### Step 4 result (query pin lane) — dev DB, 2026-07-14

`api/identifiers.detect_query_identifiers` (query-side detector, over-generates
candidate keys; the exact `doc_identifier` match is the precision gate) +
`agent/nodes/identifier_pin.apply_identifier_pins` (post-pool, wired into
`retrieve.py` before the second hop). It exact-matches the named identifier,
pins the docs into the read set (`norm_pin_doc_ids`) and protects them from RC1,
**bypassing the ranking pool and the date filter**. Guarded by a match-count cap.

**Probe set 8/12 → 12/12, 0 regressions** (stable): IP02 (es), IP03 (space form),
IP10 (bdns, body-only), IP11 (ref column) all flip; IP01/IP04/IP05/IP12 now cite
*both* es/va editions. So code 6/6, bdns 1/1, ref 1/1, norm 1/1, person 3/3.

Two findings drove the final shape:
- **IP05** looked like a regression (cited the es twin 112235 of the va gold
  112283). Both editions carry the exact code and the same amount — the probe
  gold had under-listed only the va edition. Fixed the gold + added a
  deterministic pin order (date DESC, id ASC). Not a real miss.
- **Norm scoped OUT of the pin lane.** The lane first pinned norm-refs too, which
  regressed eval **v2-042**: that question cites "Decret 167/2025" but is about
  DANA rental aid, whereas the real Decreto 167/2025 is an unrelated appointments
  decree — an exact norm pin collides with premise correction. Codes/BDNS/refs
  are precise, copy-verbatim machine identifiers; norm-refs are human-cited and
  error-prone. Scoping the lane to `{code, bdns, ref}` makes it fire on **0/100**
  eval_v2 questions (all 13 eval identifier-mentions are norm), so it provably
  cannot regress the eval. IP09 (norm) still passes via ordinary title retrieval.

Norm-pin migration (folding `second_hop._direct_title_lookup` onto `doc_identifier`)
is left as a distinct step: it *does* overlap eval_v2's 13 norm cases and must be
gated against them with premise-correction in mind.

### Step 3 result (dev DB backfill, 2026-07-14)

`doc_identifier` created (`sql/2026-07-doc-identifier.sql`) and backfilled over all
51,637 docs in ~15 s (`scripts/build_doc_identifiers.py`, wired into `ingest_pipeline`
best-effort like `doc_reference`). Classes: **code 7,161 rows / 2,695 keys** (2,542 slash
expedient codes + 153 `I`-shape body codes), **bdns 3,891 / 1,807**, **norm 1,424 / 665**.
Two calibrations from the backfill: (a) the compact body-code regex was tightened to the
`\d{2}I\d{3,4}` shape — the broader `\d{2}[A-Za-z]\d+` was also catching budget-line codes
(`08R09`, `28E050`) that are not document identifiers; (b) `norm` extracts the doc's *own*
identity (1,424 ≈ the 1,450 titles that start with a numbered norm — the audit's 3,649
counted norm-refs appearing anywhere, which is the wrong population for a pin). Acid test:
an exact `(id_kind, id_key)` lookup returns precisely the gold docs for every probe
(gacujima/2025/36→{90640,90686}, gacujima/2024/26→{11849,33750}, bdns 895054→{85608,88100},
decreto/74/2026→{86851,89343}), all match-counts ≤2 (es/va pair). `ref 2026/4148` resolves
via the existing `ref` column. No retrieval path changed yet (probe set stays 8/12).

## Open questions

- Person-name matching: exact-ish on title/body, but names are common — needs a uniqueness
  guard and probably a confidence floor before pinning. Highest false-pin risk of the classes.
- Whether the identifier lane pins at pool-injection time (like RC4 anchors) or post-pool
  (like norm-pin). Post-pool is the lower-risk migration; pre-pool is cleaner long-term.
- `ref`-column queries (users quoting a DOGV publication ref) are rare but trivially exact —
  include for completeness or defer.
