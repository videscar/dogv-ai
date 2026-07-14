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

The hard failure needs **both** near-twin-family and year-trap. `ERESAR/2026` (code-year =
issue-year) and `/12` (fewer twins) each miss one condition and succeed. This is why it
looks intermittent.

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

## Staging (each step independently shippable, gated on the probe set)

1. Build the identifier-probe set + baseline it on current prod. *(done — see the probe set)*
2. Intent year-guard (smallest, independent win).
3. Ingest extractor + `doc_identifier` table.
4. Query lane + migrate norm-pin onto it.
5. Fold in code / bdns / person classes.

## Open questions

- Person-name matching: exact-ish on title/body, but names are common — needs a uniqueness
  guard and probably a confidence floor before pinning. Highest false-pin risk of the classes.
- Whether the identifier lane pins at pool-injection time (like RC4 anchors) or post-pool
  (like norm-pin). Post-pool is the lower-risk migration; pre-pool is cleaner long-term.
- `ref`-column queries (users quoting a DOGV publication ref) are rare but trivially exact —
  include for completeness or defer.
