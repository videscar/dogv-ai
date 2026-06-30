-- Title full-text (BM25) indexes for dogv_documents.
--
-- title_bm25_search() (api/retrieval.py) ranks dogv_documents.title with
--   to_tsvector(<config>, title) @@ websearch_to_tsquery(<config>, query)
-- Without a matching expression index this seq-scanned all ~51k rows and recomputed
-- to_tsvector for every row (~1.0s/call), and the title lane runs ~5x per retrieval
-- pass (title lane + once per BM25 facet + PRF). These GIN expression indexes make the
-- membership test index-backed: measured ~1037ms -> ~1-170ms per call on production data.
--
-- ts_config_for_language() only ever yields 'spanish' or 'catalan', so one index per
-- config covers every query. The bound :ts_config param still resolves to a constant at
-- plan time, so no query change is needed (verified with EXPLAIN: the index is used).
--
-- Apply live without locking writes:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS ... (cannot run inside a txn block)
-- IF NOT EXISTS keeps this idempotent against indexes already built live.

CREATE INDEX IF NOT EXISTS idx_dogv_documents_title_tsv_es
  ON dogv_documents USING GIN (to_tsvector('spanish', coalesce(title, '')));

CREATE INDEX IF NOT EXISTS idx_dogv_documents_title_tsv_ca
  ON dogv_documents USING GIN (to_tsvector('catalan', coalesce(title, '')));
