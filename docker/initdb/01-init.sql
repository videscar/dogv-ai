-- Runs once, on first cluster init (mounted into /docker-entrypoint-initdb.d).
-- Enables the extensions the pipeline relies on and defines the `catalan`
-- text-search configuration used by the Valencian BM25 lane.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- PostgreSQL ships a `spanish` snowball config but not a `catalan` one. The
-- retrieval SQL (title FTS, rebuild_tsv, hybrid search) calls
-- to_tsvector('catalan', ...) for Valencian rows, so define it here. We alias it
-- to the Spanish stemmer as a pragmatic, dependency-free default — good enough
-- for BM25 over closely-related Ibero-Romance text; swap in a real Catalan
-- dictionary if you have one.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'catalan') THEN
        CREATE TEXT SEARCH CONFIGURATION catalan (COPY = pg_catalog.spanish);
    END IF;
END
$$;
