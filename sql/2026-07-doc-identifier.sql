-- Structured-identifier index for the identifier layer.
--
-- Lexical BM25 and semantic embeddings are both blind to structured identifiers
-- (a code like GACUJIMA/2025/36 is one atomic tsvector token; embeddings cannot
-- separate twin extracts). This table holds every identifier extracted from a
-- document's title/body at ingest, so the query side can look one up EXACTLY and
-- hard-pin the document (see api/identifiers.py, and the query lane).
--
-- id_kind:
--   'code' — letter-prefixed slash expedient/project codes (GACUJIMA/2025/36,
--            ERESAR/2026/39R07/0008) and compact body codes (24I636)
--   'bdns' — BDNS subsidy database IDs (895054), body-only
--   'norm' — the document's own numbered norm identity (DECRETO 74/2026),
--            read off the start of its title (subsumes norm-pin's title ILIKE)
-- Deliberately excluded: 'person' (names stay lexical-only) and 'ref' (already a
-- uniquely-indexed column on dogv_documents; the query lane reads it directly).
--
-- id_key is the normalized lookup key (lowercase, separators collapsed to '/'),
-- produced by the same api/identifiers.normalize_* helpers on both the ingest
-- and query sides so an exact match is possible.

CREATE TABLE IF NOT EXISTS doc_identifier (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL,
    id_kind       TEXT NOT NULL,
    id_key        TEXT NOT NULL,
    raw           TEXT,
    source        TEXT NOT NULL,          -- 'title' | 'body'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_doc_identifier_document
        FOREIGN KEY (document_id) REFERENCES dogv_documents(id) ON DELETE CASCADE,
    CONSTRAINT uq_doc_identifier
        UNIQUE (document_id, id_kind, id_key)
);

-- The exact-match query lane: SELECT document_id WHERE id_kind = ? AND id_key = ?.
CREATE INDEX IF NOT EXISTS idx_doc_identifier_key
  ON doc_identifier (id_kind, id_key);

CREATE INDEX IF NOT EXISTS idx_doc_identifier_document
  ON doc_identifier (document_id);
