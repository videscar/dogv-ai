-- Hybrid search schema for DOGV: chunks + title embeddings + tags

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE IF EXISTS dogv_documents
  ADD COLUMN IF NOT EXISTS doc_subkind TEXT,
  ADD COLUMN IF NOT EXISTS doc_tags JSONB,
  ADD COLUMN IF NOT EXISTS text_source TEXT,
  ADD COLUMN IF NOT EXISTS text_updated_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS rag_chunk (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    language        TEXT NOT NULL,
    text            TEXT NOT NULL,
    embedding       vector(1024),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv             tsvector,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_rag_chunk_document
        FOREIGN KEY (document_id) REFERENCES dogv_documents(id),
    CONSTRAINT uq_rag_chunk_doc_idx
        UNIQUE (document_id, chunk_index)
);

ALTER TABLE IF EXISTS rag_chunk
  ALTER COLUMN embedding TYPE vector(1024)
  USING embedding::vector(1024);

DROP INDEX IF EXISTS idx_rag_chunk_embedding;
DROP INDEX IF EXISTS idx_rag_chunk_embedding_hnsw;

CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding_hnsw
  ON rag_chunk USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_tsv
  ON rag_chunk USING GIN (tsv);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_document
  ON rag_chunk (document_id);

CREATE TABLE IF NOT EXISTS rag_title (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL,
    language        TEXT NOT NULL,
    title           TEXT NOT NULL,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_rag_title_document
        FOREIGN KEY (document_id) REFERENCES dogv_documents(id),
    CONSTRAINT uq_rag_title_doc
        UNIQUE (document_id)
);

DROP INDEX IF EXISTS idx_rag_title_embedding_hnsw;
CREATE INDEX IF NOT EXISTS idx_rag_title_embedding_hnsw
  ON rag_title USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_title_document
  ON rag_title (document_id);
