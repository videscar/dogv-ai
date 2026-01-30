-- RAG chunk storage for DOGV documents

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunk (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    language        TEXT NOT NULL,
    text            TEXT NOT NULL,
    embedding       vector(1536),
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_rag_chunk_document
        FOREIGN KEY (document_id) REFERENCES dogv_documents(id)
);

ALTER TABLE rag_chunk
  ADD COLUMN IF NOT EXISTS tsv tsvector;

CREATE INDEX IF NOT EXISTS idx_rag_chunk_tsv
  ON rag_chunk USING GIN (tsv);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding
  ON rag_chunk USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
