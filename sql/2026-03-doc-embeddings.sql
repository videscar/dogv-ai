-- Doc-level embeddings for DOGV documents (title + summary + metadata)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_doc (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT NOT NULL,
    language        TEXT NOT NULL,
    title           TEXT NOT NULL,
    summary         TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(1024),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_rag_doc_document
        FOREIGN KEY (document_id) REFERENCES dogv_documents(id),
    CONSTRAINT uq_rag_doc_doc
        UNIQUE (document_id)
);

DROP INDEX IF EXISTS idx_rag_doc_embedding_hnsw;
CREATE INDEX IF NOT EXISTS idx_rag_doc_embedding_hnsw
  ON rag_doc USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_rag_doc_document
  ON rag_doc (document_id);
