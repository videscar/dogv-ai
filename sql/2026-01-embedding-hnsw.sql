-- Switch rag_chunk embeddings to bge-m3 dimensions and add HNSW index

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE IF EXISTS rag_chunk
  ALTER COLUMN embedding TYPE vector(1024)
  USING embedding::vector(1024);

DROP INDEX IF EXISTS idx_rag_chunk_embedding;

CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding_hnsw
  ON rag_chunk USING hnsw (embedding vector_cosine_ops);
