-- Add doc_kind labels and trace storage

ALTER TABLE IF EXISTS dogv_documents
  ADD COLUMN IF NOT EXISTS doc_kind VARCHAR NULL,
  ADD COLUMN IF NOT EXISTS doc_kind_confidence DOUBLE PRECISION NULL;

CREATE INDEX IF NOT EXISTS idx_dogv_documents_doc_kind
  ON dogv_documents (doc_kind);

-- Optional trace storage for debug runs
CREATE TABLE IF NOT EXISTS qa_traces (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NULL,
    question TEXT NOT NULL,
    filters JSONB,
    plan JSONB,
    lanes JSONB,
    fusion JSONB,
    rerank JSONB,
    evidence JSONB,
    answer TEXT,
    verification JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_qa_traces_session_id
  ON qa_traces (session_id);
