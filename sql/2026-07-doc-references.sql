-- Explicit norm cross-reference link table (doc A references doc B by N/YYYY).
-- Extracted at ingest time from title + body; used at read time to pull
-- companion documents (e.g. a resolución that resolves a convocatoria) into
-- the read set for the reader LLM.

CREATE TABLE IF NOT EXISTS doc_reference (
    id                     BIGSERIAL PRIMARY KEY,
    source_document_id     BIGINT NOT NULL,
    target_document_id     BIGINT NULL,
    ref_tipo               TEXT NOT NULL,
    -- NULL for date-referenced Resoluciones (frequently unnumbered; see
    -- api/doc_references.py disp_day/disp_month/disp_year path). Identity for
    -- dedup purposes lives in ref_key instead, since Postgres UNIQUE treats
    -- NULL columns as always-distinct (would break idempotent re-runs).
    ref_numero             INT NULL,
    ref_anyo               INT NULL,
    ref_kind               TEXT NOT NULL,
    -- Synthetic, always-non-null identity for the referenced norm:
    -- "<ref_tipo>:<ref_numero>/<ref_anyo>" or "<ref_tipo>:D<day>-<month>-<year>".
    ref_key                TEXT NOT NULL,
    raw_text               TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT fk_doc_reference_source
        FOREIGN KEY (source_document_id) REFERENCES dogv_documents(id),
    CONSTRAINT fk_doc_reference_target
        FOREIGN KEY (target_document_id) REFERENCES dogv_documents(id),
    CONSTRAINT uq_doc_reference
        UNIQUE (source_document_id, ref_key, ref_kind)
);

CREATE INDEX IF NOT EXISTS idx_doc_reference_source
  ON doc_reference (source_document_id);

CREATE INDEX IF NOT EXISTS idx_doc_reference_target
  ON doc_reference (target_document_id);

-- Case-insensitive prefix lookups (title ILIKE 'DECRETO 3/2020%') against the
-- 51k-row dogv_documents table are a full seq scan without this: target
-- resolution during the backfill does ~2 such lookups per extracted
-- reference. upper(title) + text_pattern_ops lets Postgres use a btree index
-- for `upper(title) LIKE upper(pattern)`.
CREATE INDEX IF NOT EXISTS idx_dogv_documents_title_upper_pattern
  ON dogv_documents (upper(title) text_pattern_ops);
