CREATE TABLE IF NOT EXISTS ingest_gap_source_failures (
    id BIGSERIAL PRIMARY KEY,
    issue_date DATE NOT NULL,
    language VARCHAR(16) NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_checked_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (issue_date, language)
);

CREATE INDEX IF NOT EXISTS idx_gap_source_failures_due
    ON ingest_gap_source_failures (next_retry_at)
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_gap_source_failures_issue_lang
    ON ingest_gap_source_failures (issue_date, language);
