from dataclasses import dataclass
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_LANES = {"vector", "bm25", "title"}


@dataclass(frozen=True)
class InfraPolicy:
    database_url: str
    dogv_db_dsn: str | None
    dogv_base_url: str
    llm_base_url: str
    llm_model: str
    llm_timeout: int
    llm_max_tokens: int
    embed_base_url: str
    embed_model: str
    embed_timeout: int
    embedding_dim: int


@dataclass(frozen=True)
class ChunkingPolicy:
    min_tokens: int
    max_tokens: int
    overlap_tokens: int


@dataclass(frozen=True)
class RetrievalPolicy:
    max_docs: int
    min_docs: int
    max_facets: int
    max_docs_expand: int
    bm25_max_docs: int
    lanes: str
    doc_confidence_min: float
    rrf_expand_margin_ratio: float
    rrf_margin_probe: int
    rrf_weight_vector: float
    rrf_weight_bm25: float
    rrf_weight_title: float
    rrf_weight_title_lexical: float
    fallback_allow_margin: bool
    bm25_fuse_weight_chunk: float
    bm25_fuse_weight_strict: float
    bm25_fuse_weight_title: float
    bm25_strict_primary_min: int
    bm25_prf_docs: int
    bm25_prf_terms: int


@dataclass(frozen=True)
class ReadingPolicy:
    read_max_docs: int
    chunks_per_doc: int
    chunk_max_chars: int
    doc_fallback_chars: int
    rerank_top_n: int
    rerank_max_candidates: int
    rerank_expand_candidates: int
    rerank_expand_top_n: int
    rerank_coverage_keep: int
    rerank_recent_keep: int
    read_expand_docs: int
    read_coverage_docs: int
    read_eligibility_docs: int
    read_amount_docs: int
    full_doc_max_chars: int
    full_doc_max_docs: int
    full_doc_total_chars: int


@dataclass(frozen=True)
class TemporalPolicy:
    policy: str
    timezone: str
    week_start: str
    feed_recent_days: int
    llm_expand: bool


@dataclass(frozen=True)
class AutoIngestPolicy:
    enabled: bool
    max_days: int
    keep_months: int
    languages: str
    startup_enabled: bool
    startup_blocking: bool
    startup_purge_old: bool
    startup_repair_gaps: bool
    startup_lock_id: int
    gap_check_retries: int
    gap_check_backoff_seconds: float
    gap_repair_scan_max_days: int
    backfill_enabled: bool
    hot_index_months: int
    warm_index_months: int


@dataclass(frozen=True)
class AnswerPolicy:
    mutators_enabled: bool
    missing_notes_enabled: bool
    validator_enabled: bool
    repair_attempts: int
    claim_guard_mode: str
    repair_mode: str
    fallback_style: str
    fallback_max_items: int


@dataclass(frozen=True)
class DemoPolicy:
    enforce_ready_gate: bool
    request_timeout_seconds: int
    trace_enabled: bool


class Settings(BaseSettings):
    # Infra (DB / LLM / embeddings)
    dogv_base_url: str = "https://dogv.gva.es"
    database_url: str = "postgresql+psycopg2://dogv_ai:dogv_ai@localhost:5432/dogv_ai"
    dogv_db_dsn: str | None = None
    llm_base_url: str = "http://127.0.0.1:8000"
    llm_model: str = "qwen3.6-27b"
    llm_timeout: int = 300
    llm_max_tokens: int = 8192
    embed_base_url: str = "http://127.0.0.1:8001"
    embed_model: str = "bge-m3"
    embed_timeout: int = 60
    embedding_dim: int = 1024

    # Chunking
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80

    # Retrieval (lanes, RRF, BM25)
    ask_max_docs: int = 20
    ask_min_docs: int = 3
    ask_max_facets: int = 3
    ask_max_docs_expand: int = 20
    ask_bm25_max_docs: int = 50
    ask_lanes: str = "vector,bm25,title"
    ask_doc_confidence_min: float = 0.06
    ask_rrf_expand_margin_ratio: float = 0.12
    ask_rrf_margin_probe: int = 5
    ask_rrf_weight_vector: float = 1.0
    ask_rrf_weight_bm25: float = 1.0
    ask_rrf_weight_title: float = 1.0
    ask_rrf_weight_title_lexical: float = 0.8
    ask_fallback_allow_margin: bool = False
    bm25_fuse_weight_chunk: float = 1.0
    bm25_fuse_weight_strict: float = 1.2
    bm25_fuse_weight_title: float = 0.9
    bm25_strict_primary_min: int = 1
    bm25_prf_docs: int = 5
    bm25_prf_terms: int = 6

    # Reading / reranking
    ask_read_max_docs: int = 3
    ask_chunks_per_doc: int = 4
    ask_chunk_max_chars: int = 1200
    ask_doc_fallback_chars: int = 12000
    ask_rerank_top_n: int = 5
    ask_rerank_max_candidates: int = 10
    ask_rerank_expand_candidates: int = 10
    ask_rerank_expand_top_n: int = 2
    ask_rerank_coverage_keep: int = 4
    ask_rerank_recent_keep: int = 2
    ask_read_expand_docs: int = 2
    ask_read_coverage_docs: int = 2
    ask_read_eligibility_docs: int = 1
    ask_read_amount_docs: int = 1
    full_doc_max_chars: int = 120000
    full_doc_max_docs: int = 2
    full_doc_total_chars: int = 200000

    # Temporal
    ask_temporal_policy: str = "filter"
    temporal_timezone: str = "Europe/Madrid"
    temporal_week_start: str = "monday"
    feed_recent_days: int = 21
    ask_llm_expand: bool = True

    # Auto ingest
    auto_ingest_enabled: bool = False
    auto_ingest_max_days: int = 15
    auto_ingest_keep_months: int = 12
    auto_ingest_languages: str = "es_es,va_va"
    auto_ingest_startup_enabled: bool = True
    auto_ingest_startup_blocking: bool = False
    auto_ingest_startup_purge_old: bool = True
    auto_ingest_startup_repair_gaps: bool = True
    auto_ingest_startup_lock_id: int = 190021
    auto_ingest_gap_check_retries: int = 3
    auto_ingest_gap_check_backoff_seconds: float = 1.5
    auto_ingest_gap_repair_scan_max_days: int = 45
    backfill_enabled: bool = True
    hot_index_months: int = 6
    warm_index_months: int = 24

    # Answer pipeline
    answer_mutators_enabled: bool = False
    answer_missing_notes_enabled: bool = False
    answer_validator_enabled: bool = True
    answer_repair_attempts: int = 1
    answer_claim_guard_mode: str = "unit_aware_strict"
    answer_repair_mode: str = "conditional"
    answer_fallback_style: str = "concise_summary"
    answer_fallback_max_items: int = 3

    # Demo / observability
    demo_enforce_ready_gate: bool = True
    demo_request_timeout_seconds: int = 60
    trace_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def infra(self) -> InfraPolicy:
        return InfraPolicy(
            database_url=self.database_url,
            dogv_db_dsn=self.dogv_db_dsn,
            dogv_base_url=self.dogv_base_url,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_timeout=self.llm_timeout,
            llm_max_tokens=self.llm_max_tokens,
            embed_base_url=self.embed_base_url,
            embed_model=self.embed_model,
            embed_timeout=self.embed_timeout,
            embedding_dim=self.embedding_dim,
        )

    @property
    def chunking(self) -> ChunkingPolicy:
        return ChunkingPolicy(
            min_tokens=self.chunk_min_tokens,
            max_tokens=self.chunk_max_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )

    @property
    def retrieval(self) -> RetrievalPolicy:
        return RetrievalPolicy(
            max_docs=self.ask_max_docs,
            min_docs=self.ask_min_docs,
            max_facets=self.ask_max_facets,
            max_docs_expand=self.ask_max_docs_expand,
            bm25_max_docs=self.ask_bm25_max_docs,
            lanes=self.ask_lanes,
            doc_confidence_min=self.ask_doc_confidence_min,
            rrf_expand_margin_ratio=self.ask_rrf_expand_margin_ratio,
            rrf_margin_probe=self.ask_rrf_margin_probe,
            rrf_weight_vector=self.ask_rrf_weight_vector,
            rrf_weight_bm25=self.ask_rrf_weight_bm25,
            rrf_weight_title=self.ask_rrf_weight_title,
            rrf_weight_title_lexical=self.ask_rrf_weight_title_lexical,
            fallback_allow_margin=self.ask_fallback_allow_margin,
            bm25_fuse_weight_chunk=self.bm25_fuse_weight_chunk,
            bm25_fuse_weight_strict=self.bm25_fuse_weight_strict,
            bm25_fuse_weight_title=self.bm25_fuse_weight_title,
            bm25_strict_primary_min=self.bm25_strict_primary_min,
            bm25_prf_docs=self.bm25_prf_docs,
            bm25_prf_terms=self.bm25_prf_terms,
        )

    @property
    def reading(self) -> ReadingPolicy:
        return ReadingPolicy(
            read_max_docs=self.ask_read_max_docs,
            chunks_per_doc=self.ask_chunks_per_doc,
            chunk_max_chars=self.ask_chunk_max_chars,
            doc_fallback_chars=self.ask_doc_fallback_chars,
            rerank_top_n=self.ask_rerank_top_n,
            rerank_max_candidates=self.ask_rerank_max_candidates,
            rerank_expand_candidates=self.ask_rerank_expand_candidates,
            rerank_expand_top_n=self.ask_rerank_expand_top_n,
            rerank_coverage_keep=self.ask_rerank_coverage_keep,
            rerank_recent_keep=self.ask_rerank_recent_keep,
            read_expand_docs=self.ask_read_expand_docs,
            read_coverage_docs=self.ask_read_coverage_docs,
            read_eligibility_docs=self.ask_read_eligibility_docs,
            read_amount_docs=self.ask_read_amount_docs,
            full_doc_max_chars=self.full_doc_max_chars,
            full_doc_max_docs=self.full_doc_max_docs,
            full_doc_total_chars=self.full_doc_total_chars,
        )

    @property
    def temporal(self) -> TemporalPolicy:
        return TemporalPolicy(
            policy=self.ask_temporal_policy,
            timezone=self.temporal_timezone,
            week_start=self.temporal_week_start,
            feed_recent_days=self.feed_recent_days,
            llm_expand=self.ask_llm_expand,
        )

    @property
    def auto_ingest(self) -> AutoIngestPolicy:
        return AutoIngestPolicy(
            enabled=self.auto_ingest_enabled,
            max_days=self.auto_ingest_max_days,
            keep_months=self.auto_ingest_keep_months,
            languages=self.auto_ingest_languages,
            startup_enabled=self.auto_ingest_startup_enabled,
            startup_blocking=self.auto_ingest_startup_blocking,
            startup_purge_old=self.auto_ingest_startup_purge_old,
            startup_repair_gaps=self.auto_ingest_startup_repair_gaps,
            startup_lock_id=self.auto_ingest_startup_lock_id,
            gap_check_retries=self.auto_ingest_gap_check_retries,
            gap_check_backoff_seconds=self.auto_ingest_gap_check_backoff_seconds,
            gap_repair_scan_max_days=self.auto_ingest_gap_repair_scan_max_days,
            backfill_enabled=self.backfill_enabled,
            hot_index_months=self.hot_index_months,
            warm_index_months=self.warm_index_months,
        )

    @property
    def answer(self) -> AnswerPolicy:
        return AnswerPolicy(
            mutators_enabled=self.answer_mutators_enabled,
            missing_notes_enabled=self.answer_missing_notes_enabled,
            validator_enabled=self.answer_validator_enabled,
            repair_attempts=self.answer_repair_attempts,
            claim_guard_mode=self.answer_claim_guard_mode,
            repair_mode=self.answer_repair_mode,
            fallback_style=self.answer_fallback_style,
            fallback_max_items=self.answer_fallback_max_items,
        )

    @property
    def demo(self) -> DemoPolicy:
        return DemoPolicy(
            enforce_ready_gate=self.demo_enforce_ready_gate,
            request_timeout_seconds=self.demo_request_timeout_seconds,
            trace_enabled=self.trace_enabled,
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Convenience singleton for modules/notebooks that prefer direct import
settings = get_settings()


def enabled_lanes(settings_obj: Settings) -> set[str]:
    raw = (settings_obj.ask_lanes or "").strip()
    if not raw:
        return set(ALLOWED_LANES)
    tokens = {lane.strip().lower() for lane in raw.split(",") if lane.strip()}
    if "all" in tokens:
        return set(ALLOWED_LANES)
    lanes = {lane for lane in tokens if lane in ALLOWED_LANES}
    return lanes or {"vector"}
