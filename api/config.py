from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_LANES = {"vector", "bm25", "title"}

class Settings(BaseSettings):
    dogv_base_url: str = "https://dogv.gva.es"
    database_url: str = "postgresql+psycopg2://dogv_ai:dogv_ai@localhost:5432/dogv_ai"
    dogv_db_dsn: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss-20b-high"
    ollama_embed_model: str = "bge-m3"
    ollama_timeout: int = 300
    ollama_num_ctx: int = 65536
    embedding_dim: int = 1024
    chunk_min_tokens: int = 300
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 80
    ask_max_docs: int = 20
    ask_min_docs: int = 3
    ask_max_facets: int = 3
    ask_read_max_docs: int = 3
    ask_chunks_per_doc: int = 4
    ask_chunk_max_chars: int = 1200
    ask_doc_fallback_chars: int = 12000
    ask_rerank_top_n: int = 5
    ask_rerank_max_candidates: int = 10
    ask_rerank_expand_candidates: int = 10
    ask_rerank_expand_top_n: int = 2
    ask_rerank_coverage_keep: int = 4
    ask_bm25_max_docs: int = 50
    feed_recent_days: int = 21
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
    ask_max_docs_expand: int = 20
    ask_read_expand_docs: int = 2
    ask_read_coverage_docs: int = 2
    ask_read_eligibility_docs: int = 1
    ask_read_amount_docs: int = 1
    full_doc_max_chars: int = 120000
    full_doc_max_docs: int = 2
    full_doc_total_chars: int = 200000
    ask_temporal_policy: str = "filter"
    temporal_timezone: str = "Europe/Madrid"
    temporal_week_start: str = "monday"
    ask_llm_expand: bool = True
    auto_ingest_enabled: bool = True
    auto_ingest_max_days: int = 15
    auto_ingest_keep_months: int = 12
    auto_ingest_languages: str = "es_es,va_va"
    auto_ingest_startup_enabled: bool = True
    auto_ingest_startup_blocking: bool = False
    auto_ingest_startup_purge_old: bool = True
    auto_ingest_startup_repair_gaps: bool = True
    auto_ingest_startup_lock_id: int = 190021
    backfill_enabled: bool = True
    hot_index_months: int = 6
    warm_index_months: int = 24
    demo_enforce_ready_gate: bool = True
    demo_request_timeout_seconds: int = 60
    trace_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
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
