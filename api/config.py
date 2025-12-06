from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    dogv_base_url: str = "https://dogv.gva.es"
    database_url: str = "sqlite:///./data/sqlite/dogv.db"

    # NEW: local LLM via Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    ask_max_docs: int = 5

    class Config:
        env_file = ".env"
        env_prefix = ""
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
