from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MediAssist AI"
    database_url: str = "sqlite:///./data/mediassist.db"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    redis_url: str = "redis://localhost:6379/0"

    max_upload_mb: int = Field(default=10, gt=0)

    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = Field(default=768, gt=0)
    llm_model: str = "gemini-3.5-flash-lite"

    chat_history_turns: int = Field(default=5, ge=0)
    chat_ttl_seconds: int = Field(default=3600, gt=0)

    retrieval_top_k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.5, ge=0, le=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()