"""Configuration for the rag-worker module."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Redis settings
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_channel: str = field(default_factory=lambda: os.getenv("REDIS_CHANNEL", "document_links"))

    # PostgreSQL settings
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "second_memory"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))

    # LightRAG working directory (used for caching / temp files)
    working_dir: str = field(default_factory=lambda: os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage"))

    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536")))
    embedding_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))
    )

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
