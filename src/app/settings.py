# src/app/settings.py
from __future__ import annotations
from typing import Optional, Literal, List
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # api
    API_V1_STR: str = "v1"
    APP_NAME: str = "ragops"
    DEBUG: bool = False
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])
    MAX_REQUEST_SIZE_BYTES: int = 256 * 1024 

    # auth / rbac
    AUTH_MODE: Literal["none", "api_key"] = "none"
    API_KEYS: List[str] = Field(default_factory=list) 
    DEFAULT_ROLE: Literal["user", "admin"] = "user"

    # llm
    LLM_PROVIDER: Literal["openai", "lmstudio"] = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = Field(default="none")
    OPENAI_BASE_URL: Optional[AnyHttpUrl] = Field(default="https://api.openai.com/v1")
    LMSTUDIO_BASE_URL: Optional[AnyHttpUrl] = Field(default="http://localhost:1234/v1")

    # timeouts & retries
    LLM_REQUEST_TIMEOUT_SEC: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY_SEC: float = 0.6

    # embedding
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "huggingface"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    HF_API_TOKEN: Optional[str] = Field(default=None)

    # RAG
    RAG_ENABLED: bool = False
    VECTOR_BACKEND: str = "chroma"
    TOP_K: int = 3
    CHROMA_DIR: str = Field(default="./chroma_db")

    # Redis (cache & history & rate-limit & idempotency)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SEC: int = 30
    CACHE_DISTANCE_THRESHOLD: float = 0.2

    # rate limit
    RATE_LIMIT_RPM: int = 60  # requests per minute per key

    # SSE
    SSE_HEARTBEAT_SEC: int = 15
    SSE_RETRY_MS: int = 0
    MAX_CONCURRENT_STREAMS: int = 100  # semaphore giới hạn stream song song

    # readiness
    READY_CHECK_CACHE_SEC: int = 20

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @property
    def LLM_BASE_URL(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return str(self.OPENAI_BASE_URL)
        elif self.LLM_PROVIDER == "lmstudio":
            return str(self.LMSTUDIO_BASE_URL)
        return str(self.OPENAI_BASE_URL)

    @property
    def EMBEDDING_BASE_URL(self) -> Optional[str]:
        if self.EMBEDDING_PROVIDER == "openai":
            return str(self.OPENAI_BASE_URL)
        elif self.EMBEDDING_PROVIDER == "huggingface":
            return f"https://api-inference.huggingface.co/{self.EMBEDDING_MODEL}"
        return None

SETTINGS = Settings()
