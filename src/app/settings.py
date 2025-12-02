from __future__ import annotations
from typing import Optional, Literal, List
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env
    ENV: Literal["dev", "staging", "prod"] = "dev"

    # api
    API_V1_STR: str = "v1"
    APP_NAME: str = "ragops"
    DEBUG: bool = False

    # CORS
    CORS_ALLOW_ORIGINS: List[str] = Field(default_factory=list)

    MAX_REQUEST_SIZE_BYTES: int = 256 * 1024

    # auth / rbac
    AUTH_MODE: Literal["none", "api_key"] = "none"
    API_KEYS: List[str] = Field(default_factory=list)
    DEFAULT_ROLE: Literal["user", "admin"] = "user"

    # llm
    LLM_PROVIDER: Literal["openai", "lmstudio"] = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_BASE_URL: Optional[AnyHttpUrl] = Field(default="https://api.openai.com/v1")
    LMSTUDIO_BASE_URL: Optional[AnyHttpUrl] = Field(default="http://localhost:1234/v1")

    # LLM cost & behaviour tuning
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0

    # Circuit breaker cho LLM
    LLM_CB_ENABLED: bool = True
    LLM_CB_FAIL_THRESHOLD: int = 5
    LLM_CB_OPEN_SEC: int = 30

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
    REDIS_SOCKET_TIMEOUT_SEC: float = 2.0
    REDIS_CONNECT_TIMEOUT_SEC: float = 2.0
    REDIS_HEALTH_CHECK_INTERVAL_SEC: int = 30
    REDIS_RETRY_ON_TIMEOUT: bool = True

    # cache (semantic cache, history, v.v.)
    CACHE_ENABLED: bool = True
    CACHE_TTL_SEC: int = 120
    CACHE_DISTANCE_THRESHOLD: float = 0.2

    # Semantic cache tuning
    SEMANTIC_CACHE_VERSION: str = "v1"
    SEMANTIC_CACHE_MAX_TEXT_CHARS: int = 4000

    # rate limit
    RATE_LIMIT_RPM: int = 60

    # SSE
    SSE_HEARTBEAT_SEC: int = 15
    SSE_RETRY_MS: int = 0
    # semaphore giới hạn số stream SSE song song
    MAX_CONCURRENT_STREAMS: int = 100
    # Timeout tổng cho một SSE stream (giây). <=0 nghĩa là không giới hạn.
    SSE_STREAM_TIMEOUT_SEC: int = 180
    # Incremental cache SSE
    SSE_CACHE_FLUSH_EVERY_N_TOKENS: int = 20
    # Resume: quay lại thêm N token cuối khi dùng Last-Event-ID
    SSE_RESUME_OVERLAP_TOKENS: int = 3

    # Concurrency LLM chung cho REST + SSE
    LLM_MAX_CONCURRENT_REQUESTS: int = 32

    # readiness
    READY_CHECK_CACHE_SEC: int = 20

    # Deep health check
    LLM_HEALTH_CHECK_INTERVAL_SEC: int = 60
    VECTOR_HEALTH_CHECK_INTERVAL_SEC: int = 120

    # Privacy / PII
    PRIVACY_STRICT_MODE: bool = False

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
