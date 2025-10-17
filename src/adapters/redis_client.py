# src/adapters/redis_client.py
from __future__ import annotations
import redis.asyncio as redis
from app.settings import SETTINGS

_redis: redis.Redis | None = None

def get_redis() -> redis.Redis:
    # Không cần async ở đây; client trả về object async (methods awaitable)
    global _redis
    if _redis is None:
        _redis = redis.from_url(SETTINGS.REDIS_URL, decode_responses=True)
    return _redis
