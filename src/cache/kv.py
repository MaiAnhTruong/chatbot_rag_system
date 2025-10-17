# src/cache/kv.py
from __future__ import annotations
from typing import Any, Optional
from adapters.redis_client import get_redis

class RedisKV:
    def __init__(self):
        self._r = get_redis()

    async def setex(self, key: str, ttl_sec: int, value: Any) -> None:
        await self._r.setex(key, ttl_sec, value)

    async def get(self, key: str) -> Optional[str]:
        return await self._r.get(key)

KV = RedisKV()
