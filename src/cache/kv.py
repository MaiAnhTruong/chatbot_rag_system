from __future__ import annotations
from typing import Any, Optional

from adapters.redis_client import get_redis
from observability.logging import get_logger

log = get_logger("cache.kv")


class RedisKV:
    def __init__(self):
        self._r = get_redis()

    async def setex(self, key: str, ttl_sec: int, value: Any) -> None:
        try:
            await self._r.setex(key, ttl_sec, value)
        except Exception as e:
            # Cache lỗi -> chỉ log, không làm fail business flow
            log.warning(
                "cache.redis.setex_failed",
                extra={"key": key, "ttl": ttl_sec, "err": str(e)},
            )

    async def get(self, key: str) -> Optional[str]:
        try:
            return await self._r.get(key)
        except Exception as e:
            # Cache lỗi -> coi như cache miss
            log.warning(
                "cache.redis.get_failed",
                extra={"key": key, "err": str(e)},
            )
            return None


KV = RedisKV()
