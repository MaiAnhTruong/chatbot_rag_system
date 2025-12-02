from __future__ import annotations
import redis.asyncio as redis

from app.settings import SETTINGS
from observability.logging import get_logger

log = get_logger("redis")

_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """
    Singleton Redis client (async).

    Đã cấu hình:
    - socket_timeout: thời gian chờ mỗi command.
    - socket_connect_timeout: thời gian chờ khi connect.
    - health_check_interval: kiểm tra connection định kỳ.
    - retry_on_timeout: retry command khi gặp timeout.
    """
    global _redis
    if _redis is None:
        log.info(
            "redis.client.init",
            extra={
                "url": SETTINGS.REDIS_URL,
                "socket_timeout": SETTINGS.REDIS_SOCKET_TIMEOUT_SEC,
                "connect_timeout": SETTINGS.REDIS_CONNECT_TIMEOUT_SEC,
                "health_check_interval": SETTINGS.REDIS_HEALTH_CHECK_INTERVAL_SEC,
                "retry_on_timeout": SETTINGS.REDIS_RETRY_ON_TIMEOUT,
            },
        )
        _redis = redis.from_url(
            SETTINGS.REDIS_URL,
            decode_responses=True,
            socket_timeout=SETTINGS.REDIS_SOCKET_TIMEOUT_SEC,
            socket_connect_timeout=SETTINGS.REDIS_CONNECT_TIMEOUT_SEC,
            health_check_interval=SETTINGS.REDIS_HEALTH_CHECK_INTERVAL_SEC,
            retry_on_timeout=SETTINGS.REDIS_RETRY_ON_TIMEOUT,
        )
    return _redis
