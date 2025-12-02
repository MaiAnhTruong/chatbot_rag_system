from __future__ import annotations

import time
from fastapi import HTTPException, status, Request
from prometheus_client import Counter

from adapters.redis_client import get_redis
from app.settings import SETTINGS
from observability.logging import get_logger

log = get_logger("rate_limit")

_r = get_redis()

# Sliding window length (giây)
_WINDOW_SEC = 60

# Metric: số lần bị rate limit (cardinality thấp, chỉ theo path)
RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Number of requests blocked by the rate limiter",
    ["path"],
)


def _get_client_ip(request: Request) -> str | None:
    """
    Lấy IP thực của client khi đứng sau reverse proxy.

    Ưu tiên:
    - X-Forwarded-For: lấy IP đầu tiên trong chuỗi (client gốc)
    - X-Real-IP
    - request.client.host (fallback)

    Lưu ý: cần cấu hình LB/proxy set các header này một cách đáng tin cậy.
    """
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        # Chuỗi dạng "client_ip, proxy1, proxy2"
        ip = xff.split(",")[0].strip()
        if ip:
            return ip

    xri = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
    if xri:
        return xri.strip()

    return request.client.host if request.client else None


def _key(ident: str) -> str:
    # Sliding window: 1 key cho mỗi "ident" (user_id hoặc IP)
    return f"rl:{ident}"


async def check_rate_limit(request: Request, user_id: str | None = None) -> None:
    """
    Rate-limit per user_id / IP với sliding window (ZSET).

    - Mỗi request:
      1. Xoá các điểm cũ ngoài cửa sổ _WINDOW_SEC.
      2. Đếm số request còn lại trong cửa sổ.
      3. Thêm request hiện tại (timestamp) vào ZSET.
    - Nếu count >= RATE_LIMIT_RPM → trả 429.

    Resilience:
    - Nếu Redis lỗi (timeout, connection error, ...) → log cảnh báo,
      nhưng KHÔNG chặn request (fallback "allow").

    Identity:
    - Ưu tiên user_id (từ auth).
    - Nếu không có → dùng IP thực (_get_client_ip).
    """
    limit = max(1, SETTINGS.RATE_LIMIT_RPM)

    client_ip = _get_client_ip(request)
    ident = user_id or client_ip or "unknown"

    key = _key(ident)
    now = time.time()
    window_start = now - _WINDOW_SEC

    count: int | None = None
    try:
        # Sliding window với ZSET
        pipe = _r.pipeline(transaction=True)
        # 1) Xoá các request cũ ngoài cửa sổ
        pipe.zremrangebyscore(key, 0, window_start)
        # 2) Đếm số request hiện tại trong cửa sổ
        pipe.zcard(key)
        # 3) Thêm request hiện tại (member = timestamp string, score = timestamp)
        pipe.zadd(key, {str(now): now})
        # 4) Đặt TTL để tránh key tồn tại mãi
        pipe.expire(key, _WINDOW_SEC * 2)
        _, count, _, _ = await pipe.execute()
    except Exception as e:
        # Redis lỗi -> fallback cho phép request, chấp nhận rủi ro abuse
        log.warning(
            "ratelimit.redis_failed",
            extra={
                "key": key,
                "user_id": user_id,
                "client_ip": client_ip,
                "err": str(e),
            },
        )
        return

    if count is not None and count >= limit:
        path = request.url.path
        RATE_LIMIT_HITS.labels(path=path).inc()
        log.warning(
            "ratelimit.exceeded",
            extra={
                "user_id": user_id,
                "client_ip": client_ip,
                "path": path,
                "limit": limit,
                "count": int(count),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
