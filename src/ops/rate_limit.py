# src/ops/rate_limit.py
from __future__ import annotations
import time
from fastapi import HTTPException, status, Request
from adapters.redis_client import get_redis
from app.settings import SETTINGS

_r = get_redis()

def _key(k: str) -> str:
    now_min = int(time.time() // 60)
    return f"rl:{k}:{now_min}"

async def check_rate_limit(request: Request, user_id: str | None = None) -> None:
    limit = max(1, SETTINGS.RATE_LIMIT_RPM)
    ident = user_id or (request.client.host if request.client else "unknown")
    key = _key(ident)
    n = await _r.incr(key)
    if n == 1:
        await _r.expire(key, 60)
    if n > limit:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
