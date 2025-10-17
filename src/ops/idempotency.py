# src/ops/idempotency.py
from __future__ import annotations
import json
from typing import Optional
from adapters.redis_client import get_redis

_r = get_redis()
_PREFIX = "idem:"

async def get_cached(idem_key: str) -> Optional[dict]:
    raw = await _r.get(_PREFIX + idem_key)
    if raw:
        return json.loads(raw)
    return None

async def set_cached(idem_key: str, payload: dict, ttl_sec: int = 300) -> None:
    await _r.setex(_PREFIX + idem_key, ttl_sec, json.dumps(payload, ensure_ascii=False))
