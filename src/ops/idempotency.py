from __future__ import annotations
import json, hashlib
from typing import Optional

from adapters.redis_client import get_redis
from observability.logging import get_logger

log = get_logger("idempotency")

_r = get_redis()
_PREFIX = "idem:"
_VERSION = "v1"


def _make_key(idem_key: str, method: str, path: str) -> str:
    """
    Key idempotency gồm:
    - HTTP method
    - path (không query string)
    - Idempotency-Key header

    Dạng key trong Redis:
    idem:v1:<sha256("METHOD path idem_key")>
    """
    base = f"{method.upper()} {path} {idem_key}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"{_PREFIX}{_VERSION}:{digest}"


async def get_cached(
    idem_key: str,
    method: str,
    path: str,
) -> Optional[dict]:
    key = _make_key(idem_key, method, path)
    try:
        raw = await _r.get(key)
    except Exception as e:
        # Redis lỗi -> idempotency không hoạt động, nhưng không chặn request
        log.warning(
            "idem.redis.get_failed",
            extra={"key": key, "method": method, "path": path, "err": str(e)},
        )
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception as e:
        log.warning(
            "idem.decode_failed",
            extra={"key": key, "err": str(e)},
        )
        return None


async def set_cached(
    idem_key: str,
    payload: dict,
    method: str,
    path: str,
    ttl_sec: int = 300,
) -> None:
    key = _make_key(idem_key, method, path)
    try:
        data = json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        # Payload không serializable -> bỏ qua idempotency, chỉ log
        log.warning(
            "idem.encode_failed",
            extra={"key": key, "err": str(e)},
        )
        return

    try:
        await _r.setex(key, ttl_sec, data)
    except Exception as e:
        # Redis lỗi -> chỉ log, không làm fail request
        log.warning(
            "idem.redis.set_failed",
            extra={"key": key, "ttl": ttl_sec, "err": str(e)},
        )
