from __future__ import annotations

import time
from fastapi import APIRouter, Request

from app.settings import SETTINGS
from adapters.redis_client import get_redis

router = APIRouter()

_last_ready_check = {"t": 0.0, "status": {"ok": False}}


@router.get("/health")
async def health():
    # Liveness đơn giản: process còn sống / server chạy
    return {"ok": True}


@router.get("/ready")
async def ready(request: Request):
    """
    Readiness check:

    - orchestrator: object đã được build.
    - redis: ping OK.
    - llm: deep health từ background worker (gọi prompt ngắn định kỳ).
    - vector: deep health từ background worker (retriever.retrieve đơn giản).

    Tránh deep check trực tiếp mỗi lần /ready để không tốn tài nguyên.
    """
    now = time.time()
    if now - _last_ready_check["t"] < SETTINGS.READY_CHECK_CACHE_SEC:
        return _last_ready_check["status"]

    status = {
        "orchestrator": False,
        "redis": False,
        "vector": None,
        "llm": False,
        "ok": False,
    }

    app = request.app

    # Orchestrator object tồn tại
    orch = getattr(app.state, "orchestrator", None)
    status["orchestrator"] = bool(orch)

    # Redis
    try:
        r = get_redis()
        status["redis"] = (await r.ping()) is True
    except Exception:
        status["redis"] = False

    # Deep health state (được background worker cập nhật)
    deep = getattr(app.state, "deep_health", None)
    llm_deep_ok = bool(deep and deep.get("llm_ok", False))
    vector_deep = deep.get("vector_ok") if deep is not None else None

    # LLM: cần cả object tồn tại + deep health OK
    status["llm"] = bool(orch and getattr(orch, "llm", None) and llm_deep_ok)

    # Vector: nếu RAG bật thì dùng deep health, nếu không thì None (không bắt buộc)
    if SETTINGS.RAG_ENABLED:
        status["vector"] = vector_deep
    else:
        status["vector"] = None

    # Tổng thể ok:
    # - orchestrator ok
    # - redis ok
    # - llm ok (object + deep check ok)
    # - vector ok hoặc không bắt buộc (None)
    status["ok"] = (
        status["orchestrator"]
        and status["redis"]
        and status["llm"]
        and (status["vector"] in (True, None))
    )

    _last_ready_check["t"] = now
    _last_ready_check["status"] = status
    return status
