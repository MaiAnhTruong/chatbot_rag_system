# src/app/routers/health.py
from __future__ import annotations
import time
from fastapi import APIRouter, Request
from app.settings import SETTINGS
from adapters.redis_client import get_redis

router = APIRouter()
_last_ready_check = {"t": 0.0, "status": {"ok": False}}

@router.get("/health")
async def health():
    return {"ok": True}

@router.get("/ready")
async def ready(request: Request):
    now = time.time()
    if now - _last_ready_check["t"] < SETTINGS.READY_CHECK_CACHE_SEC:
        return _last_ready_check["status"]

    status = {"orchestrator": False, "redis": False, "vector": None, "llm": False, "ok": False}
    orch = getattr(request.app.state, "orchestrator", None)
    status["orchestrator"] = bool(orch)

    # Redis
    try:
        r = get_redis()
        status["redis"] = (await r.ping()) is True
    except Exception:
        status["redis"] = False

    # Vector
    if SETTINGS.RAG_ENABLED:
        try:
            from vector.chroma import ChromaVectorStore  # noqa
            status["vector"] = True
        except Exception:
            status["vector"] = False

    # LLM
    status["llm"] = bool(orch and getattr(orch, "llm", None))

    status["ok"] = status["orchestrator"] and status["redis"] and status["llm"] and (status["vector"] in (True, None))
    _last_ready_check["t"] = now
    _last_ready_check["status"] = status
    return status
