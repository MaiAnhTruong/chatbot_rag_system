# src/app/routers/sse.py
from __future__ import annotations
import time
from typing import AsyncIterator
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from domain.schemas import UserInput
from app.dependencies import get_orchestrator
from rag.orchestrator import Orchestrator
from security.auth import require_role, Identity
from ops.rate_limit import check_rate_limit
from app.settings import SETTINGS
from utils.sse import format_sse

router = APIRouter(prefix="/sse-retrieve", tags=["sse"])

def _last_event_id(request: Request) -> int:
    try:
        val = request.headers.get("Last-Event-ID")
        return int(val) if val else 0
    except Exception:
        return 0

@router.post("/", response_class=StreamingResponse, dependencies=[Depends(require_role("user"))])
async def sse_retrieve(
    body: UserInput,
    request: Request,
    orch: Orchestrator = Depends(get_orchestrator),
    idt: Identity = Depends(require_role("user")),
):
    await check_rate_limit(request, user_id=idt.user_id)
    sem = request.app.state.stream_sem

    async def event_gen() -> AsyncIterator[str]:
        async with sem:
            yield format_sse("metadata", {"session_id": body.session_id or "default"})
            if SETTINGS.SSE_RETRY_MS > 0:
                yield f"retry: {int(SETTINGS.SSE_RETRY_MS)}\n\n"

            resume_from = _last_event_id(request)
            eid = resume_from
            last_hb = time.monotonic()

            async for tok in orch.answer_sse_tokens(body, resume_from=resume_from):
                if await request.is_disconnected():
                    break
                eid += 1
                yield format_sse("delta", {"text": tok}, id=str(eid))
                now = time.monotonic()
                if now - last_hb >= max(5, SETTINGS.SSE_HEARTBEAT_SEC):
                    yield ":keepalive\n\n"
                    last_hb = now

            eid += 1
            yield format_sse("done", {"ok": True}, id=str(eid))

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
