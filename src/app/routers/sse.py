from __future__ import annotations

import time, asyncio
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
from observability.logging import get_logger
from observability.metrics import SSE_ACTIVE_CONNECTIONS

router = APIRouter(prefix="/sse-retrieve", tags=["sse"])

log = get_logger("sse")


def _last_event_id(request: Request) -> int:
    try:
        val = request.headers.get("Last-Event-ID")
        return int(val) if val else 0
    except Exception:
        return 0


@router.post(
    "/",
    response_class=StreamingResponse,
    dependencies=[Depends(require_role("user"))],
)
async def sse_retrieve(
    body: UserInput,
    request: Request,
    orch: Orchestrator = Depends(get_orchestrator),
    idt: Identity = Depends(require_role("user")),
):
    await check_rate_limit(request, user_id=idt.user_id)
    stream_sem: asyncio.Semaphore = request.app.state.stream_sem
    llm_sem = getattr(request.app.state, "llm_sem", None)

    async def event_gen() -> AsyncIterator[str]:
        # Hạn chế số stream SSE song song (stream_sem)
        async with stream_sem:
            # Nếu có semaphore chung cho LLM, dùng luôn để hạn chế LLM-heavy stream
            if llm_sem is not None:
                async with llm_sem:
                    SSE_ACTIVE_CONNECTIONS.inc()
                    try:
                        async for chunk in _stream_logic(body, request, orch):
                            yield chunk
                    finally:
                        SSE_ACTIVE_CONNECTIONS.dec()
            else:
                # Fallback: chỉ dùng stream_sem, không giới hạn LLM concurrency
                SSE_ACTIVE_CONNECTIONS.inc()
                try:
                    async for chunk in _stream_logic(body, request, orch):
                        yield chunk
                finally:
                    SSE_ACTIVE_CONNECTIONS.dec()

    async def _stream_logic(
        body: UserInput,
        request: Request,
        orch: Orchestrator,
    ) -> AsyncIterator[str]:
        session_id = body.session_id or "default"
        yield format_sse("metadata", {"session_id": session_id})
        if SETTINGS.SSE_RETRY_MS > 0:
            yield f"retry: {int(SETTINGS.SSE_RETRY_MS)}\n\n"

        resume_from = _last_event_id(request)
        eid = resume_from
        last_hb = time.monotonic()

        # Thiết lập timeout tổng cho stream
        stream_timeout = max(0, SETTINGS.SSE_STREAM_TIMEOUT_SEC)
        start_t = time.monotonic()
        deadline = start_t + stream_timeout if stream_timeout > 0 else None

        agen = orch.answer_sse_tokens(body, resume_from=resume_from)

        async def _anext_with_deadline():
            if deadline is None:
                # Không giới hạn thời gian sống stream
                return await agen.__anext__()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            return await asyncio.wait_for(agen.__anext__(), timeout=remaining)

        try:
            while True:
                # Kiểm tra client đã disconnect chưa (tránh giữ connection vô ích)
                if await request.is_disconnected():
                    break

                try:
                    tok = await _anext_with_deadline()
                except StopAsyncIteration:
                    # LLM stream xong bình thường
                    break
                except asyncio.TimeoutError:
                    # Hết thời gian sống của stream
                    eid += 1
                    log.warning(
                        "sse.timeout",
                        extra={
                            "session_id": session_id,
                            "timeout_sec": stream_timeout,
                        },
                    )
                    yield format_sse(
                        "done",
                        {"ok": False, "reason": "timeout"},
                        id=str(eid),
                    )
                    return

                eid += 1
                yield format_sse("delta", {"text": tok}, id=str(eid))

                now = time.monotonic()
                # Heartbeat để giữ connection sống
                if now - last_hb >= max(5, SETTINGS.SSE_HEARTBEAT_SEC):
                    yield ":keepalive\n\n"
                    last_hb = now

            # Kết thúc bình thường
            eid += 1
            yield format_sse("done", {"ok": True}, id=str(eid))

        finally:
            # cố gắng đóng generator nếu còn mở
            try:
                await agen.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
