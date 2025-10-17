# src/rag/streaming.py
from __future__ import annotations
import time
from typing import Iterable, Iterator
from utils.sse import format_sse
from app.settings import SETTINGS

def sse_stream_from_tokens(tokens: Iterable[str], session_id: str) -> Iterator[str]:
    # metadata + retry hint (nếu muốn)
    yield format_sse("metadata", {"session_id": session_id})
    if SETTINGS.SSE_RETRY_MS > 0:
        yield f"retry: {int(SETTINGS.SSE_RETRY_MS)}\n\n"

    last_beat = time.monotonic()
    eid = 0
    for t in tokens:
        eid += 1
        yield format_sse("delta", {"text": t}, id=str(eid))
        # heartbeat
        now = time.monotonic()
        if now - last_beat >= max(5, SETTINGS.SSE_HEARTBEAT_SEC):
            yield ":keepalive\n\n"
            last_beat = now

    eid += 1
    yield format_sse("done", {"ok": True}, id=str(eid))
