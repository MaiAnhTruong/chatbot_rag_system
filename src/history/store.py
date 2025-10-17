# src/history/store.py
from __future__ import annotations
import json
from typing import List
from adapters.redis_client import get_redis
from domain.schemas import Message

_r = get_redis()
_PREFIX = "hist:"

def _k(session_id: str) -> str:
    return f"{_PREFIX}{session_id}"

async def get(session_id: str) -> List[Message]:
    raw = await _r.get(_k(session_id))
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [Message(**m) for m in data]
    except Exception:
        return []

async def append(session_id: str, messages: List[Message]) -> None:
    current = await get(session_id)
    current.extend(messages)
    await _r.set(_k(session_id), json.dumps([m.model_dump() for m in current], ensure_ascii=False))

async def summarize_if_needed(session_id: str, keep_last: int = 6) -> None:
    msgs = await get(session_id)
    if len(msgs) > keep_last:
        trimmed = msgs[-keep_last:]
        await _r.set(_k(session_id), json.dumps([m.model_dump() for m in trimmed], ensure_ascii=False))
