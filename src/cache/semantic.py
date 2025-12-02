from __future__ import annotations
import json, hashlib
from typing import Optional, Dict, Any

from cache.kv import KV
from app.settings import SETTINGS
from observability.logging import get_logger

log = get_logger("cache.semantic")


def _key(kind: str, question: str, session_ctx: Dict[str, Any]) -> str:
    """
    Key semantic cache có dạng:
    sc:<version>:<sha256(kind+question+ctx)>
    VD: sc:v1:abcdef...
    """
    base = json.dumps(
        {"kind": kind, "q": question, "ctx": session_ctx},
        sort_keys=True,
        ensure_ascii=False,
    )
    version = getattr(SETTINGS, "SEMANTIC_CACHE_VERSION", "v1")
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    return f"sc:{version}:{digest}"


async def pre_cache_get(
    question: str, session_ctx: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not SETTINGS.CACHE_ENABLED:
        return None

    raw = await KV.get(_key("any", question, session_ctx))
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception as e:
        # Nếu schema thay đổi / JSON hỏng → bỏ qua cache
        log.warning(
            "cache.semantic.decode_failed",
            extra={"err": str(e)},
        )
        return None


async def post_cache_set(
    question: str,
    session_ctx: Dict[str, Any],
    response: Dict[str, Any],
) -> None:
    if not SETTINGS.CACHE_ENABLED:
        return

    # Shallow copy để không ảnh hưởng object gốc
    payload: Dict[str, Any] = dict(response)

    # Giới hạn text trong cache
    max_chars = max(
        0, int(getattr(SETTINGS, "SEMANTIC_CACHE_MAX_TEXT_CHARS", 0))
    )
    if max_chars > 0:
        # text (REST)
        txt = payload.get("text")
        if isinstance(txt, str) and len(txt) > max_chars:
            payload["text"] = txt[:max_chars] + "…"

        # frames (SSE)
        frames = payload.get("frames")
        if isinstance(frames, list):
            new_frames: list[str] = []
            used = 0
            for frame in frames:
                if not isinstance(frame, str):
                    continue
                if used >= max_chars:
                    break
                remaining = max_chars - used
                segment = frame if len(frame) <= remaining else frame[:remaining]
                new_frames.append(segment)
                used += len(segment)
            payload["frames"] = new_frames

    ttl = max(1, int(SETTINGS.CACHE_TTL_SEC))
    try:
        await KV.setex(
            _key("any", question, session_ctx),
            ttl,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as e:
        # KV.setex đã log lỗi Redis; đây chỉ phòng encode lỗi
        log.warning(
            "cache.semantic.store_failed",
            extra={"err": str(e)},
        )
