# src/cache/semantic.py
from __future__ import annotations
import json, hashlib
from typing import Optional, Dict, Any
from cache.kv import KV
from app.settings import SETTINGS

def _key(kind: str, question: str, session_ctx: Dict[str, Any]) -> str:
    base = json.dumps({"kind": kind, "q": question, "ctx": session_ctx}, sort_keys=True, ensure_ascii=False)
    return "sc:" + hashlib.sha256(base.encode("utf-8")).hexdigest()

async def pre_cache_get(question: str, session_ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not SETTINGS.CACHE_ENABLED:
        return None
    data = await KV.get(_key("any", question, session_ctx))
    if not data:
        return None
    return json.loads(data)

async def post_cache_set(question: str, session_ctx: Dict[str, Any], response: Dict[str, Any]) -> None:
    if not SETTINGS.CACHE_ENABLED:
        return
    ttl = max(1, int(SETTINGS.CACHE_TTL_SEC))
    await KV.setex(_key("any", question, session_ctx), ttl, json.dumps(response, ensure_ascii=False))
