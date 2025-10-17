# src/utils/sse.py
from __future__ import annotations
from typing import Optional, Dict, Any
import json

def format_sse(event: str, data: Dict[str, Any], id: Optional[str] = None) -> str:
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    return "\n".join(lines) + "\n\n"
