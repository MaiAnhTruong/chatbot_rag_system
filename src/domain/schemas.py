# src/domain/schemas.py
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

class RetrievalResult(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class UserInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ResponseOutput(BaseModel):
    text: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    # ➜ thêm "degraded" để hợp lệ khi fallback (429/quota hoặc lỗi provider)
    finish_reason: Literal["stop", "length", "blocked", "error", "degraded"] = "stop"

class SSEEvent(BaseModel):
    event: str = "message"
    data: Dict[str, Any]
    id: Optional[str] = None
