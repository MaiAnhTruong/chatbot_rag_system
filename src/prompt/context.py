# src/prompt/context.py
from __future__ import annotations
from typing import List
from domain.schemas import RetrievalResult, Message

MAX_SNIPPET_CHARS = 600

def _trim(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    t = (text or "").strip().replace("\r", " ")
    if len(t) <= limit:
        return t
    return t[:limit] + "…"

def build_context(results: List[RetrievalResult]) -> str:
    if not results:
        return ""
    lines = ["--- Retrieved Documents ---"]
    for i, r in enumerate(results, 1):
        src = r.metadata.get("source", "unknown")
        score = f" [sim={r.score:.3f}]" if r.score is not None else ""
        lines.append(f"[{i}] ({src}){score}\n{_trim(r.page_content)}")
    return "\n\n".join(lines)

def build_messages(user_question: str, history: List[Message], context: str, system_instructions: str | None = None) -> List[Message]:
    messages: List[Message] = []
    if system_instructions:
        messages.append(Message(role="system", content=system_instructions))
    messages.extend(history[-6:])  # giữ tối đa 6 message gần nhất
    if context:
        messages.append(Message(role="tool", content=context, name="search_docs"))
    messages.append(Message(role="user", content=user_question))
    return messages
