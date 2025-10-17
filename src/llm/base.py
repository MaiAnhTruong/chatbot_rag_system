# src/llm/base.py
from __future__ import annotations
import asyncio, threading
from typing import Iterable, AsyncIterator, List, Dict, Any
from domain.schemas import Message

class LLMClient:
    # ---- sync (giữ để tương thích) ----
    def generate(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> str:
        raise NotImplementedError

    def stream(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> Iterable[str]:
        raise NotImplementedError

    # ---- async wrappers mặc định (chạy sync trong thread) ----
    async def agenerate(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, messages, tools)

    async def astream(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=128)
        SENTINEL = object()

        def _producer():
            try:
                for tok in self.stream(messages, tools):
                    fut = asyncio.run_coroutine_threadsafe(q.put(tok), loop)
                    fut.result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(SENTINEL), loop).result()

        threading.Thread(target=_producer, daemon=True).start()
        while True:
            item = await q.get()
            if item is SENTINEL:
                break
            yield str(item)
