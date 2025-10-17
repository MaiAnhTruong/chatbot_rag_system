# src/llm/openai_compatible.py
from __future__ import annotations
import time, asyncio
from typing import Iterable, AsyncIterator, List, Dict, Any, Optional
from domain.schemas import Message
from llm.base import LLMClient
from app.settings import SETTINGS

TRANSIENT_CODES = {408, 409, 429, 500, 502, 503, 504}

def _to_openai_messages(msgs: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        role = "system" if m.role == "tool" else m.role
        out.append({"role": role, "content": m.content})
    return out

class OpenAICompatLLM(LLMClient):
    def __init__(self) -> None:
        from openai import OpenAI
        self.model = SETTINGS.LLM_MODEL
        self.timeout = SETTINGS.LLM_REQUEST_TIMEOUT_SEC
        self.max_retries = max(0, SETTINGS.LLM_MAX_RETRIES)
        self.base_delay = SETTINGS.LLM_RETRY_BASE_DELAY_SEC

        self._sync = OpenAI(
            base_url=SETTINGS.LLM_BASE_URL,
            api_key=SETTINGS.OPENAI_API_KEY or "lm-studio",
            timeout=self.timeout,
        )

        try:
            from openai import AsyncOpenAI  # type: ignore
            self._async: Optional[object] = AsyncOpenAI(
                base_url=SETTINGS.LLM_BASE_URL,
                api_key=SETTINGS.OPENAI_API_KEY or "lm-studio",
                timeout=self.timeout,
            )
        except Exception:
            self._async = None

    # ---------------- SYNC ----------------
    def _retry_loop(self, fn):
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:
                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (status is not None and status not in TRANSIENT_CODES):
                    raise
                time.sleep(self.base_delay * (2 ** attempt))
                attempt += 1

    def generate(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> str:
        payload = {"model": self.model, "messages": _to_openai_messages(messages)}
        def _call():
            resp = self._sync.chat.completions.create(**payload)
            return resp.choices[0].message.content or ""
        return self._retry_loop(_call)

    def stream(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> Iterable[str]:
        payload = {"model": self.model, "messages": _to_openai_messages(messages), "stream": True}
        def _call():
            return self._sync.chat.completions.stream(**payload)
        stream = self._retry_loop(_call)
        with stream as s:
            for ev in s:
                if ev.type == "chat.completion.chunk":
                    delta = ev.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content

    # ---------------- ASYNC ----------------
    async def agenerate(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> str:
        if self._async is None:
            return await super().agenerate(messages, tools)

        attempt = 0
        while True:
            try:
                resp = await self._async.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=_to_openai_messages(messages),
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (status is not None and status not in TRANSIENT_CODES):
                    raise
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                attempt += 1

    async def astream(self, messages: List[Message], tools: List[Dict[str, Any]] | None = None) -> AsyncIterator[str]:
        # Nếu không có AsyncOpenAI → dùng cầu nối (thread→async) của lớp cha
        if self._async is None:
            async for x in super().astream(messages, tools):
                yield x
            return

        attempt = 0
        while True:
            try:
                yielded = False

                # --- CÁCH 1: create(..., stream=True) → async iterator ---
                resp = await self._async.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=_to_openai_messages(messages),
                    stream=True,
                )
                try:
                    async for chunk in resp:  # type: ignore
                        try:
                            delta = chunk.choices[0].delta  # type: ignore
                            content = getattr(delta, "content", None)
                            if content:
                                yielded = True
                                yield content
                        except Exception:
                            # Nếu shape khác lạ, bỏ qua chunk này
                            continue
                except TypeError:
                    # --- CÁCH 2: context manager .stream(...) ---
                    stream = self._async.chat.completions.stream(  # type: ignore[attr-defined]
                        model=self.model,
                        messages=_to_openai_messages(messages),
                    )
                    async with stream as s:  # type: ignore
                        async for ev in s:
                            if getattr(ev, "type", "") == "chat.completion.chunk":
                                delta = ev.choices[0].delta
                                content = getattr(delta, "content", None)
                                if content:
                                    yielded = True
                                    yield content

                # Nếu KHÔNG có token nào → fallback non-stream để đảm bảo UI có nội dung
                if not yielded:
                    text = await self.agenerate(messages)
                    if text:
                        yield text
                break  # kết thúc thành công

            except Exception as e:
                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (status is not None and status not in TRANSIENT_CODES):
                    raise
                await asyncio.sleep(self.base_delay * (2 ** attempt))
                attempt += 1
