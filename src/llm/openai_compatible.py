from __future__ import annotations
import time, asyncio, threading
from typing import Iterable, AsyncIterator, List, Dict, Any, Optional

from domain.schemas import Message
from llm.base import LLMClient
from app.settings import SETTINGS
from observability.metrics import LLM_LAT, LLM_ERRORS, LLM_CIRCUIT_OPEN
from observability.logging import get_logger

log = get_logger("llm")

TRANSIENT_CODES = {408, 409, 429, 500, 502, 503, 504}


def _to_openai_messages(msgs: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        role = "system" if m.role == "tool" else m.role
        out.append({"role": role, "content": m.content})
    return out


# ---- Circuit breaker state (process-wide) ----
_cb_lock = threading.Lock()
_cb_state: Dict[str, float | int] = {
    "open_until": 0.0,
    "fail_count": 0,
}


def _circuit_is_open(mode: str) -> bool:
    """Trả về True nếu circuit đang mở và chưa hết thời gian cooldown."""
    if not SETTINGS.LLM_CB_ENABLED:
        return False
    now = time.time()
    with _cb_lock:
        open_until = float(_cb_state.get("open_until", 0.0))
        if open_until > now:
            # Circuit đang mở
            log.warning(
                "llm.circuit.block",
                extra={"mode": mode, "open_until": int(open_until)},
            )
            return True
        if open_until and open_until <= now:
            # Hết thời gian open -> reset về closed
            _cb_state["open_until"] = 0.0
            _cb_state["fail_count"] = 0
    return False


def _circuit_on_success() -> None:
    """Reset counters khi có call thành công."""
    if not SETTINGS.LLM_CB_ENABLED:
        return
    with _cb_lock:
        _cb_state["fail_count"] = 0
        _cb_state["open_until"] = 0.0


def _circuit_on_error(e: Exception, provider: str, model: str, mode: str) -> None:
    """
    Tăng fail_count khi gặp lỗi transient. Nếu vượt ngưỡng -> mở circuit
    trong LLM_CB_OPEN_SEC.
    """
    if not SETTINGS.LLM_CB_ENABLED:
        return

    status = getattr(e, "status", None) or getattr(e, "status_code", None)
    if status is not None and status not in TRANSIENT_CODES:
        # Không coi các lỗi non-transient là tín hiệu mở circuit.
        return

    now = time.time()
    with _cb_lock:
        fail_count = int(_cb_state.get("fail_count", 0)) + 1
        _cb_state["fail_count"] = fail_count
        if fail_count >= SETTINGS.LLM_CB_FAIL_THRESHOLD:
            open_until = now + SETTINGS.LLM_CB_OPEN_SEC
            _cb_state["open_until"] = open_until
            # Metrics + log
            try:
                LLM_CIRCUIT_OPEN.labels(provider, model).inc()
            except Exception:
                pass
            log.error(
                "llm.circuit.open",
                extra={
                    "provider": provider,
                    "model": model,
                    "mode": mode,
                    "fail_count": fail_count,
                    "open_sec": SETTINGS.LLM_CB_OPEN_SEC,
                },
            )


class OpenAICompatLLM(LLMClient):
    def __init__(self) -> None:
        from openai import OpenAI

        self.model = SETTINGS.LLM_MODEL
        self.timeout = SETTINGS.LLM_REQUEST_TIMEOUT_SEC
        self.max_retries = max(0, SETTINGS.LLM_MAX_RETRIES)
        self.base_delay = SETTINGS.LLM_RETRY_BASE_DELAY_SEC

        # cost & behaviour
        self.max_tokens = SETTINGS.LLM_MAX_TOKENS
        self.temperature = SETTINGS.LLM_TEMPERATURE
        self.top_p = SETTINGS.LLM_TOP_P
        self.provider = SETTINGS.LLM_PROVIDER

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

    # ----- metrics helpers -----
    def _observe_latency(self, mode: str, start: float) -> None:
        dt = time.perf_counter() - start
        try:
            LLM_LAT.labels(self.provider, self.model, mode).observe(dt)
        except Exception:
            # không để lỗi metrics làm hỏng request LLM
            pass

    def _record_error(self, mode: str, e: Exception) -> None:
        code = getattr(e, "status", None) or getattr(e, "status_code", None) or "unknown"
        try:
            LLM_ERRORS.labels(self.provider, self.model, mode, str(code)).inc()
        except Exception:
            pass

    # ---------------- SYNC ----------------
    def _retry_loop(self, fn, mode: str):
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:
                self._record_error(mode, e)
                _circuit_on_error(e, self.provider, self.model, mode)

                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (
                    status is not None and status not in TRANSIENT_CODES
                ):
                    raise
                time.sleep(self.base_delay * (2**attempt))
                attempt += 1

    def generate(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
    ) -> str:
        mode = "sync_generate"
        if _circuit_is_open(mode):
            # Circuit đang open -> không gọi LLM, ném lỗi nhanh để tầng trên fallback.
            raise RuntimeError("llm_circuit_open")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        try:
            def _call():
                return self._sync.chat.completions.create(**payload)

            resp = self._retry_loop(_call, mode)
            self._observe_latency(mode, start)
            _circuit_on_success()
            return resp.choices[0].message.content or ""
        except Exception:
            # lỗi sẽ đã được ghi metrics trong _retry_loop
            raise

    def stream(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
    ) -> Iterable[str]:
        mode = "sync_stream"
        if _circuit_is_open(mode):
            raise RuntimeError("llm_circuit_open")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        try:
            def _call():
                return self._sync.chat.completions.stream(**payload)

            stream = self._retry_loop(_call, mode)
        except Exception:
            # lỗi đã được record trong _retry_loop
            raise

        with stream as s:
            try:
                for ev in s:
                    if ev.type == "chat.completion.chunk":
                        delta = ev.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            yield content
            finally:
                self._observe_latency(mode, start)
                _circuit_on_success()

    # ---------------- ASYNC ----------------
    async def agenerate(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
    ) -> str:
        if self._async is None:
            # fallback sang wrapper sync→async của lớp cha (không có metrics chi tiết)
            return await super().agenerate(messages, tools)

        mode = "async_generate"
        if _circuit_is_open(mode):
            raise RuntimeError("llm_circuit_open")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            payload["tools"] = tools

        attempt = 0
        start = time.perf_counter()
        while True:
            try:
                resp = await self._async.chat.completions.create(  # type: ignore[attr-defined]
                    **payload
                )
                self._observe_latency(mode, start)
                _circuit_on_success()
                return resp.choices[0].message.content or ""
            except Exception as e:
                self._record_error(mode, e)
                _circuit_on_error(e, self.provider, self.model, mode)

                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (
                    status is not None and status not in TRANSIENT_CODES
                ):
                    raise
                await asyncio.sleep(self.base_delay * (2**attempt))
                attempt += 1

    async def astream(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        # Nếu không có AsyncOpenAI → dùng cầu nối (thread→async) của lớp cha
        if self._async is None:
            async for x in super().astream(messages, tools):
                yield x
            return

        mode = "async_stream"
        if _circuit_is_open(mode):
            raise RuntimeError("llm_circuit_open")

        base_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(messages),
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if tools:
            base_payload["tools"] = tools

        attempt = 0
        start = time.perf_counter()
        while True:
            try:
                yielded = False

                # --- CÁCH 1: create(..., stream=True) → async iterator ---
                resp = await self._async.chat.completions.create(  # type: ignore[attr-defined]
                    **base_payload
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
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        tools=tools,
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
                    text = await self.agenerate(messages, tools)
                    if text:
                        yield text

                self._observe_latency(mode, start)
                _circuit_on_success()
                break  # kết thúc thành công

            except Exception as e:
                self._record_error(mode, e)
                _circuit_on_error(e, self.provider, self.model, mode)

                status = getattr(e, "status", None)
                if attempt >= self.max_retries or (
                    status is not None and status not in TRANSIENT_CODES
                ):
                    raise
                await asyncio.sleep(self.base_delay * (2**attempt))
                attempt += 1
