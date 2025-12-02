from __future__ import annotations

import hashlib
from typing import AsyncIterator, Iterable, List, Dict, Any

from domain.schemas import UserInput, ResponseOutput, Message
from prompt.context import build_context, build_messages
from llm.base import LLMClient
from safety.rails import check_input, check_output, check_stream_token
from cache.semantic import pre_cache_get, post_cache_set
from history.store import (
    get as history_get,
    append as history_append,
    summarize_if_needed as history_summarize,
)
from observability.tracing import start_span
from observability.logging import get_logger
from observability.metrics import (
    LLM_FALLBACK_DEGRADED,
    SEMANTIC_CACHE_HITS,
    SEMANTIC_CACHE_MISSES,
    RAG_REQUESTS,
)
from app.settings import SETTINGS

log = get_logger("orchestrator")


def _hash_user_id(user_id: str | None) -> str:
    if not user_id:
        return "anon"
    try:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "anon"


class Orchestrator:
    def __init__(self, llm: LLMClient, retriever):
        self.llm = llm
        self.retriever = retriever
        self.top_k = SETTINGS.TOP_K

    def _session_ctx(self, ui: UserInput) -> Dict[str, Any]:
        return {"user_id": ui.user_id or "anon", "session_id": ui.session_id or "anon"}

    def _fallback_from_context(self, question: str, results: List) -> str:
        if not SETTINGS.RAG_ENABLED or not results:
            return (
                "Hiện không thể tạo câu trả lời do LLM gặp sự cố (vd. 429/quota) "
                "và RAG đang tắt hoặc không có dữ liệu. Vui lòng kiểm tra cấu hình/quota."
            )
        parts = ["Tóm tắt nhanh (fallback, không dùng LLM):"]
        for i, r in enumerate(results[:3], 1):
            src = r.metadata.get("source", "unknown")
            snippet = (r.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            parts.append(f"- [{i}] ({src}) {snippet}")
        parts.append(f"\nCâu hỏi: {question}")
        return "\n".join(parts)

    def _log_chat_completed(
        self,
        *,
        kind: str,
        ui: UserInput,
        question: str,
        answer_text: str,
        finish_reason: str,
        cache_hit: bool,
        rag_used: bool,
        citations_count: int,
    ) -> None:
        """
        Log business-level event cho cả REST & SSE, không log nội dung raw,
        chỉ log độ dài & trạng thái.
        """
        user_hash = _hash_user_id(ui.user_id)
        session_id = ui.session_id or "default"
        try:
            log.info(
                "chat.completed",
                extra={
                    "kind": kind,
                    "user_hash": user_hash,
                    "session_id": session_id,
                    "rag_enabled": bool(rag_used),
                    "cache_hit": bool(cache_hit),
                    "finish_reason": finish_reason,
                    "question_len": len(question or ""),
                    "answer_len": len(answer_text or ""),
                    "citations": citations_count,
                },
            )
        except Exception:
            # Không để lỗi logging phá flow chính
            pass

    # ----------- REST (async) -----------
    async def answer_rest(self, ui: UserInput) -> ResponseOutput:
        with start_span("guardrails.input"):
            gate = check_input(ui.question)
            if not gate.get("allowed", True):
                resp = ResponseOutput(
                    text="Yêu cầu bị chặn do chứa nội dung nhạy cảm.",
                    citations=[],
                    finish_reason="blocked",
                )
                # Không gọi RAG / cache trong trường hợp bị block
                self._log_chat_completed(
                    kind="rest",
                    ui=ui,
                    question=ui.question,
                    answer_text=resp.text,
                    finish_reason=resp.finish_reason,
                    cache_hit=False,
                    rag_used=False,
                    citations_count=0,
                )
                return resp
            question = gate.get("transformed_text", ui.question)

        session_ctx = self._session_ctx(ui)
        session_id = ui.session_id or "default"
        cache_hit = False

        # Semantic cache (REST)
        cache_data: Dict[str, Any] | None = None
        if SETTINGS.CACHE_ENABLED:
            cache_data = await pre_cache_get(question, session_ctx)
            if cache_data and "text" in cache_data:
                cache_hit = True
                try:
                    SEMANTIC_CACHE_HITS.labels(kind="rest").inc()
                except Exception:
                    pass
                resp = ResponseOutput(**cache_data)
                # Metrics: RAG usage (config level)
                try:
                    RAG_REQUESTS.labels(
                        kind="rest",
                        used=str(bool(SETTINGS.RAG_ENABLED)).lower(),
                    ).inc()
                except Exception:
                    pass
                self._log_chat_completed(
                    kind="rest",
                    ui=ui,
                    question=question,
                    answer_text=resp.text,
                    finish_reason=resp.finish_reason,
                    cache_hit=True,
                    rag_used=SETTINGS.RAG_ENABLED,
                    citations_count=len(resp.citations or []),
                )
                return resp
            else:
                try:
                    SEMANTIC_CACHE_MISSES.labels(kind="rest").inc()
                except Exception:
                    pass

        with start_span("history.load"):
            history_msgs: List[Message] = await history_get(session_id)
            await history_summarize(session_id)

        results: List[Any] = []
        if SETTINGS.RAG_ENABLED:
            try:
                with start_span("retrieval"):
                    results = self.retriever.retrieve(question, top_k=self.top_k)
            except Exception as e:
                log.error("retrieval.failed", extra={"err": str(e)})
        else:
            log.info("retrieval.skipped", extra={"rag_enabled": False})

        ctx = build_context(results)
        messages = build_messages(
            question,
            history_msgs,
            ctx,
            system_instructions="Bạn là trợ lý hữu ích.",
        )

        finish_reason = "stop"
        try:
            with start_span("llm.generate"):
                text = await self.llm.agenerate(messages)
        except Exception as e:
            log.error("llm.generate.failed", extra={"err": str(e)})
            text = self._fallback_from_context(question, results)
            finish_reason = "degraded"
            # metric: fallback degraded (REST)
            try:
                LLM_FALLBACK_DEGRADED.labels(kind="rest").inc()
            except Exception:
                pass

        with start_span("guardrails.output"):
            out = check_output(text)
            text = out.get("redacted_text", text)

        resp = ResponseOutput(
            text=text,
            citations=[r.metadata for r in results],
            finish_reason=finish_reason,
        )
        try:
            with start_span("persist"):
                await history_append(
                    session_id,
                    [
                        Message(role="user", content=question),
                        Message(role="assistant", content=text),
                    ],
                )
                await post_cache_set(
                    question,
                    session_ctx,
                    resp.model_dump(),
                )
        except Exception as e:
            log.error("persist.failed", extra={"err": str(e)})

        # Metrics & logging sau khi xử lý xong
        try:
            RAG_REQUESTS.labels(
                kind="rest",
                used=str(bool(SETTINGS.RAG_ENABLED)).lower(),
            ).inc()
        except Exception:
            pass

        self._log_chat_completed(
            kind="rest",
            ui=ui,
            question=question,
            answer_text=resp.text,
            finish_reason=resp.finish_reason,
            cache_hit=cache_hit,
            rag_used=SETTINGS.RAG_ENABLED,
            citations_count=len(resp.citations or []),
        )

        return resp

    # ----------- SSE (async) -----------
    async def answer_sse_tokens(
        self,
        ui: UserInput,
        resume_from: int = 0,
    ) -> AsyncIterator[str]:
        with start_span("guardrails.input"):
            gate = check_input(ui.question)
            if not gate.get("allowed", True):
                # Không stream đầy đủ trong trường hợp bị block
                yield "[blocked]"
                return
            question = gate.get("transformed_text", ui.question)

        session_ctx = self._session_ctx(ui)
        session_id = ui.session_id or "default"
        cache_hit = False
        finish_reason = "stop"

        # Pre-cache: hỗ trợ resume từ semantic cache (frames)
        cache_data = await pre_cache_get(question, session_ctx) if SETTINGS.CACHE_ENABLED else None
        if SETTINGS.CACHE_ENABLED:
            if cache_data and "frames" in cache_data:
                cache_hit = True
                try:
                    SEMANTIC_CACHE_HITS.labels(kind="sse").inc()
                except Exception:
                    pass

                frames = cache_data["frames"]
                cached_text = cache_data.get("text", "")
                text_for_log = (
                    cached_text
                    if isinstance(cached_text, str)
                    else "".join(frames)
                )

                overlap = max(
                    0, int(getattr(SETTINGS, "SSE_RESUME_OVERLAP_TOKENS", 0))
                )
                if resume_from > 0:
                    start_idx = max(0, resume_from - overlap)
                else:
                    start_idx = 0

                for _i, frame in enumerate(frames[start_idx:], start_idx + 1):
                    yield frame

                # Metrics & logging cho SSE cache hit
                try:
                    RAG_REQUESTS.labels(
                        kind="sse",
                        used=str(bool(SETTINGS.RAG_ENABLED)).lower(),
                    ).inc()
                except Exception:
                    pass

                self._log_chat_completed(
                    kind="sse",
                    ui=ui,
                    question=question,
                    answer_text=text_for_log,
                    finish_reason="cache",
                    cache_hit=True,
                    rag_used=SETTINGS.RAG_ENABLED,
                    citations_count=0,
                )
                return
            else:
                try:
                    SEMANTIC_CACHE_MISSES.labels(kind="sse").inc()
                except Exception:
                    pass

        history_msgs: List[Message] = []
        try:
            with start_span("history.load"):
                history_msgs = await history_get(session_id)
                await history_summarize(session_id)
        except Exception as e:
            log.error("history.load.failed", extra={"err": str(e)})

        results: List[Any] = []
        if SETTINGS.RAG_ENABLED:
            try:
                with start_span("retrieval"):
                    results = self.retriever.retrieve(question, top_k=self.top_k)
            except Exception as e:
                log.error("retrieval.failed", extra={"err": str(e)})

        ctx = build_context(results)
        messages = build_messages(
            question,
            history_msgs,
            ctx,
            system_instructions="Bạn là trợ lý hữu ích.",
        )

        frames: List[str] = []
        full: List[str] = []

        # Incremental cache config
        flush_every = max(
            0, int(getattr(SETTINGS, "SSE_CACHE_FLUSH_EVERY_N_TOKENS", 0))
        )
        tokens_since_flush = 0

        try:
            with start_span("llm.stream"):
                async for tok in self.llm.astream(messages):
                    if check_stream_token(tok).get("blocked", False):
                        continue
                    full.append(tok)
                    frames.append(tok)
                    tokens_since_flush += 1
                    yield tok

                    # Incremental cache: lưu frames vào semantic cache mỗi N token
                    if (
                        SETTINGS.CACHE_ENABLED
                        and flush_every > 0
                        and tokens_since_flush >= flush_every
                    ):
                        tokens_since_flush = 0
                        try:
                            await post_cache_set(
                                question,
                                session_ctx,
                                {"frames": list(frames)},
                            )
                        except Exception as e:
                            log.error(
                                "sse.cache.flush_failed",
                                extra={"err": str(e), "session_id": session_id},
                            )
        except Exception as e:
            log.error("llm.stream.failed", extra={"err": str(e)})
            fallback_text = self._fallback_from_context(question, results)
            full = [fallback_text]
            frames.append(fallback_text)
            yield fallback_text
            finish_reason = "degraded"
            # metric: fallback degraded (SSE)
            try:
                LLM_FALLBACK_DEGRADED.labels(kind="sse").inc()
            except Exception:
                pass

        text = "".join(full)
        try:
            with start_span("guardrails.output"):
                out = check_output(text)
                text = out.get("redacted_text", text)
        except Exception as e:
            log.error("guardrails.output.failed", extra={"err": str(e)})

        # Lưu history + semantic cache (full)
        try:
            await history_append(
                session_id,
                [
                    Message(role="user", content=question),
                    Message(role="assistant", content=text),
                ],
            )
            await post_cache_set(
                question,
                session_ctx,
                {"frames": frames, "text": text},
            )
        except Exception as e:
            log.error("persist.failed", extra={"err": str(e)})

        # Metrics & logging cho SSE
        try:
            RAG_REQUESTS.labels(
                kind="sse",
                used=str(bool(SETTINGS.RAG_ENABLED)).lower(),
            ).inc()
        except Exception:
            pass

        self._log_chat_completed(
            kind="sse",
            ui=ui,
            question=question,
            answer_text=text,
            finish_reason=finish_reason,
            cache_hit=cache_hit,
            rag_used=SETTINGS.RAG_ENABLED,
            citations_count=len(results),
        )
