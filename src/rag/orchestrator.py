# src/rag/orchestrator.py
from __future__ import annotations
from typing import AsyncIterator, Iterable, List, Dict, Any
from domain.schemas import UserInput, ResponseOutput, Message
from prompt.context import build_context, build_messages
from llm.base import LLMClient
from safety.rails import check_input, check_output, check_stream_token
from cache.semantic import pre_cache_get, post_cache_set
from history.store import get as history_get, append as history_append, summarize_if_needed as history_summarize
from observability.tracing import start_span
from observability.logging import get_logger
from app.settings import SETTINGS

log = get_logger("orchestrator")

class Orchestrator:
    def __init__(self, llm: LLMClient, retriever):
        self.llm = llm
        self.retriever = retriever
        self.top_k = SETTINGS.TOP_K

    def _session_ctx(self, ui: UserInput) -> Dict[str, Any]:
        return {"user_id": ui.user_id or "anon", "session_id": ui.session_id or "anon"}

    def _fallback_from_context(self, question: str, results: List) -> str:
        if not SETTINGS.RAG_ENABLED or not results:
            return ("Hiện không thể tạo câu trả lời do LLM gặp sự cố (vd. 429/quota) "
                    "và RAG đang tắt hoặc không có dữ liệu. Vui lòng kiểm tra cấu hình/quota.")
        parts = ["Tóm tắt nhanh (fallback, không dùng LLM):"]
        for i, r in enumerate(results[:3], 1):
            src = r.metadata.get("source", "unknown")
            snippet = (r.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            parts.append(f"- [{i}] ({src}) {snippet}")
        parts.append(f"\nCâu hỏi: {question}")
        return "\n".join(parts)

    # ----------- REST (async) -----------
    async def answer_rest(self, ui: UserInput) -> ResponseOutput:
        with start_span("guardrails.input"):
            gate = check_input(ui.question)
            if not gate.get("allowed", True):
                return ResponseOutput(text="Yêu cầu bị chặn do chứa nội dung nhạy cảm.", citations=[], finish_reason="blocked")
            question = gate.get("transformed_text", ui.question)

        cache_hit = await pre_cache_get(question, self._session_ctx(ui))
        if cache_hit and "text" in cache_hit:
            return ResponseOutput(**cache_hit)

        session_id = ui.session_id or "default"

        with start_span("history.load"):
            history_msgs: List[Message] = await history_get(session_id)
            await history_summarize(session_id)

        results = []
        if SETTINGS.RAG_ENABLED:
            try:
                with start_span("retrieval"):
                    results = self.retriever.retrieve(question, top_k=self.top_k)
            except Exception as e:
                log.error("retrieval.failed", extra={"err": str(e)})
        else:
            log.info("retrieval.skipped", extra={"rag_enabled": False})

        ctx = build_context(results)
        messages = build_messages(question, history_msgs, ctx, system_instructions="Bạn là trợ lý hữu ích.")

        finish_reason = "stop"
        try:
            with start_span("llm.generate"):
                text = await self.llm.agenerate(messages)
        except Exception as e:
            log.error("llm.generate.failed", extra={"err": str(e)})
            text = self._fallback_from_context(question, results)
            finish_reason = "degraded"

        with start_span("guardrails.output"):
            out = check_output(text)
            text = out.get("redacted_text", text)

        resp = ResponseOutput(text=text, citations=[r.metadata for r in results], finish_reason=finish_reason)
        try:
            with start_span("persist"):
                await history_append(session_id, [Message(role="user", content=question), Message(role="assistant", content=text)])
                await post_cache_set(question, self._session_ctx(ui), resp.model_dump())
        except Exception as e:
            log.error("persist.failed", extra={"err": str(e)})
        return resp

    # ----------- SSE (async) -----------
    async def answer_sse_tokens(self, ui: UserInput, resume_from: int = 0) -> AsyncIterator[str]:
        with start_span("guardrails.input"):
            gate = check_input(ui.question)
            if not gate.get("allowed", True):
                yield "[blocked]"
                return
            question = gate.get("transformed_text", ui.question)

        cache_hit = await pre_cache_get(question, self._session_ctx(ui))
        if cache_hit and "frames" in cache_hit:
            frames = cache_hit["frames"]
            start_idx = max(0, resume_from)
            for i, frame in enumerate(frames[start_idx:], start_idx + 1):
                yield frame
            return

        session_id = ui.session_id or "default"

        history_msgs: List[Message] = []
        try:
            with start_span("history.load"):
                history_msgs = await history_get(session_id)
                await history_summarize(session_id)
        except Exception as e:
            log.error("history.load.failed", extra={"err": str(e)})

        results: List = []
        if SETTINGS.RAG_ENABLED:
            try:
                with start_span("retrieval"):
                    results = self.retriever.retrieve(question, top_k=self.top_k)
            except Exception as e:
                log.error("retrieval.failed", extra={"err": str(e)})

        ctx = build_context(results)
        messages = build_messages(question, history_msgs, ctx, system_instructions="Bạn là trợ lý hữu ích.")

        frames: List[str] = []
        full: List[str] = []
        try:
            with start_span("llm.stream"):
                async for tok in self.llm.astream(messages):
                    if check_stream_token(tok).get("blocked", False):
                        continue
                    full.append(tok)
                    frames.append(tok)
                    yield tok
        except Exception as e:
            log.error("llm.stream.failed", extra={"err": str(e)})
            fallback_text = self._fallback_from_context(question, results)
            full = [fallback_text]
            frames.append(fallback_text)
            yield fallback_text

        text = "".join(full)
        try:
            with start_span("guardrails.output"):
                out = check_output(text)
                text = out.get("redacted_text", text)
        except Exception as e:
            log.error("guardrails.output.failed", extra={"err": str(e)})

        try:
            await history_append(session_id, [Message(role="user", content=question), Message(role="assistant", content=text)])
            await post_cache_set(question, self._session_ctx(ui), {"frames": frames, "text": text})
        except Exception as e:
            log.error("persist.failed", extra={"err": str(e)})
