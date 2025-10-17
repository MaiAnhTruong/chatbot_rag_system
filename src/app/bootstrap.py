# src/app/bootstrap.py
from __future__ import annotations
from app.settings import SETTINGS
from observability.logging import get_logger
from rag.orchestrator import Orchestrator
from retriever.noop import NoopRetriever
from retriever.service import RetrieverService

log = get_logger("bootstrap")

def _build_llm():
    # bắt lỗi thiếu package hoặc config ngay khi khởi động
    from openai import OpenAI  # noqa: F401
    from llm.openai_compatible import OpenAICompatLLM
    llm = OpenAICompatLLM()
    log.info("llm.ready", extra={"provider": SETTINGS.LLM_PROVIDER, "base_url": SETTINGS.LLM_BASE_URL, "model": SETTINGS.LLM_MODEL})
    return llm

def _build_retriever():
    if not SETTINGS.RAG_ENABLED:
        log.info("rag.disabled")
        return NoopRetriever()

    try:
        from embeddings.hf_e5 import HFE5Embeddings
        from vector.chroma import ChromaVectorStore
    except Exception as e:
        raise RuntimeError(f"RAG components missing: {e}")

    emb = HFE5Embeddings(SETTINGS.EMBEDDING_MODEL)
    log.info("emb.ready", extra={"provider": SETTINGS.EMBEDDING_PROVIDER, "model": SETTINGS.EMBEDDING_MODEL})

    store = ChromaVectorStore(collection_name=f"{SETTINGS.APP_NAME}-default", embedder=emb)
    log.info("vector.ready", extra={"backend": "chroma", "path": SETTINGS.CHROMA_DIR})

    # Seed chỉ để smoke test; bọc try để không cản startup
    try:
        store.upsert([
            ("RAG kết hợp truy hồi tài liệu với mô hình sinh để tăng độ chính xác.", {"source": "seed:guide", "id": "seed-1"}),
            ("SSE là cơ chế text/event-stream, với các dòng 'event:' và 'data:' kết thúc bằng dòng trống.", {"source": "seed:sse", "id": "seed-2"}),
        ])
        log.info("seed.ok")
    except Exception as e:
        log.error("seed.failed", extra={"err": str(e)})

    return RetrieverService(emb, store)

def build_orchestrator() -> Orchestrator:
    # Cảnh báo nếu OPENAI_API_KEY thiếu khi provider=openai
    if SETTINGS.LLM_PROVIDER == "openai" and not SETTINGS.OPENAI_API_KEY:
        log.warning("openai.key.missing", extra={"note": "Set OPENAI_API_KEY in .env for production"})
    llm = _build_llm()
    retriever = _build_retriever()
    return Orchestrator(llm=llm, retriever=retriever)
