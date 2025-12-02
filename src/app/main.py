from __future__ import annotations

import uuid, asyncio, time
from contextlib import suppress

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.settings import SETTINGS
from app.routers import health, rest, sse
from app.bootstrap import build_orchestrator
from domain.schemas import Message
from observability.logging import set_request_id, get_logger
from observability.metrics import metrics_endpoint, metrics_middleware

log = get_logger("app")


async def _llm_health_check(app: FastAPI) -> None:
    """
    Deep check nhẹ cho LLM:
    - Gửi 1 prompt cực ngắn.
    - Không parse kết quả; chỉ cần call thành công là đủ.
    """
    dh = getattr(app.state, "deep_health", None)
    if dh is None:
        app.state.deep_health = {
            "llm_ok": False,
            "llm_last_checked": 0.0,
            "vector_ok": None,
            "vector_last_checked": 0.0,
        }
        dh = app.state.deep_health

    orch = getattr(app.state, "orchestrator", None)
    if orch is None or not getattr(orch, "llm", None):
        dh["llm_ok"] = False
        dh["llm_last_checked"] = time.time()
        return

    msgs = [Message(role="user", content="healthcheck")]
    try:
        await orch.llm.agenerate(msgs)
        ok = True
    except Exception as e:
        log.error("llm.health.failed", extra={"err": str(e)})
        ok = False

    dh["llm_ok"] = ok
    dh["llm_last_checked"] = time.time()


async def _vector_health_check(app: FastAPI) -> None:
    """
    Deep check nhẹ cho vector store / retriever:
    - Gọi retriever.retrieve("__healthcheck__", top_k=1) trong executor.
    - Nếu RAG_DISABLED → vector_ok = None.
    """
    dh = getattr(app.state, "deep_health", None)
    if dh is None:
        app.state.deep_health = {
            "llm_ok": False,
            "llm_last_checked": 0.0,
            "vector_ok": None,
            "vector_last_checked": 0.0,
        }
        dh = app.state.deep_health

    orch = getattr(app.state, "orchestrator", None)

    if orch is None or not SETTINGS.RAG_ENABLED:
        dh["vector_ok"] = None
        dh["vector_last_checked"] = time.time()
        return

    retriever = getattr(orch, "retriever", None)
    if retriever is None:
        dh["vector_ok"] = False
        dh["vector_last_checked"] = time.time()
        return

    loop = asyncio.get_running_loop()
    try:
        # Gọi sync retrieve trong thread pool để không block event loop
        await loop.run_in_executor(
            None,
            lambda: retriever.retrieve("__healthcheck__", top_k=1),
        )
        ok = True
    except Exception as e:
        log.error("vector.health.failed", extra={"err": str(e)})
        ok = False

    dh["vector_ok"] = ok
    dh["vector_last_checked"] = time.time()


async def _health_worker(app: FastAPI) -> None:
    """
    Background worker:
    - Định kỳ gọi _llm_health_check và _vector_health_check.
    - Kết quả lưu vào app.state.deep_health để /ready đọc.
    """
    llm_interval = max(5, int(SETTINGS.LLM_HEALTH_CHECK_INTERVAL_SEC))
    vec_interval = max(5, int(SETTINGS.VECTOR_HEALTH_CHECK_INTERVAL_SEC))

    while True:
        try:
            orch = getattr(app.state, "orchestrator", None)
            if orch is None:
                # Chờ orchestrator được build xong
                await asyncio.sleep(1.0)
                continue

            dh = getattr(app.state, "deep_health", None)
            if dh is None:
                app.state.deep_health = {
                    "llm_ok": False,
                    "llm_last_checked": 0.0,
                    "vector_ok": None,
                    "vector_last_checked": 0.0,
                }
                dh = app.state.deep_health

            now = time.time()

            # LLM
            if now - float(dh.get("llm_last_checked", 0.0)) >= llm_interval:
                await _llm_health_check(app)

            # Vector / retriever (chỉ khi RAG bật)
            if SETTINGS.RAG_ENABLED and now - float(
                dh.get("vector_last_checked", 0.0)
            ) >= vec_interval:
                await _vector_health_check(app)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("health.worker.error", extra={"err": str(e)})
            await asyncio.sleep(5.0)

        await asyncio.sleep(1.0)


def create_app() -> FastAPI:
    app = FastAPI(title=SETTINGS.APP_NAME, debug=SETTINGS.DEBUG)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.CORS_ALLOW_ORIGINS
        or (["*"] if SETTINGS.ENV == "dev" else SETTINGS.CORS_ALLOW_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics middleware
    app.middleware("http")(metrics_middleware)

    # Correlation-id + request size guard
    @app.middleware("http")
    async def _common_middlewares(request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        set_request_id(rid)

        # size limit (nếu Content-Length có mặt)
        cl = request.headers.get("content-length")
        if cl and int(cl) > SETTINGS.MAX_REQUEST_SIZE_BYTES:
            return JSONResponse(
                status_code=413, content={"error": "Request too large", "rid": rid}
            )

        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    # Exception handlers (JSON hoá lỗi)
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "rid": request.headers.get("x-request-id")},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "ValidationError", "details": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        log.error("unhandled", extra={"err": str(exc)})
        return JSONResponse(status_code=500, content={"error": "InternalServerError"})

    # Routers
    app.include_router(health.router)
    app.include_router(rest.router, prefix=f"/{SETTINGS.API_V1_STR}")
    app.include_router(sse.router, prefix=f"/{SETTINGS.API_V1_STR}")

    # /metrics
    @app.get("/metrics")
    def _metrics():
        return metrics_endpoint()

    # Shared semaphore cho SSE backpressure (số stream SSE song song)
    app.state.stream_sem = asyncio.Semaphore(SETTINGS.MAX_CONCURRENT_STREAMS)
    # Semaphore chung cho các “LLM-heavy” operations (REST + SSE)
    app.state.llm_sem = asyncio.Semaphore(
        max(1, int(SETTINGS.LLM_MAX_CONCURRENT_REQUESTS))
    )

    # Deep health state cho /ready
    app.state.deep_health = {
        "llm_ok": False,
        "llm_last_checked": 0.0,
        "vector_ok": None,
        "vector_last_checked": 0.0,
    }

    @app.on_event("startup")
    async def _startup():
        app.state.orchestrator = build_orchestrator()
        # Background health worker
        app.state.health_task = asyncio.create_task(_health_worker(app))

    @app.on_event("shutdown")
    async def _shutdown():
        task = getattr(app.state, "health_task", None)
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return app


app = create_app()
