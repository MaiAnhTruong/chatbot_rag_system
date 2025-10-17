# src/app/main.py
from __future__ import annotations
import uuid, asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.settings import SETTINGS
from app.routers import health, rest, sse
from app.bootstrap import build_orchestrator
from observability.logging import set_request_id, get_logger
from observability.metrics import metrics_endpoint, metrics_middleware

log = get_logger("app")

def create_app() -> FastAPI:
    app = FastAPI(title=SETTINGS.APP_NAME, debug=SETTINGS.DEBUG)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.CORS_ALLOW_ORIGINS,
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
            return JSONResponse(status_code=413, content={"error": "Request too large", "rid": rid})

        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    # Exception handlers (JSON hoá lỗi)
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "rid": request.headers.get("x-request-id")})

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": "ValidationError", "details": exc.errors()})

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):
        log.error("unhandled", extra={"err": str(exc)})
        return JSONResponse(status_code=500, content={"error": "InternalServerError"})

    # Routers
    app.include_router(health.router)
    app.include_router(rest.router, prefix=f"/{SETTINGS.API_V1_STR}")
    app.include_router(sse.router,  prefix=f"/{SETTINGS.API_V1_STR}")

    # /metrics
    @app.get("/metrics")
    def _metrics():
        return metrics_endpoint()

    # Shared semaphore cho SSE backpressure
    app.state.stream_sem = asyncio.Semaphore(SETTINGS.MAX_CONCURRENT_STREAMS)

    @app.on_event("startup")
    async def _startup():
        app.state.orchestrator = build_orchestrator()

    return app

app = create_app()
