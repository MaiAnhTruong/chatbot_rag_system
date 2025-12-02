from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# HTTP metrics
HTTP_REQS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "code"],
)
HTTP_LAT = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
)

# LLM metrics
LLM_LAT = Histogram(
    "llm_latency_seconds",
    "LLM request latency (including retries)",
    ["provider", "model", "mode"],  # mode: sync_generate / sync_stream / async_generate / async_stream
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM request errors",
    ["provider", "model", "mode", "code"],  # code: HTTP status code hoặc 'unknown'
)

LLM_FALLBACK_DEGRADED = Counter(
    "llm_fallback_degraded_total",
    "Number of times degraded fallback answer was used",
    ["kind"],  # kind: 'rest' / 'sse' / ...
)

LLM_CIRCUIT_OPEN = Counter(
    "llm_circuit_open_total",
    "Number of times the LLM circuit breaker opened",
    ["provider", "model"],
)

# RAG usage metrics
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "RAG retrieval usage per request",
    ["kind", "used"],  # kind: rest/sse, used: "true"/"false"
)

# Semantic cache metrics
SEMANTIC_CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Semantic cache hits",
    ["kind"],  # kind: rest/sse
)

SEMANTIC_CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Semantic cache misses",
    ["kind"],  # kind: rest/sse
)

# SSE active connections
SSE_ACTIVE_CONNECTIONS = Gauge(
    "sse_active_connections",
    "Number of active SSE connections",
)


def metrics_endpoint():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


async def metrics_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - start
    path = request.url.path
    HTTP_REQS.labels(request.method, path, str(response.status_code)).inc()
    HTTP_LAT.labels(request.method, path).observe(dt)
    return response
