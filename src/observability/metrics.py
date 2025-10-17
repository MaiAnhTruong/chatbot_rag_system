# src/observability/metrics.py
from __future__ import annotations
import time
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

HTTP_REQS = Counter("http_requests_total", "HTTP requests", ["method", "path", "code"])
HTTP_LAT = Histogram("http_request_latency_seconds", "HTTP request latency", ["method", "path"])

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
