# src/observability/tracing.py
from __future__ import annotations
import contextlib, time
from typing import Iterator
from observability.logging import get_logger

log = get_logger("trace")

@contextlib.contextmanager
def start_span(name: str, **fields) -> Iterator[None]:
    t0 = time.perf_counter()
    log.info("span.start", extra={"span": name, **fields})
    try:
        yield
        dt = (time.perf_counter() - t0) * 1000
        log.info("span.end", extra={"span": name, "ms": round(dt, 2)})
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        log.error("span.err", extra={"span": name, "ms": round(dt, 2), "err": str(e)})
        raise
