# src/observability/logging.py
from __future__ import annotations
import logging, sys, json, time
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("_request_id", default="-")

RESERVED = {
    "name","msg","args","levelname","levelno","pathname","filename","module",
    "exc_info","exc_text","stack_info","lineno","funcName","created","msecs",
    "relativeCreated","thread","threadName","processName","process"
}

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": int(time.time() * 1000),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "rid": _request_id.get(),
        }
        for k, v in record.__dict__.items():
            if k not in RESERVED and k not in payload:
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(h)
    logger.propagate = False
    return logger

def set_request_id(rid: str) -> None:
    _request_id.set(rid)
