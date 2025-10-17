# src/safety/rails.py
from __future__ import annotations
import re
from typing import Dict, Any

BLOCKLIST = {"password", "api_key", "secret", "private_key"}
PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE = re.compile(r"\b(\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")

def check_input(text: str) -> Dict[str, Any]:
    lowered = text.lower()
    if any(b in lowered for b in BLOCKLIST) or PII_EMAIL.search(text) or PII_PHONE.search(text):
        return {"allowed": False, "reason": "sensitive-request"}
    return {"allowed": True, "transformed_text": text}

def check_stream_token(token: str) -> Dict[str, Any]:
    return {"blocked": False}

def check_output(text: str) -> Dict[str, Any]:
    redacted = PII_EMAIL.sub("[REDACTED_EMAIL]", text)
    redacted = PII_PHONE.sub("[REDACTED_PHONE]", redacted)
    for b in BLOCKLIST:
        redacted = redacted.replace(b, "[REDACTED]")
    return {"allowed": True, "redacted_text": redacted}
