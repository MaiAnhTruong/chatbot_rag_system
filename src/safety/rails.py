from __future__ import annotations
import re
from typing import Dict, Any

from app.settings import SETTINGS

# Các keyword nhạy cảm (chỉ dùng cho output redaction, KHÔNG dùng để block input thẳng tay)
BLOCKLIST = {"password", "api_key", "secret", "private_key"}

PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Regex phone được siết lại:
# - Yêu cầu chuỗi có 8–15 chữ số
# - Cho phép +country code, khoảng trắng, -, (, ), .
# - Chỉ dùng khi PRIVACY_STRICT_MODE=True để tránh false positive trong mode thường.
PII_PHONE = re.compile(
    r"""
    (?<!\d)                             # không dính liền với số trước đó
    (?:\+?\d{1,3}[\s\-().]*)?           # optional country code
    (?:\d[\s\-().]*){8,15}              # 8-15 digits tổng, với ngăn cách
    (?!\d)                              # không dính liền số sau
    """,
    re.VERBOSE,
)

# Pattern phát hiện user ĐANG GỬI bí mật (không chỉ hỏi khái niệm)
SENSITIVE_SHARE_REGEX = re.compile(
    r"""
    (
      # "my password", "my api key", "my secret", "my private key" ...
      (my\s+(password|api[_-]?key|secret|private\s+key)) |
      (here\s+is\s+my\s+(password|api[_-]?key|secret|private\s+key)) |

      # "password=xxx", "password: xxx"
      (password\s*[:=]\s*\S{4,}) |
      (api[_-]?key\s*[:=]\s*\S{4,}) |
      (secret\s*[:=]\s*\S{4,}) |
      (private\s+key\s*[:=]\s*\S{4,})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _strict_mode() -> bool:
    return bool(getattr(SETTINGS, "PRIVACY_STRICT_MODE", False))


def check_input(text: str) -> Dict[str, Any]:
    """
    Guard input:

    - KHÔNG block chỉ vì chứa từ "password", "api_key", v.v.
      -> Các câu hỏi học thuật kiểu "Giải thích password là gì" vẫn được allow.

    - CHỈ block khi:
      + Có dấu hiệu user đang gửi bí mật thật, ví dụ "my password is ...",
        "password=1234", "here is my api_key: ...".

    - Email / phone: không block input, chỉ xử lý ở output (redact).
    """
    if SENSITIVE_SHARE_REGEX.search(text):
        return {"allowed": False, "reason": "sensitive-sharing"}

    return {"allowed": True, "transformed_text": text}


def check_stream_token(token: str) -> Dict[str, Any]:
    """
    Guard token khi stream SSE.

    - Mặc định (PRIVACY_STRICT_MODE=False):
        -> luôn allow, giống behaviour cũ.
    - Strict mode:
        -> block token nếu:
           + chứa email
           + hoặc match pattern phone
           + hoặc có pattern chia sẻ secret (hiếm khi trọn vẹn trong 1 token,
             nhưng vẫn check được phần nào).
    Lưu ý: vì chỉ nhìn từng token riêng lẻ nên không hoàn hảo, nhưng giúp
    giảm rủi ro SSE leak PII trong mode strict.
    """
    if not _strict_mode():
        return {"blocked": False}

    if PII_EMAIL.search(token) or PII_PHONE.search(token):
        return {"blocked": True, "reason": "pii"}

    if SENSITIVE_SHARE_REGEX.search(token):
        return {"blocked": True, "reason": "secret"}

    return {"blocked": False}


def check_output(text: str) -> Dict[str, Any]:
    """
    Guard output cuối cùng:

    - Luôn redact email.
    - Phone:
        + Chỉ redact khi PRIVACY_STRICT_MODE=True để tránh false positive
          trong mode thường.
    - Redact các từ trong BLOCKLIST (đơn giản: thay bằng [REDACTED]).
      => Có thể tinh chỉnh thêm sau nếu muốn giữ chữ "password" trong
         ngữ cảnh học thuật mà không che.
    """
    redacted = PII_EMAIL.sub("[REDACTED_EMAIL]", text)

    if _strict_mode():
        redacted = PII_PHONE.sub("[REDACTED_PHONE]", redacted)

    for b in BLOCKLIST:
        redacted = redacted.replace(b, "[REDACTED]")

    return {"allowed": True, "redacted_text": redacted}
