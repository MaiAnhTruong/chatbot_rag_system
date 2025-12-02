from __future__ import annotations
from fastapi import Depends, HTTPException, status, Request

from app.settings import SETTINGS
from observability.logging import get_logger

log = get_logger("auth")

class Identity:
    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role


# Cache set key cho lookup nhanh; vẫn lấy từ env, rotate/revoke bằng cách chỉnh env rồi restart.
_API_KEYS_SET = set(SETTINGS.API_KEYS)


def assert_auth_config_on_startup() -> None:
    """
    Đảm bảo config auth an toàn trước khi app phục vụ request.
    - ENV=prod + AUTH_MODE="none" -> raise RuntimeError (không cho startup).
    - AUTH_MODE="api_key" nhưng không có API_KEYS -> log error, prod thì raise.
    - Dev (ENV=dev, AUTH_MODE="none") -> cho phép nhưng log cảnh báo to.
    """
    if SETTINGS.ENV == "prod" and SETTINGS.AUTH_MODE == "none":
        log.error(
            "auth.misconfig",
            extra={"env": SETTINGS.ENV, "auth_mode": SETTINGS.AUTH_MODE},
        )
        raise RuntimeError(
            "AUTH_MODE='none' is not allowed in prod. "
            "Set AUTH_MODE='api_key' and configure API_KEYS in environment."
        )

    if SETTINGS.AUTH_MODE == "api_key":
        if not _API_KEYS_SET:
            log.error(
                "auth.api_keys.empty",
                extra={"env": SETTINGS.ENV, "auth_mode": SETTINGS.AUTH_MODE},
            )
            if SETTINGS.ENV == "prod":
                raise RuntimeError(
                    "In prod, AUTH_MODE='api_key' requires at least one API key "
                    "configured via API_KEYS."
                )
        else:
            prefixes = [k[:4] + "..." for k in SETTINGS.API_KEYS]
            log.info(
                "auth.api_keys.loaded",
                extra={
                    "env": SETTINGS.ENV,
                    "count": len(_API_KEYS_SET),
                    "prefixes": prefixes,  # chỉ log prefix, không log full token
                },
            )
    else:
        # Dev mode, cho phép AUTH_MODE="none" nhưng log cảnh báo rõ ràng
        log.warning(
            "auth.dev_mode",
            extra={"env": SETTINGS.ENV, "auth_mode": SETTINGS.AUTH_MODE},
        )


def get_identity(request: Request) -> Identity:
    # Dev / test mode: AUTH_MODE="none"
    if SETTINGS.AUTH_MODE == "none":
        uid = request.headers.get("x-user-id", "anon")
        return Identity(user_id=uid, role=SETTINGS.DEFAULT_ROLE)

    # api_key mode
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        log.warning(
            "auth.missing_token",
            extra={
                "client": request.client.host if request.client else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = auth.split(" ", 1)[1].strip()
    token_prefix = token[:6]

    if token not in _API_KEYS_SET:
        # Log chỉ prefix, tuyệt đối không log full token
        log.warning(
            "auth.invalid_token",
            extra={
                "token_prefix": token_prefix,
                "client": request.client.host if request.client else None,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # user_id: dùng prefix để vào log/metrics, không để lộ toàn bộ token
    user_id = f"key:{token_prefix}"
    return Identity(user_id=user_id, role=SETTINGS.DEFAULT_ROLE)


def require_role(role: str):
    def _dep(idt: Identity = Depends(get_identity)) -> Identity:
        # đơn giản: nếu role yêu cầu là 'admin' thì kiểm tra
        if role == "admin" and idt.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
            )
        return idt

    return _dep
