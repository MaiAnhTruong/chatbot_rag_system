# src/security/auth.py
from __future__ import annotations
from fastapi import Depends, HTTPException, status, Request
from app.settings import SETTINGS

class Identity:
    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role

def get_identity(request: Request) -> Identity:
    if SETTINGS.AUTH_MODE == "none":
        # dev mode
        uid = request.headers.get("x-user-id", "anon")
        return Identity(user_id=uid, role=SETTINGS.DEFAULT_ROLE)

    # api_key mode
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token not in set(SETTINGS.API_KEYS):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    # map key -> user_id (ở đây dùng chính token để đơn giản) / có thể map khác tùy nhu cầu
    return Identity(user_id=token[:6], role=SETTINGS.DEFAULT_ROLE)

def require_role(role: str):
    def _dep(idt: Identity = Depends(get_identity)) -> Identity:
        # đơn giản: nếu role yêu cầu là 'admin' thì kiểm tra
        if role == "admin" and idt.role != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return idt
    return _dep
