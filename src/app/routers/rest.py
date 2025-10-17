# src/app/routers/rest.py
from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from domain.schemas import UserInput, ResponseOutput
from app.dependencies import get_orchestrator
from rag.orchestrator import Orchestrator
from security.auth import require_role, Identity
from ops.rate_limit import check_rate_limit
from ops.idempotency import get_cached, set_cached

router = APIRouter(prefix="/rest-retrieve", tags=["rest"])

@router.post("/", response_model=ResponseOutput, dependencies=[Depends(require_role("user"))])
async def rest_retrieve(
    body: UserInput,
    request: Request,
    orch: Orchestrator = Depends(get_orchestrator),
    idt: Identity = Depends(require_role("user")),
):
    await check_rate_limit(request, user_id=idt.user_id)

    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = await get_cached(idem_key)
        if cached:
            return ResponseOutput(**cached)

    resp = await orch.answer_rest(body)

    if idem_key:
        await set_cached(idem_key, resp.model_dump())
    return resp
