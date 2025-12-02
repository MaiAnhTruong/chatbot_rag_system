from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from domain.schemas import UserInput, ResponseOutput
from app.dependencies import get_orchestrator
from rag.orchestrator import Orchestrator
from security.auth import require_role, Identity
from ops.rate_limit import check_rate_limit
from ops.idempotency import get_cached, set_cached

router = APIRouter(prefix="/rest-retrieve", tags=["rest"])


@router.post(
    "/",
    response_model=ResponseOutput,
    dependencies=[Depends(require_role("user"))],
)
async def rest_retrieve(
    body: UserInput,
    request: Request,
    orch: Orchestrator = Depends(get_orchestrator),
    idt: Identity = Depends(require_role("user")),
):
    # Rate limit theo user_id / IP trước
    await check_rate_limit(request, user_id=idt.user_id)

    # Idempotency cache: nếu đã có kết quả sẵn thì trả luôn, KHÔNG tốn slot LLM
    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = await get_cached(idem_key)
        if cached:
            return ResponseOutput(**cached)

    # Semaphore chung cho các LLM-heavy operations (REST + SSE)
    llm_sem = getattr(request.app.state, "llm_sem", None)

    if llm_sem is None:
        # Fallback: nếu vì lý do gì đó không có semaphore, cứ gọi trực tiếp
        resp = await orch.answer_rest(body)
    else:
        async with llm_sem:
            resp = await orch.answer_rest(body)

    # Lưu idempotency sau khi có kết quả
    if idem_key:
        await set_cached(idem_key, resp.model_dump())
    return resp
