from __future__ import annotations

import collections.abc

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response as FastAPIResponse, StreamingResponse

from ..deps import get_call_service
from ..services.call import CallService

router = APIRouter(tags=["call"])


@router.post("")
async def execute_call(
    request: Request,
    service: CallService = Depends(get_call_service),
):
    body = await request.body()
    accept = request.headers.get("accept", "")
    result_data, content_type, headers = await service.execute_call(body, accept)

    if isinstance(result_data, collections.abc.Iterator):
        return StreamingResponse(
            result_data,
            media_type=content_type,
            headers=headers,
        )

    return FastAPIResponse(
        content=result_data,
        media_type=content_type,
        headers=headers,
    )


@router.post("/stream")
async def execute_call_stream(
    request: Request,
    service: CallService = Depends(get_call_service),
):
    """Always-streaming variant. Returns Arrow IPC stream for tabular,
    pickle bytes for scalars. Preferred for keep-alive connections."""
    body = await request.body()
    result_data, content_type, headers = await service.execute_call(
        body, "application/vnd.apache.arrow.stream"
    )

    if isinstance(result_data, collections.abc.Iterator):
        return StreamingResponse(
            result_data,
            media_type=content_type,
            headers=headers,
        )

    return FastAPIResponse(
        content=result_data,
        media_type=content_type,
        headers=headers,
    )


@router.get("/registry")
async def list_registry(
    service: CallService = Depends(get_call_service),
):
    return JSONResponse(content=service.get_registry())
