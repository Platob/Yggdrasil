"""AI endpoints — Loki reasoning via JSON response and SSE streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from yggdrasil.exceptions.api import BadRequestError

from ..models import AiReasonRequest, AiReasonResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v2/ai", tags=["ai"])


def _loki():
    """Lazy Loki import — heavy; only loaded on first AI call."""
    from yggdrasil.loki.agent import Loki
    return Loki()


@router.post("/reason", response_model=AiReasonResponse)
async def reason(request: Request, body: AiReasonRequest) -> AiReasonResponse:
    """Synchronous AI reasoning: returns the full answer in one JSON response."""
    settings = request.app.state.settings
    engine_pin = body.engine or settings.loki_engine

    def _run() -> tuple[str, str, int]:
        agent = _loki()
        result = agent.reason(
            body.prompt,
            engine=engine_pin,
            context=body.context or {},
        )
        answer = str(result) if not isinstance(result, str) else result
        engine_used = engine_pin or "auto"
        return answer, engine_used, 0

    try:
        answer, engine_used, steps = await asyncio.to_thread(_run)
    except Exception as exc:
        log.warning("loki reason error: %s", exc)
        raise BadRequestError(f"AI reasoning failed: {exc}") from exc

    return AiReasonResponse(answer=answer, engine=engine_used, steps=steps)


@router.post("/stream")
async def stream(request: Request, body: AiReasonRequest) -> StreamingResponse:
    """SSE streaming AI response — yields data: <chunk> lines."""
    settings = request.app.state.settings
    engine_pin = body.engine or settings.loki_engine

    async def _gen():
        yield f"data: {json.dumps({'kind': 'start', 'ts': time.time()})}\n\n"
        try:
            def _run_stream():
                agent = _loki()
                # reason_stream is a generator of text chunks
                return agent.reason_stream(
                    body.prompt,
                    engine=engine_pin,
                    context=body.context or {},
                )

            gen = await asyncio.to_thread(_run_stream)
            for chunk in gen:
                payload = json.dumps({"kind": "chunk", "text": str(chunk)})
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)          # yield control to event loop
        except Exception as exc:
            log.warning("loki stream error: %s", exc)
            yield f"data: {json.dumps({'kind': 'error', 'detail': str(exc)})}\n\n"
        finally:
            yield f"data: {json.dumps({'kind': 'done', 'ts': time.time()})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
