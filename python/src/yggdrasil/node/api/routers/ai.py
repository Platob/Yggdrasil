from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..schemas.analysis import AiAnalyzeRequest

router = APIRouter(tags=["ai"])


@router.post("/analyze")
async def ai_analyze(req: AiAnalyzeRequest) -> StreamingResponse:
    """Stream an LLM analysis of trading data as SSE. Falls back to a static
    message (no model call) when ANTHROPIC_API_KEY is absent, so the endpoint
    always streams something useful."""

    async def _fallback():
        msg = (
            "Configure ANTHROPIC_API_KEY in your environment to enable AI "
            f"analysis. Current indicators show: {req.summary}"
        )
        yield f"data: {msg}\n\n"
        yield "data: [DONE]\n\n"

    if not os.getenv("ANTHROPIC_API_KEY"):
        return StreamingResponse(_fallback(), media_type="text/event-stream")

    async def _stream():
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            f"{req.question}\n\nTrading data / indicator summary:\n{req.summary}"
        )
        try:
            with client.messages.stream(
                model=req.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {text}\n\n"
        except Exception as exc:
            yield f"data: AI analysis failed: {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
