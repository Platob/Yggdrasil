from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["ai"])


class AnalyzeRequest(BaseModel):
    func_id: int
    query: str | None = None


class QueryRequest(BaseModel):
    question: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


def _svc(request: Request):
    return request.app.state.ai_service


@router.get("/status")
async def ai_status(request: Request):
    svc = _svc(request)
    return {"available": svc.available}


@router.post("/analyze")
async def analyze_func(req: AnalyzeRequest, request: Request):
    return await _svc(request).analyze_func(req.func_id, query=req.query)


@router.post("/query")
async def query(req: QueryRequest, request: Request):
    return await _svc(request).query(req.question)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    svc = _svc(request)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async def _generate():
        async for chunk in svc.stream_chat(messages):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
