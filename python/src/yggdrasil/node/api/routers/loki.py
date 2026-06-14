"""Loki AI chat endpoint — routes a message through the global Loki agent."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    engine: str = "claude"
    history: list[dict] = []


@router.post("/loki/chat")
async def chat(req: ChatRequest):
    try:
        from yggdrasil.loki.agent import Loki

        loki = Loki()
        reply = loki.reason(req.message)
        return {"reply": reply, "engine": getattr(loki, "engine_name", req.engine)}
    except Exception as exc:
        return {"reply": f"Loki unavailable: {exc}", "engine": None}
