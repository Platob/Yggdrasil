"""Messenger endpoints — send messages, manage channels, fetch history."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from yggdrasil.node.schemas.messenger import MessageSend

router = APIRouter()


@router.post("/messenger")
async def send(request: Request, msg: MessageSend):
    return (await request.app.state.messenger.send_message(msg)).model_dump()


@router.get("/messenger/channels")
async def list_channels(request: Request):
    chans = await request.app.state.messenger.list_channels()
    return {"channels": [c.model_dump() for c in chans]}


@router.post("/messenger/channels")
async def create_channel(request: Request, name: str = Query(...)):
    return (await request.app.state.messenger.create_channel(name)).model_dump()


@router.get("/messenger/channels/{name}")
async def get_channel(request: Request, name: str):
    return (await request.app.state.messenger.get_channel(name)).model_dump()


@router.get("/messenger/channels/{name}/messages")
async def get_messages(request: Request, name: str, limit: int = Query(50)):
    msgs = await request.app.state.messenger.get_messages(name, limit=limit)
    return {"messages": [m.model_dump() for m in msgs]}
