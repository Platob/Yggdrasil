"""Messenger endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from yggdrasil.exceptions.node import NodeNotFoundError

from ..schemas.messenger import Channel, Message, MessageSend
from ..services.messenger import MessengerService

router = APIRouter(prefix="/messenger", tags=["messenger"])


def _service(request: Request) -> MessengerService:
    return request.app.state.messenger


@router.post("", response_model=Message)
async def send_message(payload: MessageSend, request: Request) -> Message:
    return await _service(request).send_message(payload)


@router.get("/channels", response_model=list[Channel])
async def list_channels(request: Request) -> list[Channel]:
    return await _service(request).list_channels()


@router.post("/channels", response_model=Channel)
async def create_channel(name: str, request: Request) -> Channel:
    return await _service(request).create_channel(name)


@router.get("/channels/{channel}", response_model=Channel)
async def get_channel(channel: str, request: Request) -> Channel:
    try:
        return await _service(request).get_channel(channel)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/channels/{channel}/messages", response_model=list[Message])
async def get_messages(channel: str, request: Request, limit: int = 50) -> list[Message]:
    try:
        return await _service(request).get_messages(channel, limit=limit)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
