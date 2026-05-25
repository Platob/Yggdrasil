from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_messenger_service
from ..schemas.messenger import (
    ChannelListResponse,
    ChannelResponse,
    Message,
    MessageListResponse,
    MessageSend,
)
from ..services.messenger import MessengerService

router = APIRouter(tags=["messenger"])


@router.post("", response_model=Message)
async def send_message(
    req: MessageSend,
    service: MessengerService = Depends(get_messenger_service),
) -> Message:
    return await service.send_message(req)


@router.get("/channels", response_model=ChannelListResponse)
async def list_channels(
    service: MessengerService = Depends(get_messenger_service),
) -> ChannelListResponse:
    return await service.list_channels()


@router.post("/channels", response_model=ChannelResponse)
async def create_channel(
    name: str = Query(..., description="Channel name to create."),
    service: MessengerService = Depends(get_messenger_service),
) -> ChannelResponse:
    return await service.create_channel(name)


@router.get("/channels/{name}", response_model=ChannelResponse)
async def get_channel(
    name: str,
    service: MessengerService = Depends(get_messenger_service),
) -> ChannelResponse:
    return await service.get_channel(name)


@router.delete("/channels/{name}", response_model=ChannelResponse)
async def delete_channel(
    name: str,
    service: MessengerService = Depends(get_messenger_service),
) -> ChannelResponse:
    return await service.delete_channel(name)


@router.get("/channels/{name}/messages", response_model=MessageListResponse)
async def get_messages(
    name: str,
    limit: int = Query(default=50, ge=1, le=1000, description="Max messages to return."),
    after: str | None = Query(default=None, description="ISO timestamp; return messages after this time."),
    service: MessengerService = Depends(get_messenger_service),
) -> MessageListResponse:
    return await service.get_messages(name, limit=limit, after=after)


@router.get("/channels/{name}/poll", response_model=MessageListResponse)
async def poll_messages(
    name: str,
    after_id: str | None = Query(default=None, description="Message id; return messages after this id."),
    timeout: float = Query(default=30.0, ge=0, le=120, description="Long-poll timeout in seconds."),
    service: MessengerService = Depends(get_messenger_service),
) -> MessageListResponse:
    return await service.poll_messages(name, after_id=after_id, timeout=timeout)
