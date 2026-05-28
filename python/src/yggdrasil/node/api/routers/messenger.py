from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_messenger_service, get_user_service
from ..schemas.messenger import (
    ChannelListResponse,
    Message,
    MessageListResponse,
    MessageSend,
)
from ..services.messenger import MessengerService
from ..services.user import UserService

router = APIRouter(tags=["messenger"])


@router.get("/channels", response_model=ChannelListResponse)
async def list_channels(
    service: MessengerService = Depends(get_messenger_service),
) -> ChannelListResponse:
    return service.list_channels()


@router.get("/{channel}", response_model=MessageListResponse)
async def get_messages(
    channel: str,
    limit: int = 50,
    service: MessengerService = Depends(get_messenger_service),
) -> MessageListResponse:
    return service.get_messages(channel, limit=limit)


@router.post("/{channel}", response_model=Message)
async def send_message(
    channel: str,
    body: MessageSend,
    messenger: MessengerService = Depends(get_messenger_service),
    user_service: UserService = Depends(get_user_service),
) -> Message:
    user = user_service.get_self()
    return messenger.send(
        channel=channel,
        user_id=user.user_id,
        user_key=user.key,
        content=body.content,
        node_id=user.node_id,
    )


@router.get("/{channel}/stream")
async def stream_messages(
    channel: str,
    service: MessengerService = Depends(get_messenger_service),
) -> StreamingResponse:
    """SSE stream of new messages in a channel."""

    async def event_stream():
        async for msg in service.stream_messages(channel):
            payload = msg.model_dump()
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
