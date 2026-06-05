from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class Message(StrictModel):
    id: int
    channel: str
    user_id: int
    user_key: str
    content: str
    timestamp: str
    node_id: str = ""


class MessageSend(StrictModel):
    channel: str = "general"
    content: str


class ChannelInfo(StrictModel):
    name: str
    message_count: int = 0
    last_message_at: str | None = None
    members: list[str] = Field(default_factory=list)


class ChannelListResponse(StrictModel):
    node_id: str
    channels: list[ChannelInfo]


class MessageListResponse(StrictModel):
    channel: str
    messages: list[Message]
