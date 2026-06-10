from __future__ import annotations

from pydantic import BaseModel


class MessageSend(BaseModel):
    text: str
    sender: str = "user"
    channel: str = "general"


class Message(BaseModel):
    id: int
    text: str
    sender: str
    channel: str
    timestamp: str


class Channel(BaseModel):
    name: str
    message_count: int
    created_at: str


class ChannelListResponse(BaseModel):
    channels: list[Channel]


class MessageListResponse(BaseModel):
    messages: list[Message]
    channel: str
    total: int
