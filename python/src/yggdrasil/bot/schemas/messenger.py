from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class MessageSend(StrictModel):
    text: str = Field(..., min_length=1, description="Message body.")
    sender: str | None = Field(
        default=None,
        description="Display name. Defaults to the node_id when omitted.",
    )
    channel: str = Field(
        default="general",
        description="Target channel name.",
    )


class Message(StrictModel):
    id: str
    sender: str
    text: str
    channel: str
    timestamp: str
    node_id: str


class ChannelInfo(StrictModel):
    name: str
    created_at: str
    last_active: str
    message_count: int
    members: list[str]


class ChannelListResponse(StrictModel):
    node_id: str
    channels: list[ChannelInfo]


class MessageListResponse(StrictModel):
    node_id: str
    channel: str
    messages: list[Message]


class ChannelResponse(StrictModel):
    node_id: str
    channel: ChannelInfo
