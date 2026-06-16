"""Messenger schemas — channels and messages."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MessageSend(BaseModel):
    """Inbound payload for posting a message."""

    text: str
    sender: str
    channel: str = "general"


class Message(BaseModel):
    """A stored message."""

    id: str
    channel: str
    sender: str
    text: str
    created_at: datetime = Field(default_factory=_now)


class Channel(BaseModel):
    """A chat channel."""

    name: str
    created_at: datetime = Field(default_factory=_now)
    message_count: int = 0
