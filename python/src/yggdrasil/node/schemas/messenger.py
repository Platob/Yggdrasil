"""Wire models for the in-memory messenger (chat) service."""
from __future__ import annotations

from pydantic import BaseModel


class MessageSend(BaseModel):
    text: str
    sender: str
    channel: str = "general"


class Message(BaseModel):
    id: str
    text: str
    sender: str
    channel: str
    timestamp: str  # ISO 8601, UTC


class Channel(BaseModel):
    id: str
    name: str
    message_count: int = 0
    created_at: str  # ISO 8601, UTC
