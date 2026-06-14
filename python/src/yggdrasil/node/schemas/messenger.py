"""Messenger schemas."""
from __future__ import annotations

from pydantic import BaseModel


class MessageSend(BaseModel):
    text: str
    sender: str
    channel: str = "general"


class Message(BaseModel):
    id: int
    text: str
    sender: str
    channel: str
    ts: float


class Channel(BaseModel):
    name: str
    message_count: int
    created_at: float
