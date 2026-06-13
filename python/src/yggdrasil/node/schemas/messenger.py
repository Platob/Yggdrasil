"""Messenger (chat) schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class MessageSend(BaseModel):
    text: str
    sender: str = "anonymous"
    channel: str = "general"


class Message(BaseModel):
    id: str
    text: str
    sender: str
    channel: str
    timestamp: float


class Channel(BaseModel):
    name: str
    message_count: int
