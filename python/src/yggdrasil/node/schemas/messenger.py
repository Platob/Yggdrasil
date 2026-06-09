"""Messenger contracts: send a message, channels, messages."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MessageSend(BaseModel):
    text: str
    sender: str
    channel: str = "general"


class Message(BaseModel):
    id: int
    text: str
    sender: str
    channel: str
    timestamp: str


class Channel(BaseModel):
    name: str
    message_count: int = 0
