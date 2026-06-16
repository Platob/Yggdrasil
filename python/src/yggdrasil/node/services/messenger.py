"""In-memory messenger service.

Channels and their messages live in process memory, guarded by a single
:class:`asyncio.Lock`. The ``general`` channel always exists. Sending to a
channel that doesn't exist yet creates it on the fly.
"""
from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from yggdrasil.exceptions.node import NodeNotFoundError

from ..config import Settings
from ..schemas.messenger import Channel, Message, MessageSend


def _make_id() -> str:
    """xxhash-free composite is overkill here — a uuid4 hex is plenty for
    in-memory, process-local message identity."""
    return uuid4().hex


class MessengerService:
    """Channel + message store with async-safe mutation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._channels: dict[str, Channel] = {"general": Channel(name="general")}
        self._messages: dict[str, list[Message]] = {"general": []}

    async def send_message(self, msg: MessageSend) -> Message:
        async with self._lock:
            channel = self._channels.get(msg.channel)
            if channel is None:
                channel = Channel(name=msg.channel)
                self._channels[msg.channel] = channel
                self._messages[msg.channel] = []
            message = Message(
                id=_make_id(),
                channel=msg.channel,
                sender=msg.sender,
                text=msg.text,
            )
            self._messages[msg.channel].append(message)
            channel.message_count += 1
            return message

    async def create_channel(self, name: str) -> Channel:
        async with self._lock:
            existing = self._channels.get(name)
            if existing is not None:
                return existing
            channel = Channel(name=name)
            self._channels[name] = channel
            self._messages[name] = []
            return channel

    async def list_channels(self) -> list[Channel]:
        async with self._lock:
            return list(self._channels.values())

    async def get_channel(self, name: str) -> Channel:
        async with self._lock:
            channel = self._channels.get(name)
            if channel is None:
                raise NodeNotFoundError(
                    f"No channel named {name!r}. Known channels: "
                    f"{sorted(self._channels)}. Create it with create_channel()."
                )
            return channel

    async def get_messages(self, channel: str, *, limit: int = 50) -> list[Message]:
        async with self._lock:
            messages = self._messages.get(channel)
            if messages is None:
                raise NodeNotFoundError(
                    f"No channel named {channel!r}. Known channels: "
                    f"{sorted(self._channels)}."
                )
            if limit <= 0:
                return []
            return messages[-limit:]
