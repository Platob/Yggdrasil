"""In-memory messenger (chat) service.

Channels are deques of messages; there is no file I/O on the send path so the
node sustains >100k msg/s. The ``general`` channel always exists.
"""
from __future__ import annotations

import time
from collections import deque

from ..config import Settings
from ..schemas.messenger import Channel, Message, MessageSend

_MAX_PER_CHANNEL = 10_000


class MessengerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._channels: dict[str, deque[Message]] = {"general": deque(maxlen=_MAX_PER_CHANNEL)}
        self._seq = 0

    async def send_message(self, msg: MessageSend) -> Message:
        chan = msg.channel or "general"
        bucket = self._channels.get(chan)
        if bucket is None:
            bucket = self._channels[chan] = deque(maxlen=_MAX_PER_CHANNEL)
        self._seq += 1
        message = Message(
            id=str(self._seq),
            text=msg.text,
            sender=msg.sender,
            channel=chan,
            timestamp=time.time(),
        )
        bucket.append(message)
        return message

    async def list_channels(self) -> list[Channel]:
        return [Channel(name=name, message_count=len(msgs)) for name, msgs in self._channels.items()]

    async def get_messages(self, channel: str, *, limit: int = 50) -> list[Message]:
        bucket = self._channels.get(channel)
        if bucket is None:
            raise KeyError(f"no such channel: {channel!r}")
        if limit <= 0:
            return list(bucket)
        return list(bucket)[-limit:]

    async def get_channel(self, name: str) -> Channel:
        bucket = self._channels.get(name)
        if bucket is None:
            raise KeyError(f"no such channel: {name!r}")
        return Channel(name=name, message_count=len(bucket))

    async def create_channel(self, name: str) -> Channel:
        # Upsert: creating an existing channel just returns it.
        if name not in self._channels:
            self._channels[name] = deque(maxlen=_MAX_PER_CHANNEL)
        return Channel(name=name, message_count=len(self._channels[name]))
