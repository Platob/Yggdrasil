"""In-memory chat messenger — channels of bounded message history.

Messages live in a ``dict[channel, deque]`` guarded by an asyncio lock;
each channel keeps its last :data:`_CHANNEL_MAXLEN` messages. IDs are
xxhash composites so two messages sent in the same millisecond still
differ. No persistence — this is the live, hot-path chat surface.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections import deque

import xxhash

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.messenger import Channel, Message, MessageSend

_CHANNEL_MAXLEN = 10_000


class MessengerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._messages: dict[str, deque[Message]] = {"general": deque(maxlen=_CHANNEL_MAXLEN)}
        self._created: dict[str, str] = {"general": _now_iso()}
        self._lock = asyncio.Lock()

    async def send_message(self, msg: MessageSend) -> Message:
        ts_ms = time.time_ns() // 1_000_000
        mid = (xxhash.xxh32(f"{msg.channel}:{msg.text}:{msg.sender}".encode()).intdigest() << 32) | (ts_ms & 0xFFFFFFFF)
        message = Message(
            id=mid, text=msg.text, sender=msg.sender,
            channel=msg.channel, timestamp=_now_iso(),
        )
        async with self._lock:
            chan = self._messages.get(msg.channel)
            if chan is None:
                chan = self._messages[msg.channel] = deque(maxlen=_CHANNEL_MAXLEN)
                self._created[msg.channel] = message.timestamp
            chan.append(message)
        return message

    async def get_messages(self, channel: str, limit: int = 100) -> list[Message]:
        async with self._lock:
            chan = self._messages.get(channel)
            if chan is None:
                raise NotFoundError(f"No such channel: {channel!r}.")
            # deque slicing isn't supported; take the tail without copying the whole buffer.
            n = len(chan)
            start = max(0, n - limit)
            return [chan[i] for i in range(start, n)]

    async def list_channels(self) -> list[Channel]:
        async with self._lock:
            return [
                Channel(name=name, message_count=len(msgs), created_at=self._created[name])
                for name, msgs in self._messages.items()
            ]

    async def get_channel(self, name: str) -> Channel:
        async with self._lock:
            msgs = self._messages.get(name)
            if msgs is None:
                raise NotFoundError(f"No such channel: {name!r}.")
            return Channel(name=name, message_count=len(msgs), created_at=self._created[name])

    async def create_channel(self, name: str) -> Channel:
        async with self._lock:
            if name not in self._messages:
                self._messages[name] = deque(maxlen=_CHANNEL_MAXLEN)
                self._created[name] = _now_iso()
            return Channel(name=name, message_count=len(self._messages[name]), created_at=self._created[name])


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
