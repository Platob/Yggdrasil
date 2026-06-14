"""In-memory messenger — channels + bounded per-channel message rings."""
from __future__ import annotations

import time

import xxhash

from yggdrasil.node.schemas.messenger import Channel, Message, MessageSend


class MessengerService:
    """Channels and messages held in memory, bounded per channel."""

    def __init__(self, settings: object) -> None:
        self._settings = settings
        self._channels: dict[str, Channel] = {}
        self._messages: dict[str, list[Message]] = {}
        self._max_per_channel = getattr(settings, "max_messages_per_channel", 10_000)
        self._ensure_channel("general")

    def _ensure_channel(self, name: str) -> Channel:
        chan = self._channels.get(name)
        if chan is None:
            chan = Channel(name=name, message_count=0, created_at=time.time())
            self._channels[name] = chan
            self._messages[name] = []
        return chan

    async def send_message(self, msg: MessageSend) -> Message:
        chan = self._ensure_channel(msg.channel)
        ts = time.time()
        m = Message(
            id=xxhash.xxh32(f"{msg.channel}:{ts}:{msg.sender}").intdigest(),
            text=msg.text,
            sender=msg.sender,
            channel=msg.channel,
            ts=ts,
        )
        msgs = self._messages[msg.channel]
        msgs.append(m)
        if len(msgs) > self._max_per_channel:
            msgs = msgs[-self._max_per_channel:]
            self._messages[msg.channel] = msgs
        chan.message_count = len(msgs)
        return m

    async def create_channel(self, name: str) -> Channel:
        return self._ensure_channel(name)

    async def list_channels(self) -> list[Channel]:
        return list(self._channels.values())

    async def get_messages(self, channel: str, limit: int = 50) -> list[Message]:
        self._ensure_channel(channel)
        return self._messages[channel][-limit:]

    async def get_channel(self, name: str) -> Channel:
        return self._ensure_channel(name)
