"""In-memory messenger (chat) service.

State lives in plain dicts on the instance — channels keyed by name, messages
keyed by channel name. No persistence; the node loses chat history on restart.
That's intentional: this is a lightweight presence/notification channel for the
front, not a system of record. IDs are xxh32 composites of a semantic key and a
monotonic counter so they're stable ints rendered as strings on the wire.
"""
from __future__ import annotations

import datetime as dt
import itertools

import xxhash

from ..config import Settings
from ..schemas.messenger import Channel, Message, MessageSend


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class MessengerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._channels: dict[str, Channel] = {}
        self._messages: dict[str, list[Message]] = {}
        self._counter = itertools.count(1)
        # Every node has a "general" channel from boot so the first send never
        # has to create one.
        self._make_channel("general")

    def _make_channel(self, name: str) -> Channel:
        cid = xxhash.xxh32(name.encode()).intdigest()
        chan = Channel(id=str(cid), name=name, message_count=0, created_at=_now_iso())
        self._channels[name] = chan
        self._messages[name] = []
        return chan

    async def send_message(self, msg: MessageSend) -> Message:
        chan = self._channels.get(msg.channel)
        if chan is None:
            chan = self._make_channel(msg.channel)
        seq = next(self._counter)
        mid = xxhash.xxh32(f"{msg.channel}:{seq}".encode()).intdigest() << 32 | seq
        message = Message(
            id=str(mid),
            text=msg.text,
            sender=msg.sender,
            channel=msg.channel,
            timestamp=_now_iso(),
        )
        self._messages[msg.channel].append(message)
        chan.message_count += 1
        return message

    async def list_channels(self) -> list[Channel]:
        return list(self._channels.values())

    async def get_messages(self, channel: str, limit: int = 50) -> list[Message]:
        msgs = self._messages.get(channel)
        if msgs is None:
            raise KeyError(
                f"No channel {channel!r}. Known channels: {sorted(self._channels)}. "
                f"Send a message to it (or call create_channel) to create it."
            )
        # Most-recent-last; return the tail window.
        return msgs[-limit:] if limit else list(msgs)

    async def get_channel(self, channel: str) -> Channel:
        chan = self._channels.get(channel)
        if chan is None:
            raise KeyError(
                f"No channel {channel!r}. Known channels: {sorted(self._channels)}."
            )
        return chan

    async def create_channel(self, name: str) -> Channel:
        existing = self._channels.get(name)
        if existing is not None:
            return existing
        return self._make_channel(name)
