"""MessengerService — in-memory channels + messages.

Everything lives in process dicts: ``_channels`` maps channel name to a list
of ``Message``, ``_counts`` tracks the running count. Send appends; reads slice
the tail. There is no disk or lock — the node is single-process and the hot
path is a list append, which is what the throughput bench measures.
"""
from __future__ import annotations

import datetime as dt
import itertools
from typing import Any

from yggdrasil.node.schemas.messenger import Channel, Message, MessageSend


class MessengerService:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._channels: dict[str, list[Message]] = {"general": []}
        self._ids = itertools.count(1)

    async def send_message(self, msg: MessageSend) -> Message:
        bucket = self._channels.setdefault(msg.channel, [])
        message = Message(
            id=next(self._ids),
            text=msg.text,
            sender=msg.sender,
            channel=msg.channel,
            timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        )
        bucket.append(message)
        return message

    async def list_channels(self) -> list[Channel]:
        return [Channel(name=name, message_count=len(msgs))
                for name, msgs in self._channels.items()]

    async def get_messages(self, channel: str, limit: int = 50) -> list[Message]:
        bucket = self._channels.get(channel, [])
        return bucket[-limit:]

    async def get_channel(self, channel: str) -> Channel:
        bucket = self._channels.get(channel)
        if bucket is None:
            raise KeyError(
                f"no channel {channel!r}; known: {sorted(self._channels)}. "
                f"Create it with create_channel({channel!r})."
            )
        return Channel(name=channel, message_count=len(bucket))

    async def create_channel(self, name: str) -> Channel:
        self._channels.setdefault(name, [])
        return Channel(name=name, message_count=len(self._channels[name]))
