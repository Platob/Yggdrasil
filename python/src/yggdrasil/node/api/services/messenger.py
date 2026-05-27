from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from collections import defaultdict
from threading import Lock
from typing import AsyncIterator

from ...config import Settings
from ...ids import make_id
from ..schemas.messenger import (
    ChannelInfo,
    ChannelListResponse,
    Message,
    MessageListResponse,
)

LOGGER = logging.getLogger(__name__)


class MessengerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._channels: dict[str, list[Message]] = defaultdict(list)
        self._lock = Lock()
        # Per-channel event that fires when a new message arrives, waking SSE streams
        self._channel_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)

        # Auto-create the default channel
        self._channels["general"] = []

    def send(
        self,
        channel: str,
        user_id: int,
        user_key: str,
        content: str,
        node_id: str = "",
    ) -> Message:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        msg_id = make_id(f"msg:{channel}:{now}")
        msg = Message(
            id=msg_id,
            channel=channel,
            user_id=user_id,
            user_key=user_key,
            content=content,
            timestamp=now,
            node_id=node_id or self.settings.node_id,
        )
        with self._lock:
            self._channels[channel].append(msg)
        # Wake any SSE streams waiting for new messages on this channel
        self._channel_events[channel].set()
        return msg

    def list_channels(self) -> ChannelListResponse:
        with self._lock:
            channels: list[ChannelInfo] = []
            for name, messages in self._channels.items():
                members: list[str] = list({m.user_key for m in messages})
                channels.append(ChannelInfo(
                    name=name,
                    message_count=len(messages),
                    last_message_at=messages[-1].timestamp if messages else None,
                    members=members,
                ))
        return ChannelListResponse(node_id=self.settings.node_id, channels=channels)

    def get_messages(self, channel: str, limit: int = 50) -> MessageListResponse:
        with self._lock:
            messages = list(self._channels[channel][-limit:])
        return MessageListResponse(channel=channel, messages=messages)

    async def stream_messages(self, channel: str) -> AsyncIterator[Message]:
        """Yield new messages as they arrive. Wakes immediately on new messages via asyncio.Event."""
        event = self._channel_events[channel]
        with self._lock:
            last_seen = len(self._channels[channel])
        while True:
            event.clear()
            with self._lock:
                msgs = self._channels[channel]
                current_len = len(msgs)
                if current_len > last_seen:
                    new_msgs = msgs[last_seen:current_len]
                    last_seen = current_len
                else:
                    new_msgs = []
            for msg in new_msgs:
                yield msg
            if not new_msgs:
                # Wait for a new message or timeout (keeps connection alive with heartbeats)
                try:
                    await asyncio.wait_for(event.wait(), timeout=30.0)
                except TimeoutError:
                    pass

