from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from collections import deque
from threading import Lock

from ..config import Settings
from ..exceptions import BotError, NotFoundError
from ..schemas.messenger import (
    ChannelInfo,
    ChannelListResponse,
    ChannelResponse,
    Message,
    MessageListResponse,
    MessageSend,
)

LOGGER = logging.getLogger(__name__)

_CHANNEL_TTL = 86400  # 1 day in seconds
_MAX_MESSAGES = 1000


class _Channel:
    __slots__ = (
        "name", "messages", "members", "created_at", "last_active",
        "_last_active_mono", "_notify", "_subscribers",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.messages: deque[Message] = deque(maxlen=_MAX_MESSAGES)
        self.members: set[str] = set()
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        self.created_at: str = now
        self.last_active: str = now
        self._last_active_mono: float = time.monotonic()
        self._notify: asyncio.Event | None = None
        # SSE subscriber queues — each connected client gets its own asyncio.Queue
        self._subscribers: set[asyncio.Queue[Message]] = set()

    def touch(self) -> None:
        self.last_active = dt.datetime.now(dt.timezone.utc).isoformat()
        self._last_active_mono = time.monotonic()

    def to_info(self) -> ChannelInfo:
        return ChannelInfo(
            name=self.name,
            created_at=self.created_at,
            last_active=self.last_active,
            message_count=len(self.messages),
            members=sorted(self.members),
        )

    def get_event(self) -> asyncio.Event:
        """Return the notification event, creating it lazily."""
        if self._notify is None:
            self._notify = asyncio.Event()
        return self._notify

    def wake_pollers(self, msg: Message | None = None) -> None:
        if self._notify is not None:
            self._notify.set()
            self._notify.clear()
        if msg is not None and self._subscribers:
            # Fan out to all SSE subscribers. Drop the message on a full
            # queue rather than block other subscribers.
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

    def subscribe(self) -> asyncio.Queue[Message]:
        q: asyncio.Queue[Message] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Message]) -> None:
        self._subscribers.discard(q)


class MessengerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._channels: dict[str, _Channel] = {}
        self._lock = Lock()
        # Seed the default channel.
        self._channels["general"] = _Channel("general")

    # -- public API --------------------------------------------------------

    async def send_message(self, req: MessageSend) -> Message:
        sender = req.sender or self.settings.node_id
        msg = Message(
            id=uuid.uuid4().hex[:12],
            sender=sender,
            text=req.text,
            channel=req.channel,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            node_id=self.settings.node_id,
        )

        with self._lock:
            self._purge_stale()
            ch = self._channels.get(req.channel)
            if ch is None:
                ch = _Channel(req.channel)
                self._channels[req.channel] = ch
            ch.messages.append(msg)
            ch.members.add(sender)
            ch.touch()

        LOGGER.info("Sent message %r in channel %r (sender=%s)", msg.id, req.channel, sender)

        # Wake any long-pollers and SSE subscribers outside the lock.
        ch.wake_pollers(msg)
        return msg

    async def list_channels(self) -> ChannelListResponse:
        with self._lock:
            self._purge_stale()
            infos = [ch.to_info() for ch in self._channels.values()]
        return ChannelListResponse(
            node_id=self.settings.node_id,
            channels=infos,
        )

    async def get_channel(self, name: str) -> ChannelResponse:
        with self._lock:
            self._purge_stale()
            ch = self._channels.get(name)
        if ch is None:
            raise NotFoundError(f"Channel {name!r} not found")
        return ChannelResponse(
            node_id=self.settings.node_id,
            channel=ch.to_info(),
        )

    async def create_channel(self, name: str) -> ChannelResponse:
        with self._lock:
            self._purge_stale()
            if name in self._channels:
                raise BotError(f"Channel {name!r} already exists", status_code=409)
            ch = _Channel(name)
            self._channels[name] = ch

        LOGGER.info("Created channel %r", name)
        return ChannelResponse(
            node_id=self.settings.node_id,
            channel=ch.to_info(),
        )

    async def delete_channel(self, name: str) -> ChannelResponse:
        if name == "general":
            raise BotError("Cannot delete the default 'general' channel", status_code=403)
        with self._lock:
            ch = self._channels.pop(name, None)
        if ch is None:
            raise NotFoundError(f"Channel {name!r} not found")
        LOGGER.info("Deleted channel %r", name)
        return ChannelResponse(
            node_id=self.settings.node_id,
            channel=ch.to_info(),
        )

    async def get_messages(
        self,
        channel: str,
        *,
        limit: int = 50,
        after: str | None = None,
    ) -> MessageListResponse:
        with self._lock:
            self._purge_stale()
            ch = self._channels.get(channel)
        if ch is None:
            raise NotFoundError(f"Channel {channel!r} not found")

        msgs = list(ch.messages)
        if after is not None:
            # Return only messages with a timestamp strictly after the given value.
            msgs = [m for m in msgs if m.timestamp > after]
        msgs = msgs[-limit:]

        return MessageListResponse(
            node_id=self.settings.node_id,
            channel=channel,
            messages=msgs,
        )

    async def poll_messages(
        self,
        channel: str,
        *,
        after_id: str | None = None,
        timeout: float = 30.0,
    ) -> MessageListResponse:
        with self._lock:
            self._purge_stale()
            ch = self._channels.get(channel)
        if ch is None:
            raise NotFoundError(f"Channel {channel!r} not found")

        # Collect messages newer than after_id (if given).
        new = self._messages_after_id(ch, after_id)
        if new:
            return MessageListResponse(
                node_id=self.settings.node_id,
                channel=channel,
                messages=new,
            )

        # No new messages yet -- wait for the channel's notification event.
        event = ch.get_event()
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            pass

        new = self._messages_after_id(ch, after_id)
        return MessageListResponse(
            node_id=self.settings.node_id,
            channel=channel,
            messages=new,
        )

    async def stream_messages(self, channel: str):
        """Async generator yielding new messages for an SSE stream.

        Subscribes the caller to the channel's fan-out queue. Yields each
        new message as it is published. The generator runs forever until
        the client disconnects; cleanup unsubscribes the queue.
        """
        with self._lock:
            ch = self._channels.get(channel)
            if ch is None:
                ch = _Channel(channel)
                self._channels[channel] = ch

        q = ch.subscribe()
        try:
            # Replay-on-subscribe: yield the most recent message so the
            # client immediately has context, then stream live updates.
            if ch.messages:
                yield ch.messages[-1]
            while True:
                msg = await q.get()
                yield msg
        finally:
            ch.unsubscribe(q)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _messages_after_id(ch: _Channel, after_id: str | None) -> list[Message]:
        if after_id is None:
            return []
        found = False
        result: list[Message] = []
        for m in ch.messages:
            if found:
                result.append(m)
            elif m.id == after_id:
                found = True
        return result

    def _purge_stale(self) -> None:
        """Remove channels inactive for longer than ``_CHANNEL_TTL``.

        Must be called while holding ``self._lock``.  The ``general``
        channel is never purged.  Uses monotonic time to avoid ISO
        string parsing on every call.
        """
        now = time.monotonic()
        stale: list[str] = []
        for name, ch in self._channels.items():
            if name == "general":
                continue
            if now - ch._last_active_mono > _CHANNEL_TTL:
                stale.append(name)
        for name in stale:
            del self._channels[name]
            LOGGER.debug("Purged stale channel %r", name)
