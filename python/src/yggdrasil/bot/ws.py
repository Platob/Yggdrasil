"""WebSocket connection manager and tick broadcaster.

ConnectionManager keeps a set of live WebSocket connections and provides
a broadcast() helper.  The background broadcaster task runs on app startup
and sends a WsTick every `tick_interval` seconds.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import WebSocket

if TYPE_CHECKING:
    from .config import BotSettings

log = logging.getLogger(__name__)

_stats = {"messages_sent": 0, "connections": 0}


class ConnectionManager:
    def __init__(self) -> None:
        self._active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.add(ws)
            _stats["connections"] = len(self._active)
        log.debug("ws connect: %d active", len(self._active))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active.discard(ws)
            _stats["connections"] = len(self._active)
        log.debug("ws disconnect: %d active", len(self._active))

    async def broadcast(self, payload: dict) -> None:
        text = json.dumps(payload)
        dead: list[WebSocket] = []
        async with self._lock:
            snapshot = list(self._active)
        for ws in snapshot:
            try:
                await ws.send_text(text)
                _stats["messages_sent"] += 1
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._active.discard(ws)
                _stats["connections"] = len(self._active)

    def active_count(self) -> int:
        return _stats["connections"]

    def messages_sent(self) -> int:
        return _stats["messages_sent"]


manager = ConnectionManager()


async def broadcast_loop(settings: "BotSettings") -> None:
    """Background task: emit a market tick to all WS clients periodically."""
    import asyncio

    from .market import fetch_prices
    from .signals import compute_signals

    zone = "DE_LU"
    series = "day_ahead_prices"

    while True:
        await asyncio.sleep(settings.ws_tick_interval)
        try:
            prices = await asyncio.to_thread(
                fetch_prices, zone, series, 1,
                security_token=settings.entsoe_token,
                cache_ttl=settings.market_cache_ttl,
            )
            latest = prices[-1]["value"] if prices else None
            sigs = compute_signals(prices, zone, series)
            signal_kind = sigs[0].kind if sigs else None

            payload = {
                "kind": "tick",
                "ts": time.time(),
                "zone": zone,
                "series": series,
                "price": latest,
                "signal": signal_kind,
            }
            await manager.broadcast(payload)
        except Exception as exc:
            log.debug("broadcast_loop error: %s", exc)
