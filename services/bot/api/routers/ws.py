from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.market import fetch_quote

router = APIRouter(tags=["realtime"])

# Active connections keyed by subscription set
_connections: dict[WebSocket, set[str]] = {}


async def _broadcast_quotes(ws: WebSocket, symbols: set[str], interval: float = 3.0) -> None:
    """Push quote updates to a single WebSocket at `interval` seconds."""
    while True:
        quotes = await asyncio.gather(*[fetch_quote(s) for s in symbols], return_exceptions=True)
        payload = []
        for q in quotes:
            if isinstance(q, Exception):
                continue
            payload.append({
                "symbol": q.symbol,
                "price": q.price,
                "change": q.change,
                "change_pct": q.change_pct,
                "ts": q.timestamp.isoformat(),
            })
        try:
            await ws.send_text(json.dumps({"type": "quotes", "data": payload}))
        except Exception:
            break
        await asyncio.sleep(interval)


@router.websocket("/ws/market")
async def market_ws(ws: WebSocket) -> None:
    """
    WebSocket endpoint for real-time market data.

    Client sends:  {"action": "subscribe",   "symbols": ["AAPL", "MSFT"]}
                   {"action": "unsubscribe",  "symbols": ["MSFT"]}
    Server sends:  {"type": "quotes", "data": [{...}, ...]}
    """
    await ws.accept()
    subscribed: set[str] = set()
    push_task: Optional[asyncio.Task] = None

    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            msg = json.loads(raw)
            action = msg.get("action", "")
            symbols = {s.upper() for s in msg.get("symbols", [])}

            if action == "subscribe" and symbols:
                subscribed |= symbols
                if push_task:
                    push_task.cancel()
                push_task = asyncio.create_task(_broadcast_quotes(ws, subscribed))
                await ws.send_text(json.dumps({"type": "subscribed", "symbols": list(subscribed)}))

            elif action == "unsubscribe" and symbols:
                subscribed -= symbols
                if push_task:
                    push_task.cancel()
                if subscribed:
                    push_task = asyncio.create_task(_broadcast_quotes(ws, subscribed))
                await ws.send_text(json.dumps({"type": "subscribed", "symbols": list(subscribed)}))

            elif action == "ping":
                await ws.send_text(json.dumps({"type": "pong", "ts": time.time()}))

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if push_task:
            push_task.cancel()
