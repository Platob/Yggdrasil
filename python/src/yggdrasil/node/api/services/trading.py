from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from threading import Lock

from ...config import Settings
from ...ids import make_id
from ..schemas.trading import (
    Portfolio,
    PortfolioResponse,
    Position,
    PositionUpsert,
    SignalListResponse,
    TradeSignal,
    TradeSignalCreate,
)

_MAX_SIGNALS = 500


class TradingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._signals: deque[TradeSignal] = deque(maxlen=_MAX_SIGNALS)
        self._positions: dict[str, Position] = {}
        self._lock = Lock()
        self._subscribers: list[asyncio.Queue[TradeSignal]] = []

    # -- signals ---------------------------------------------------------------

    def emit_signal(self, req: TradeSignalCreate) -> TradeSignal:
        now = datetime.now(timezone.utc).isoformat()
        sig = TradeSignal(
            id=make_id(f"{req.symbol}:{req.direction}:{now}"),
            func_id=req.func_id,
            name=req.name,
            symbol=req.symbol.upper(),
            direction=req.direction.lower(),
            confidence=max(0.0, min(1.0, req.confidence)),
            price=req.price,
            metadata=req.metadata,
            created_at=now,
            expires_at=req.expires_at,
        )
        with self._lock:
            self._signals.append(sig)
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(sig)
            except asyncio.QueueFull:
                pass
        return sig

    def list_signals(self, symbol: str | None = None, limit: int = 50) -> SignalListResponse:
        with self._lock:
            items = list(self._signals)
        if symbol:
            items = [s for s in items if s.symbol == symbol.upper()]
        items = items[-limit:][::-1]
        return SignalListResponse(node_id=self.settings.node_id, signals=items)

    async def stream_signals(self):
        q: asyncio.Queue[TradeSignal] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers.append(q)
        try:
            while True:
                sig = await q.get()
                yield sig
        finally:
            with self._lock:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    # -- portfolio -------------------------------------------------------------

    def upsert_position(self, req: PositionUpsert) -> Position:
        now = datetime.now(timezone.utc).isoformat()
        symbol = req.symbol.upper()
        pnl: float | None = None
        pnl_pct: float | None = None
        if req.current_price is not None and req.avg_price > 0:
            pnl = (req.current_price - req.avg_price) * req.qty
            pnl_pct = round((req.current_price - req.avg_price) / req.avg_price * 100, 2)
        pos = Position(
            symbol=symbol,
            qty=req.qty,
            avg_price=req.avg_price,
            current_price=req.current_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            opened_at=now,
            updated_at=now,
        )
        with self._lock:
            existing = self._positions.get(symbol)
            if existing:
                pos = pos.model_copy(update={"opened_at": existing.opened_at})
            self._positions[symbol] = pos
        return pos

    def delete_position(self, symbol: str) -> bool:
        with self._lock:
            return self._positions.pop(symbol.upper(), None) is not None

    def get_portfolio(self) -> PortfolioResponse:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            positions = list(self._positions.values())
        total_pnl = sum(p.pnl or 0.0 for p in positions)
        total_cost = sum(p.avg_price * abs(p.qty) for p in positions)
        total_pnl_pct = round(total_pnl / total_cost * 100, 2) if total_cost > 0 else 0.0
        return PortfolioResponse(
            node_id=self.settings.node_id,
            portfolio=Portfolio(
                positions=positions,
                total_pnl=round(total_pnl, 4),
                total_pnl_pct=total_pnl_pct,
                updated_at=now,
            ),
        )
