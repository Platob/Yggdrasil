from __future__ import annotations

from fastapi import APIRouter, Query

from ..core.market import fetch_ohlcv
from ..core.signals import generate_signals
from ..models.signal import Signal

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/{symbol}", response_model=Signal)
async def get_signal(
    symbol: str,
    period: str = Query("3mo", pattern="^(1mo|3mo|6mo|1y|2y)$"),
    interval: str = Query("1d", pattern="^(1d|1wk)$"),
) -> Signal:
    bars = await fetch_ohlcv(symbol.upper(), period=period, interval=interval)
    return generate_signals(symbol.upper(), bars, timeframe=interval)


@router.get("/batch/scan")
async def scan_signals(
    symbols: list[str] = Query(default=["AAPL", "MSFT", "NVDA", "BTC-USD", "SPY"]),
    period: str = Query("3mo"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> list[Signal]:
    import asyncio
    bars_list = await asyncio.gather(*[
        fetch_ohlcv(s.upper(), period=period, interval="1d") for s in symbols
    ])
    signals = [generate_signals(s.upper(), bars) for s, bars in zip(symbols, bars_list)]
    return [s for s in signals if s.confidence >= min_confidence]
