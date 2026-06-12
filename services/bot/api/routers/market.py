from __future__ import annotations

import io
from typing import Annotated, Optional

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, HTTPException, Query, Response

from ..core.market import fetch_ohlcv, fetch_ohlcv_arrow, fetch_quote, ohlcv_to_arrow, search_tickers
from ..models.market import OHLCV, Quote, Ticker

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/search")
async def search(q: str = Query(..., min_length=1)) -> list[Ticker]:
    return search_tickers(q)


@router.get("/quote/{symbol}")
async def quote(symbol: str) -> Quote:
    return await fetch_quote(symbol.upper())


@router.get("/quotes")
async def quotes(
    symbols: Annotated[list[str], Query()] = [],
) -> list[Quote]:
    import asyncio
    if not symbols:
        raise HTTPException(400, "No symbols provided")
    results = await asyncio.gather(*[fetch_quote(s.upper()) for s in symbols])
    return list(results)


@router.get("/ohlcv/{symbol}")
async def ohlcv(
    symbol: str,
    period: str = Query("1mo", pattern="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$"),
    interval: str = Query("1d", pattern="^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$"),
    fmt: str = Query("json", pattern="^(json|arrow)$"),
) -> Response:
    sym = symbol.upper()
    if fmt == "arrow":
        # Arrow table cache avoids repeated list→Arrow conversion
        table = await fetch_ohlcv_arrow(sym, period=period, interval=interval)
        buf = io.BytesIO()
        with ipc.new_stream(buf, table.schema) as writer:
            writer.write_table(table)
        return Response(
            content=buf.getvalue(),
            media_type="application/vnd.apache.arrow.stream",
        )

    bars = await fetch_ohlcv(sym, period=period, interval=interval)
    return Response(content=_bars_to_json(bars), media_type="application/json")


def _bars_to_json(bars: list[OHLCV]) -> str:
    import orjson
    return orjson.dumps([b.model_dump(mode="json") for b in bars]).decode()


@router.get("/sectors")
async def sectors() -> dict[str, list[str]]:
    return {
        "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
        "Financials":  ["JPM", "V", "BRK-B"],
        "Consumer":    ["AMZN", "TSLA"],
        "ETFs":        ["SPY", "QQQ"],
        "Crypto":      ["BTC-USD", "ETH-USD", "SOL-USD"],
    }
