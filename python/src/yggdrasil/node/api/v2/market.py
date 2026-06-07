"""Market data routes — assets, candles, ticks, order books.

Candles use a pre-serialised JSON-bytes cache in the service: the route
returns a raw ``Response`` on every path so FastAPI never re-serialises the
Pydantic model (200 rows × 8 fields × per-field validators adds up fast).
``Accept: application/vnd.apache.arrow.stream`` gets the cached Arrow table.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from ...transport import CONTENT_TYPE_ARROW_STREAM, arrow_response
from ..schemas.market import AssetInfo, OrderBook, Tick

__all__ = ["router"]

router = APIRouter(prefix="/v2/market", tags=["market"])


@router.get("/assets", response_model=list[AssetInfo])
async def assets(request: Request) -> list[AssetInfo]:
    return await request.app.state.market.get_assets()


@router.get("/candles")
async def candles(request: Request, symbol: str, interval: str = "1h", limit: int = 200):
    try:
        json_bytes, arrow_table = request.app.state.market.get_candles_cached(
            symbol, interval, limit
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol!r}")

    accept = request.headers.get("accept") or ""
    if CONTENT_TYPE_ARROW_STREAM in accept:
        return arrow_response(arrow_table)
    return Response(content=json_bytes, media_type="application/json")


@router.get("/tick", response_model=Tick)
async def tick(request: Request, symbol: str) -> Tick:
    try:
        return await request.app.state.market.get_tick(symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol!r}")


@router.get("/book", response_model=OrderBook)
async def book(request: Request, symbol: str, depth: int = 10) -> OrderBook:
    try:
        return await request.app.state.market.get_book(symbol, depth)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol {symbol!r}")
