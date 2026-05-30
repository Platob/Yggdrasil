"""Market data endpoints — watchlist + quotes + history (yfinance)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from ..deps import get_market_service

router = APIRouter(tags=["market"])


@router.get("/watchlist")
async def get_watchlist(request: Request):
    return get_market_service(request).get_watchlist()


@router.post("/watchlist")
async def add_watchlist(request: Request, body: dict):
    ticker = body.get("ticker", "").strip()
    if not ticker:
        from yggdrasil.exceptions.api import BadRequestError
        raise BadRequestError("ticker required")
    return get_market_service(request).add_to_watchlist(ticker)


@router.delete("/watchlist/{ticker}")
async def remove_watchlist(request: Request, ticker: str):
    return get_market_service(request).remove_from_watchlist(ticker)


@router.get("/quote")
async def get_quote(request: Request, ticker: str):
    return await get_market_service(request).get_quote(ticker)


@router.get("/history")
async def get_history(
    request: Request, ticker: str, period: str = "1mo", interval: str = "1d"
):
    return await get_market_service(request).get_history(ticker, period, interval)
