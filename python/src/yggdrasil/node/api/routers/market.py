from __future__ import annotations
from fastapi import APIRouter, Depends
from ..deps import get_market_service
from ..schemas.market import MarketOHLCV, MarketQuote, MarketSearchResult
from ..services.market import MarketService

router = APIRouter(tags=["market"])


@router.get("/quote/{symbol}", response_model=MarketQuote)
async def get_quote(symbol: str, service: MarketService = Depends(get_market_service)) -> MarketQuote:
    """Current market quote for a symbol (30s TTL cache)."""
    return await service.quote(symbol)


@router.get("/ohlcv/{symbol}", response_model=MarketOHLCV)
async def get_ohlcv(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    service: MarketService = Depends(get_market_service),
) -> MarketOHLCV:
    """Historical OHLCV bars. period: 1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max. interval: 1m|5m|15m|1h|1d|1wk|1mo."""
    return await service.ohlcv(symbol, period, interval)


@router.get("/search", response_model=list[MarketSearchResult])
async def search_symbols(q: str, service: MarketService = Depends(get_market_service)) -> list[MarketSearchResult]:
    """Search for ticker symbols."""
    return await service.search(q)
