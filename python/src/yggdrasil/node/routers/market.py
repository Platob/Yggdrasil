from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from ..deps import get_market_service
from ..middleware import cached_response
from ..schemas.market import (
    FxConvertResponse,
    FxHistoryResponse,
    FxLatestResponse,
    WatchlistAddRequest,
    WatchlistResponse,
)
from ..services.market import MarketService

router = APIRouter(tags=["market"])


@router.get("/fx/latest", response_model=FxLatestResponse)
@cached_response(ttl_seconds=30.0)
async def get_fx_latest(
    pairs: str = Query(
        "EUR/USD,GBP/USD,USD/JPY",
        description="Comma-separated pairs in SOURCE/TARGET format",
    ),
    service: MarketService = Depends(get_market_service),
) -> FxLatestResponse:
    pair_list = [p.strip() for p in pairs.split(",") if p.strip()]
    if not pair_list:
        raise HTTPException(400, "At least one pair required")
    try:
        return await service.get_latest(pair_list)
    except Exception as exc:
        raise HTTPException(503, f"FX data unavailable: {exc}") from exc


@router.get("/fx/history", response_model=FxHistoryResponse)
@cached_response(ttl_seconds=300.0)
async def get_fx_history(
    pair: str = Query("EUR/USD"),
    days: int = Query(30, ge=1, le=365),
    service: MarketService = Depends(get_market_service),
) -> FxHistoryResponse:
    try:
        return await service.get_history(pair, days)
    except Exception as exc:
        raise HTTPException(503, f"FX history unavailable: {exc}") from exc


@router.get("/fx/convert", response_model=FxConvertResponse)
@cached_response(ttl_seconds=30.0)
async def convert_fx(
    amount: float = Query(1.0, gt=0),
    source: str = Query("EUR"),
    target: str = Query("USD"),
    service: MarketService = Depends(get_market_service),
) -> FxConvertResponse:
    try:
        return await service.convert(amount, source.upper(), target.upper())
    except Exception as exc:
        raise HTTPException(503, f"FX conversion unavailable: {exc}") from exc


@router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    service: MarketService = Depends(get_market_service),
) -> WatchlistResponse:
    return service.get_watchlist()


@router.post("/watchlist", response_model=WatchlistResponse)
async def add_to_watchlist(
    req: WatchlistAddRequest,
    service: MarketService = Depends(get_market_service),
) -> WatchlistResponse:
    try:
        return service.add_to_watchlist(req.pair)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/watchlist/{pair:path}", response_model=WatchlistResponse)
async def remove_from_watchlist(
    pair: str,
    service: MarketService = Depends(get_market_service),
) -> WatchlistResponse:
    try:
        return service.remove_from_watchlist(pair)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/watchlist/rates", response_model=FxLatestResponse)
@cached_response(ttl_seconds=60.0)
async def get_watchlist_rates(
    service: MarketService = Depends(get_market_service),
) -> FxLatestResponse:
    watchlist = service.get_watchlist()
    if not watchlist.pairs:
        return FxLatestResponse(
            rates=[], cached=False,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
    pairs = [w.pair for w in watchlist.pairs]
    try:
        return await service.get_latest(pairs)
    except Exception as exc:
        raise HTTPException(503, f"FX rates unavailable: {exc}") from exc
