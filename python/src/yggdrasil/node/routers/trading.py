from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ..deps import get_trading_service
from ..schemas.trading import (
    AlertResponse,
    AlertsResponse,
    OrderCreate,
    OrderResponse,
    OrdersResponse,
    PortfolioSummary,
    PriceAlertCreate,
    PriceQuote,
    PricesResponse,
    SignalsResponse,
    TradeHistoryResponse,
    TradingSignal,
    WatchlistAdd,
    WatchlistEntryResponse,
    WatchlistResponse,
)
from ..services.trading import TradingService

router = APIRouter(tags=["trading"])


# -- prices ---------------------------------------------------------------

@router.get("/prices", response_model=PricesResponse)
async def get_prices(
    response: Response,
    service: TradingService = Depends(get_trading_service),
) -> PricesResponse:
    response.headers["Cache-Control"] = "max-age=1"
    return PricesResponse(prices=service.get_all_prices())


@router.get("/prices/stream")
async def stream_prices(
    interval: float = 1.0,
    service: TradingService = Depends(get_trading_service),
) -> StreamingResponse:
    interval = max(0.5, min(interval, 30.0))

    async def _gen():
        while True:
            payload = {"prices": [p.model_dump() for p in service.get_all_prices()]}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/prices/{symbol}", response_model=PriceQuote)
async def get_price(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> PriceQuote:
    return service.get_price(symbol)


# -- portfolio ------------------------------------------------------------

@router.get("/portfolio", response_model=PortfolioSummary)
async def get_portfolio(
    service: TradingService = Depends(get_trading_service),
) -> PortfolioSummary:
    return service.get_portfolio()


# -- orders ---------------------------------------------------------------

@router.post("/orders", response_model=OrderResponse)
async def place_order(
    req: OrderCreate,
    service: TradingService = Depends(get_trading_service),
) -> OrderResponse:
    return OrderResponse(order=service.place_order(req))


@router.get("/orders", response_model=OrdersResponse)
async def list_orders(
    service: TradingService = Depends(get_trading_service),
) -> OrdersResponse:
    return OrdersResponse(orders=service.get_orders())


@router.delete("/orders/{order_id}", response_model=OrderResponse)
async def cancel_order(
    order_id: int,
    service: TradingService = Depends(get_trading_service),
) -> OrderResponse:
    return OrderResponse(order=service.cancel_order(order_id))


# -- watchlist ------------------------------------------------------------

@router.get("/watchlist", response_model=WatchlistResponse)
async def get_watchlist(
    service: TradingService = Depends(get_trading_service),
) -> WatchlistResponse:
    return WatchlistResponse(entries=service.get_watchlist())


@router.post("/watchlist", response_model=WatchlistEntryResponse)
async def add_watchlist(
    req: WatchlistAdd,
    service: TradingService = Depends(get_trading_service),
) -> WatchlistEntryResponse:
    return WatchlistEntryResponse(entry=service.add_watchlist(req.symbol))


@router.delete("/watchlist/{symbol}", response_model=WatchlistEntryResponse)
async def remove_watchlist(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> WatchlistEntryResponse:
    return WatchlistEntryResponse(entry=service.remove_watchlist(symbol))


# -- signals --------------------------------------------------------------

@router.get("/signals", response_model=SignalsResponse)
async def get_signals(
    service: TradingService = Depends(get_trading_service),
) -> SignalsResponse:
    return SignalsResponse(signals=service.get_all_signals())


@router.get("/signals/{symbol}", response_model=TradingSignal)
async def get_signal(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> TradingSignal:
    return service.get_signal(symbol)


# -- alerts ---------------------------------------------------------------

@router.post("/alerts", response_model=AlertResponse)
async def create_alert(
    req: PriceAlertCreate,
    service: TradingService = Depends(get_trading_service),
) -> AlertResponse:
    return AlertResponse(alert=service.create_alert(req))


@router.get("/alerts", response_model=AlertsResponse)
async def list_alerts(
    service: TradingService = Depends(get_trading_service),
) -> AlertsResponse:
    return AlertsResponse(alerts=service.list_alerts())


@router.delete("/alerts/{alert_id}", response_model=AlertResponse)
async def delete_alert(
    alert_id: int,
    service: TradingService = Depends(get_trading_service),
) -> AlertResponse:
    return AlertResponse(alert=service.delete_alert(alert_id))


# -- history --------------------------------------------------------------

@router.get("/history", response_model=TradeHistoryResponse)
async def get_history(
    service: TradingService = Depends(get_trading_service),
) -> TradeHistoryResponse:
    return TradeHistoryResponse(trades=service.get_trade_history())
