from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..deps import get_trading_service
from ..middleware import cached_response
from ..schemas.trading import (
    AlertCreate,
    AlertsResponse,
    PortfolioResponse,
    PositionCreate,
    PriceAlert,
    PriceHistoryResponse,
    PricesResponse,
    TechnicalIndicators,
)
from ..services.trading import TradingService

router = APIRouter(tags=["trading"])


@router.get("", response_model=PricesResponse)
@cached_response(ttl_seconds=30.0)
async def get_default_prices(
    service: TradingService = Depends(get_trading_service),
) -> PricesResponse:
    return service.get_prices()


@router.get("/prices", response_model=PricesResponse)
@cached_response(ttl_seconds=30.0)
async def get_prices(
    symbols: str = Query("", description="Comma-separated symbols, e.g. EUR/USD,BTC-USD"),
    service: TradingService = Depends(get_trading_service),
) -> PricesResponse:
    if symbols.strip():
        parsed = tuple(s.strip() for s in symbols.split(",") if s.strip())
    else:
        parsed = None
    return service.get_prices(parsed)


@router.get("/history/{symbol:path}", response_model=PriceHistoryResponse)
async def get_price_history(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> PriceHistoryResponse:
    return service.get_price_history(symbol)


@router.get("/stream")
async def stream_prices(
    symbols: str = Query("", description="Comma-separated symbols"),
    interval: float = Query(5.0, ge=1.0, le=60.0),
    service: TradingService = Depends(get_trading_service),
):
    parsed = None
    if symbols.strip():
        parsed = tuple(s.strip() for s in symbols.split(",") if s.strip())

    async def _generate():
        async for chunk in service.stream_prices(parsed, interval):
            yield chunk

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    service: TradingService = Depends(get_trading_service),
) -> PortfolioResponse:
    return service.get_portfolio()


@router.post("/portfolio/position", response_model=PortfolioResponse)
async def upsert_position(
    body: PositionCreate,
    service: TradingService = Depends(get_trading_service),
) -> PortfolioResponse:
    service.upsert_position(body.symbol, body.quantity, body.avg_cost, body.currency)
    return service.get_portfolio()


@router.delete("/portfolio/position/{symbol:path}", response_model=PortfolioResponse)
async def remove_position(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> PortfolioResponse:
    service.remove_position(symbol)
    return service.get_portfolio()


@router.get("/indicators/{symbol:path}", response_model=TechnicalIndicators)
@cached_response(ttl_seconds=5.0)
async def get_indicators(
    symbol: str,
    service: TradingService = Depends(get_trading_service),
) -> TechnicalIndicators:
    return service.get_indicators(symbol)


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    service: TradingService = Depends(get_trading_service),
) -> AlertsResponse:
    return service.get_alerts()


@router.post("/alerts", response_model=PriceAlert)
async def create_alert(
    body: AlertCreate,
    service: TradingService = Depends(get_trading_service),
) -> PriceAlert:
    return service.set_alert(body)


@router.delete("/alerts/{alert_id}", response_model=PriceAlert)
async def remove_alert(
    alert_id: int,
    service: TradingService = Depends(get_trading_service),
) -> PriceAlert:
    return service.remove_alert(alert_id)
