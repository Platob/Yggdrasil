from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_network_service, get_trading_service
from ..schemas.trading import (
    CorrelationRequest,
    CorrelationResult,
    IndicatorRequest,
    IndicatorResult,
    PortfolioRequest,
    PortfolioResult,
    SignalRequest,
    SignalResult,
    VaRRequest,
    VaRResult,
)
from ..services.network import NetworkService
from ..services.trading import TradingService

router = APIRouter(tags=["trading"])


@router.post("/indicators", response_model=IndicatorResult)
async def indicators(
    req: IndicatorRequest,
    node: str | None = None,
    service: TradingService = Depends(get_trading_service),
    network: NetworkService = Depends(get_network_service),
) -> IndicatorResult:
    """Compute RSI, MACD, Bollinger Bands, ATR for a price series."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/trading/indicators", json_body=req.model_dump())
    return await service.indicators(req)


@router.post("/correlation", response_model=CorrelationResult)
async def correlation(
    req: CorrelationRequest,
    node: str | None = None,
    service: TradingService = Depends(get_trading_service),
    network: NetworkService = Depends(get_network_service),
) -> CorrelationResult:
    """Pearson or Spearman correlation matrix between multiple price series."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/trading/correlation", json_body=req.model_dump())
    return await service.correlation(req)


@router.post("/portfolio", response_model=PortfolioResult)
async def portfolio(
    req: PortfolioRequest,
    node: str | None = None,
    service: TradingService = Depends(get_trading_service),
    network: NetworkService = Depends(get_network_service),
) -> PortfolioResult:
    """Multi-asset portfolio analytics: weighted return, risk metrics, correlation."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/trading/portfolio", json_body=req.model_dump())
    return await service.portfolio(req)


@router.post("/var", response_model=VaRResult)
async def value_at_risk(
    req: VaRRequest,
    node: str | None = None,
    service: TradingService = Depends(get_trading_service),
    network: NetworkService = Depends(get_network_service),
) -> VaRResult:
    """Value-at-Risk: historical simulation, parametric (normal), or Cornish-Fisher."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/trading/var", json_body=req.model_dump())
    return await service.var(req)


@router.post("/signals", response_model=SignalResult)
async def signals(
    req: SignalRequest,
    node: str | None = None,
    service: TradingService = Depends(get_trading_service),
    network: NetworkService = Depends(get_network_service),
) -> SignalResult:
    """Generate trade signals from RSI, MACD crossover, and Bollinger Band analysis."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/trading/signals", json_body=req.model_dump())
    return await service.signals(req)
