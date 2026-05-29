from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_analysis_service, get_network_service
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    DescribeResult,
    FinanceRequest,
    FinanceResult,
    OhlcRequest,
    OhlcResult,
    SeriesRequest,
    SeriesResult,
)
from ..services.analysis import AnalysisService
from ..services.network import NetworkService

router = APIRouter(tags=["analysis"])


@router.post("/aggregate", response_model=AggregateResult)
async def aggregate(
    req: AggregateRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> AggregateResult:
    """Group-by / pivot aggregation (sum/mean/min/max/count/median/std/var)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/aggregate", json_body=req.model_dump())
    return await service.aggregate(req)


@router.get("/describe", response_model=DescribeResult)
async def describe(
    path: str,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> DescribeResult:
    """Per-column summary statistics (count/mean/std/min/quartiles/max)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/analysis/describe", params={"path": path})
    return await service.describe(path)


@router.post("/finance", response_model=FinanceResult)
async def finance(
    req: FinanceRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> FinanceResult:
    """Returns, cumulative return, rolling mean, and rolling volatility of a
    numeric series (optionally ordered by a column)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/finance", json_body=req.model_dump())
    return await service.finance(req)


@router.post("/series", response_model=SeriesResult)
async def series(
    req: SeriesRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> SeriesResult:
    """Adaptive downsample of a numeric series to ~`points` buckets (mean +
    min/max envelope), with an optional x zoom window pushed into the scan."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/series", json_body=req.model_dump())
    return await service.series(req)


@router.post("/ohlc", response_model=OhlcResult)
async def ohlc(
    req: OhlcRequest,
    node: str | None = None,
    service: AnalysisService = Depends(get_analysis_service),
    network: NetworkService = Depends(get_network_service),
) -> OhlcResult:
    """Resample a price series into `buckets` open/high/low/close bars
    (+ summed volume) for candlestick plotting."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/analysis/ohlc", json_body=req.model_dump())
    return await service.ohlc(req)
