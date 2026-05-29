from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_analysis_service, get_network_service
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    DescribeResult,
    FinanceRequest,
    FinanceResult,
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
