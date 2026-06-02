from __future__ import annotations
from fastapi import APIRouter, Depends
from ..deps import get_network_service, get_technical_service
from ..schemas.technical import TechnicalRequest, TechnicalResult
from ..services.network import NetworkService
from ..services.technical import TechnicalService

router = APIRouter(tags=["technical"])


@router.post("/", response_model=TechnicalResult)
async def compute_indicators(
    req: TechnicalRequest,
    node: str | None = None,
    service: TechnicalService = Depends(get_technical_service),
    network: NetworkService = Depends(get_network_service),
) -> TechnicalResult:
    """Compute technical analysis indicators (RSI/MACD/BB/SMA/EMA/ATR/VWAP/OBV/Stoch) over a price file."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/technical/", json_body=req.model_dump())
    return await service.compute(req)
