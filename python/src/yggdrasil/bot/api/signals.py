"""Trading signal endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, Request

from yggdrasil.exceptions.api import BadRequestError

from ..market import fetch_prices, peek_prices
from ..models import SignalsResponse
from ..signals import compute_signals

router = APIRouter(prefix="/api/v2/signals", tags=["signals"])

_VALID_ZONES = {
    "DE_LU", "FR", "NL", "BE", "AT", "CH", "GB", "ES", "IT_NORTH",
    "PL", "SE_1", "SE_2", "SE_3", "SE_4", "NO_1", "NO_2", "DK_1", "DK_2",
}


@router.get("", response_model=SignalsResponse)
async def get_signals(
    request: Request,
    zone: str = Query(default="DE_LU"),
    series: str = Query(default="day_ahead_prices"),
    days: int = Query(default=7, ge=3, le=90),
) -> SignalsResponse:
    zone = zone.upper().replace("-", "_")
    if zone not in _VALID_ZONES:
        raise BadRequestError(f"Unknown zone '{zone}'.")

    settings = request.app.state.settings
    prices = peek_prices(zone, series, days)
    if prices is None:
        prices = await asyncio.to_thread(
            fetch_prices, zone, series, days,
            security_token=settings.entsoe_token,
            cache_ttl=settings.market_cache_ttl,
        )
    signals = compute_signals(prices, zone, series)
    return SignalsResponse(signals=signals)
