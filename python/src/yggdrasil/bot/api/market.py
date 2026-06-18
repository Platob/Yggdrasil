"""Market data endpoints: prices and FX rates."""
from __future__ import annotations

import asyncio
import datetime as dt

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from yggdrasil.exceptions.api import BadRequestError

from ..market import fetch_fx, fetch_prices, peek_fx, peek_prices
from ..models import FxRate, FxResponse, PricePoint, PricesResponse

router = APIRouter(prefix="/api/v2/market", tags=["market"])

_VALID_ZONES = {
    "DE_LU", "FR", "NL", "BE", "AT", "CH", "GB", "ES", "IT_NORTH",
    "PL", "SE_1", "SE_2", "SE_3", "SE_4", "NO_1", "NO_2", "DK_1", "DK_2",
}
_VALID_SERIES = {"day_ahead_prices", "load", "generation"}


@router.get("/prices", response_model=PricesResponse)
async def get_prices(
    request: Request,
    zone: str = Query(default="DE_LU", description="ENTSOE bidding zone alias"),
    series: str = Query(default="day_ahead_prices"),
    days: int = Query(default=7, ge=1, le=90),
) -> PricesResponse:
    zone = zone.upper().replace("-", "_")
    if zone not in _VALID_ZONES:
        raise BadRequestError(f"Unknown zone '{zone}'. Valid: {sorted(_VALID_ZONES)}")
    if series not in _VALID_SERIES:
        raise BadRequestError(f"Unknown series '{series}'. Valid: {sorted(_VALID_SERIES)}")

    settings = request.app.state.settings
    # Fast path: synchronous cache hit avoids thread-pool overhead
    rows = peek_prices(zone, series, days)
    if rows is None:
        rows = await asyncio.to_thread(
            fetch_prices, zone, series, days,
            security_token=settings.entsoe_token,
            cache_ttl=settings.market_cache_ttl,
        )
    prices = [
        PricePoint(
            timestamp=r["timestamp"] if isinstance(r["timestamp"], dt.datetime)
                      else dt.datetime.fromisoformat(str(r["timestamp"])),
            value=float(r["value"]),
            unit=str(r.get("unit", "MWh")),
            currency=str(r.get("currency", "EUR")),
        )
        for r in rows
    ]
    return PricesResponse(zone=zone, series=series, days=days, count=len(prices), prices=prices)


@router.get("/fx", response_model=FxResponse)
async def get_fx(
    request: Request,
    base: str = Query(default="EUR"),
    targets: str = Query(default="USD,GBP,CHF,JPY,CAD"),
) -> FxResponse:
    base = base.upper()
    target_list = [t.strip().upper() for t in targets.split(",") if t.strip()]
    if not target_list:
        raise BadRequestError("At least one target currency required.")

    settings = request.app.state.settings
    # Fast path: synchronous cache hit avoids thread-pool overhead
    data = peek_fx(base, target_list)
    if data is None:
        data = await asyncio.to_thread(
            fetch_fx, base, target_list,
            cache_ttl=settings.fx_cache_ttl,
        )
    rates = [
        FxRate(pair=f"{base}/{tgt}", rate=rate, date=data.get("date", ""))
        for tgt, rate in data.get("rates", {}).items()
    ]
    return FxResponse(base=base, rates=rates, source="frankfurter.app")
