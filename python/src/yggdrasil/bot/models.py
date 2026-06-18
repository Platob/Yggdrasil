"""Pydantic request/response models for the YGG bot API."""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core envelopes
# ---------------------------------------------------------------------------

class OkResponse(BaseModel):
    ok: bool = True
    ts: float = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).timestamp())


class PingResponse(OkResponse):
    service: str = "ygg-bot"
    version: str


class HealthResponse(OkResponse):
    uptime_s: float
    market_cache_size: int
    ws_connections: int


class StatsResponse(OkResponse):
    requests_total: int
    ws_messages_sent: int
    cache_hits: int
    cache_misses: int
    uptime_s: float


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class PricePoint(BaseModel):
    timestamp: dt.datetime
    value: float
    unit: str
    currency: str


class PricesResponse(OkResponse):
    zone: str
    series: str
    days: int
    count: int
    prices: list[PricePoint]


class FxRate(BaseModel):
    pair: str       # e.g. "EUR/USD"
    rate: float
    date: str       # ISO date


class FxResponse(OkResponse):
    base: str
    rates: list[FxRate]
    source: str


# ---------------------------------------------------------------------------
# Trading signals
# ---------------------------------------------------------------------------

SignalKind = Literal["BUY", "SELL", "HOLD"]


class Signal(BaseModel):
    zone: str
    series: str
    kind: SignalKind
    price: float
    mean: float
    zscore: float
    ts: dt.datetime
    reason: str


class SignalsResponse(OkResponse):
    signals: list[Signal]


# ---------------------------------------------------------------------------
# AI / Loki
# ---------------------------------------------------------------------------

class AiReasonRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    engine: str | None = None
    max_steps: int = Field(default=4, ge=1, le=16)
    context: dict[str, Any] = Field(default_factory=dict)


class AiReasonResponse(OkResponse):
    answer: str
    engine: str
    steps: int


# ---------------------------------------------------------------------------
# WebSocket broadcast payload
# ---------------------------------------------------------------------------

class WsTick(BaseModel):
    kind: Literal["tick"] = "tick"
    ts: float
    zone: str
    series: str
    price: float | None
    signal: SignalKind | None
