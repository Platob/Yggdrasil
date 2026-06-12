from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class Indicator(BaseModel):
    name: str
    value: float
    signal: SignalDirection
    description: str = ""


class Signal(BaseModel):
    symbol: str
    direction: SignalDirection
    confidence: float  # 0-1
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    indicators: list[Indicator] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: str = "1d"

    @property
    def score(self) -> float:
        """Aggregate score from -1 (strong sell) to +1 (strong buy)."""
        mapping = {
            SignalDirection.STRONG_BUY: 1.0,
            SignalDirection.BUY: 0.5,
            SignalDirection.NEUTRAL: 0.0,
            SignalDirection.SELL: -0.5,
            SignalDirection.STRONG_SELL: -1.0,
        }
        return mapping[self.direction] * self.confidence


class AIAnalysis(BaseModel):
    symbol: str
    summary: str
    sentiment: str  # bullish/bearish/neutral
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendation: SignalDirection
    confidence: float
    model: str = "claude-sonnet-4-6"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
