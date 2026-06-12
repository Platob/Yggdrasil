from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.market import fetch_ohlcv, fetch_quote
from ..core.signals import generate_signals
from ..models.signal import AIAnalysis, SignalDirection

router = APIRouter(prefix="/ai", tags=["ai"])


class AnalysisRequest(BaseModel):
    symbol: str
    context: Optional[str] = None


@router.post("/analyze", response_model=AIAnalysis)
async def analyze(body: AnalysisRequest) -> AIAnalysis:
    """Generate an AI-driven analysis using Claude via the Anthropic API."""
    symbol = body.symbol.upper()

    # Gather context
    quote, bars = await _gather_context(symbol)
    signals = generate_signals(symbol, bars)

    prompt = _build_prompt(symbol, quote, signals, body.context)

    try:
        summary, sentiment, key_factors, risks = await _call_claude(prompt)
    except Exception as e:
        # Graceful degradation: return rule-based analysis when AI unavailable
        summary = f"Technical analysis for {symbol}: {signals.direction.value} signal with {signals.confidence:.0%} confidence."
        sentiment = _direction_to_sentiment(signals.direction)
        key_factors = [i.description for i in signals.indicators if i.description]
        risks = ["AI service unavailable — using technical analysis only"]

    return AIAnalysis(
        symbol=symbol,
        summary=summary,
        sentiment=sentiment,
        key_factors=key_factors,
        risks=risks,
        recommendation=signals.direction,
        confidence=signals.confidence,
    )


@router.get("/scan", response_model=list[AIAnalysis])
async def scan(
    symbols: list[str] = Query(default=["AAPL", "MSFT", "NVDA"]),
) -> list[AIAnalysis]:
    import asyncio
    return list(await asyncio.gather(*[
        analyze(AnalysisRequest(symbol=s)) for s in symbols
    ]))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

async def _gather_context(symbol: str):
    import asyncio
    quote, bars = await asyncio.gather(
        fetch_quote(symbol),
        fetch_ohlcv(symbol, period="3mo", interval="1d"),
    )
    return quote, bars


def _build_prompt(symbol, quote, signals, extra_context: Optional[str]) -> str:
    ind_lines = "\n".join(
        f"  - {i.name}: {i.value} ({i.signal.value}) — {i.description}"
        for i in signals.indicators
    )
    return f"""You are a quantitative trading analyst. Analyze {symbol} and provide a concise assessment.

Current data:
- Price: ${quote.price:.2f} ({quote.change_pct:+.2f}%)
- Technical signal: {signals.direction.value} (confidence: {signals.confidence:.0%})
- Indicators:
{ind_lines}
{f'- Additional context: {extra_context}' if extra_context else ''}

Respond in exactly this JSON format:
{{
  "summary": "<2-3 sentence market assessment>",
  "sentiment": "<bullish|bearish|neutral>",
  "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "risks": ["<risk1>", "<risk2>"]
}}"""


async def _call_claude(prompt: str) -> tuple[str, str, list[str], list[str]]:
    import anthropic
    import json as _json

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Extract JSON from potential markdown block
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    data = _json.loads(text)
    return (
        data.get("summary", ""),
        data.get("sentiment", "neutral"),
        data.get("key_factors", []),
        data.get("risks", []),
    )


def _direction_to_sentiment(d: SignalDirection) -> str:
    if d in (SignalDirection.BUY, SignalDirection.STRONG_BUY):
        return "bullish"
    if d in (SignalDirection.SELL, SignalDirection.STRONG_SELL):
        return "bearish"
    return "neutral"
