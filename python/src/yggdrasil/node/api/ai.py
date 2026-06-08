"""AI analysis endpoint — structured market insight from signals + OHLCV.

When Loki (the global yggdrasil agent) is reachable the analysis is LLM-
generated; otherwise it falls back to a deterministic rule-based summary
that is still useful and never fails.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v2/ai")


class AnalysisRequest(BaseModel):
    symbol: str
    question: str | None = None


def _rule_based_analysis(symbol: str, signals: dict, quote: dict) -> str:
    """Generate a readable analysis string from signal data without an LLM."""
    sig = signals.get("signal", "HOLD")
    strength = signals.get("strength", 0.0)
    reasons = signals.get("reasons") or []
    inds = signals.get("indicators") or {}
    price = quote.get("price", 0)
    change_pct = quote.get("change_pct", 0) or 0

    direction = "up" if change_pct >= 0 else "down"
    lines = [
        f"{symbol} is trading at {price:.2f}, {direction} {abs(change_pct):.2f}% today.",
        f"Technical analysis gives a **{sig}** signal with {strength*100:.0f}% strength.",
    ]
    if reasons:
        lines.append("Key factors: " + "; ".join(reasons) + ".")
    rsi = inds.get("rsi")
    if rsi is not None:
        if rsi < 30:
            lines.append(f"RSI of {rsi:.1f} suggests the stock is oversold — potential reversal zone.")
        elif rsi > 70:
            lines.append(f"RSI of {rsi:.1f} indicates overbought conditions — caution advised.")
        else:
            lines.append(f"RSI of {rsi:.1f} is in neutral territory.")
    sma20, sma50 = inds.get("sma20"), inds.get("sma50")
    if sma20 and sma50:
        if sma20 > sma50:
            lines.append(f"The 20-day SMA ({sma20:.2f}) is above the 50-day SMA ({sma50:.2f}), confirming a short-term uptrend.")
        else:
            lines.append(f"The 20-day SMA ({sma20:.2f}) is below the 50-day SMA ({sma50:.2f}), suggesting bearish momentum.")
    if sig == "BUY":
        lines.append("Overall, the technical picture favours buyers at current levels.")
    elif sig == "SELL":
        lines.append("Overall, the technical picture suggests reducing exposure or waiting for a better entry.")
    else:
        lines.append("The mixed signals call for patience — wait for a clearer directional break.")
    lines.append("This is not financial advice. Always do your own research.")
    return " ".join(lines)


async def _loki_analysis(symbol: str, signals: dict, quote: dict, question: str | None) -> str | None:
    """Try to get an LLM analysis via Loki.  Returns None if Loki is offline."""
    try:
        from yggdrasil.loki import Loki

        loki = Loki()
        if not loki.online:
            return None
        prompt_q = question or f"Provide a concise trading analysis for {symbol}."
        context = (
            f"Symbol: {symbol}\n"
            f"Price: {quote.get('price')}, Change: {quote.get('change_pct')}%\n"
            f"Signal: {signals.get('signal')} (strength {signals.get('strength')})\n"
            f"Indicators: {signals.get('indicators')}\n"
            f"Reasons: {', '.join(signals.get('reasons') or [])}\n\n"
            f"Question: {prompt_q}"
        )
        result = await loki.arun(context)
        return str(result) if result else None
    except Exception:
        return None


@router.post("/analyze")
async def analyze_market(req: AnalysisRequest) -> dict[str, Any]:
    """Return a structured market analysis for *symbol*.

    Tries Loki (LLM) first; falls back to a deterministic rule-based summary.
    """
    from yggdrasil.node.api.market import fetch_chart, _quote_from_chart, _ohlcv_from_chart
    from yggdrasil.node.api.trading import _compute_signals

    sym = req.symbol.upper()
    chart_1d = await fetch_chart(sym, "1d", "6mo")
    ohlcv = _ohlcv_from_chart(sym, "1d", chart_1d)
    signals = _compute_signals(ohlcv)

    chart_quote = await fetch_chart(sym, "1d", "1d")
    quote = _quote_from_chart(sym, chart_quote)

    loki_text = await _loki_analysis(sym, signals, quote, req.question)
    analysis = loki_text or _rule_based_analysis(sym, signals, quote)
    source = "loki" if loki_text else "rules"

    return {
        "symbol": sym,
        "question": req.question,
        "analysis": analysis,
        "source": source,
        "recommendation": signals.get("signal"),
        "strength": signals.get("strength"),
        "signals": signals.get("indicators"),
        "quote": quote,
        "ts": int(time.time() * 1000),
    }
