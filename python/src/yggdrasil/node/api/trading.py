"""Trading signals, portfolio, and market-scan endpoints.

Signal computation runs on Polars — vectorised, no row-level Python loops.
The portfolio is in-process for now (no DB dependency); a persistent
backing store can be dropped in later by replacing ``_POSITIONS`` /
``_HISTORY`` with real persistence calls.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from yggdrasil.exceptions.api import BadRequestError, UnprocessableError
from yggdrasil.lazy_imports import polars as pl
from yggdrasil.node.api.market import fetch_chart, _ohlcv_from_chart, _quote_from_chart
from yggdrasil.node.signals import compute_signals

router = APIRouter(prefix="/api/v2/trading")

# ---------------------------------------------------------------------------
# signal computation
# ---------------------------------------------------------------------------

_DEFAULT_WATCHLIST = "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,BRK-B"


def _compute_signals(ohlcv: dict) -> dict:
    """Compute the signal dict from an OHLCV payload (``{"data": [...]}``).

    Thin adapter over :func:`yggdrasil.node.signals.compute_signals`, which
    owns the vectorised Polars math. Bars are sorted ascending by time before
    the rolling windows run.
    """
    rows = ohlcv.get("data") or []
    if not rows:
        return {"signal": "HOLD", "strength": 0.0, "score": 0.0, "reasons": [], "indicators": {}, "prices": []}
    # compute_signals guards the <20-bar warmup case itself.
    return compute_signals(pl.DataFrame(rows).sort("ts"))


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@router.get("/signals/{symbol}")
async def get_signals(symbol: str, interval: str = "1d") -> dict:
    """Trading signals for *symbol* derived from 6 months of OHLCV data."""
    chart = await fetch_chart(symbol, interval, "6mo")
    ohlcv = _ohlcv_from_chart(symbol, interval, chart)
    signals = _compute_signals(ohlcv)
    quote_chart = await fetch_chart(symbol, "1d", "1d")
    quote = _quote_from_chart(symbol, quote_chart)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "quote": quote,
        **signals,
    }


@router.get("/scan")
async def market_scan(symbols: str = _DEFAULT_WATCHLIST) -> dict:
    """Compute signals for all symbols and rank by signal strength."""
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    async def _one(sym: str) -> dict | None:
        try:
            chart = await fetch_chart(sym, "1d", "6mo")
            ohlcv = _ohlcv_from_chart(sym, "1d", chart)
            sigs = _compute_signals(ohlcv)
            quote = _quote_from_chart(sym, chart)
            return {"symbol": sym, **sigs, "price": quote.get("price"), "change_pct": quote.get("change_pct")}
        except Exception:
            return None

    results = await asyncio.gather(*(_one(s) for s in tickers))
    rows = [r for r in results if r is not None]

    order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    rows.sort(key=lambda r: (order.get(r["signal"], 1), -r["strength"]))
    return {"scan": rows, "ts": int(time.time() * 1000)}


# ---------------------------------------------------------------------------
# portfolio (in-process)
# ---------------------------------------------------------------------------

# symbol → {shares, avg_cost, opened_at}
_POSITIONS: dict[str, dict[str, Any]] = {}
_HISTORY: list[dict[str, Any]] = []


class TradeRequest(BaseModel):
    symbol: str
    action: str          # "BUY" or "SELL"
    shares: float
    price: float | None = None  # None → fetch live


@router.get("/portfolio")
async def get_portfolio() -> dict:
    """All open positions with live P&L."""
    if not _POSITIONS:
        return {"positions": [], "total_value": 0.0, "total_pnl": 0.0, "total_pnl_pct": 0.0}

    async def _enrich(sym: str, pos: dict) -> dict:
        try:
            chart = await fetch_chart(sym, "1d", "1d")
            q = _quote_from_chart(sym, chart)
            price = q.get("price") or pos["avg_cost"]
        except Exception:
            price = pos["avg_cost"]
        cost = pos["avg_cost"] * pos["shares"]
        value = price * pos["shares"]
        pnl = value - cost
        return {
            "symbol": sym,
            "shares": pos["shares"],
            "avg_cost": round(pos["avg_cost"], 4),
            "current_price": round(price, 4),
            "cost_basis": round(cost, 4),
            "market_value": round(value, 4),
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl / cost * 100, 4) if cost else 0.0,
        }

    positions = await asyncio.gather(*(_enrich(s, p) for s, p in _POSITIONS.items()))
    total_value = sum(p["market_value"] for p in positions)
    total_cost = sum(p["cost_basis"] for p in positions)
    total_pnl = total_value - total_cost
    return {
        "positions": list(positions),
        "total_value": round(total_value, 4),
        "total_cost": round(total_cost, 4),
        "total_pnl": round(total_pnl, 4),
        "total_pnl_pct": round(total_pnl / total_cost * 100, 4) if total_cost else 0.0,
    }


@router.post("/portfolio/trade")
async def execute_trade(req: TradeRequest) -> dict:
    """Execute a simulated BUY or SELL trade."""
    sym = req.symbol.upper()
    action = req.action.upper()
    if action not in ("BUY", "SELL"):
        raise BadRequestError(
            f"Trade action must be 'BUY' or 'SELL', got {req.action!r}."
        )
    if req.shares <= 0:
        raise BadRequestError(
            f"Trade shares must be positive, got {req.shares}. Use action='SELL' "
            f"to reduce a position."
        )

    price = req.price
    if price is None:
        chart = await fetch_chart(sym, "1d", "1d")
        q = _quote_from_chart(sym, chart)
        price = q.get("price")
        if not price:
            raise UnprocessableError(
                f"No live price available for {sym!r}; pass an explicit 'price' "
                f"to record this trade."
            )

    trade = {"symbol": sym, "action": action, "shares": req.shares, "price": price, "ts": int(time.time() * 1000)}
    _HISTORY.append(trade)

    pos = _POSITIONS.get(sym)
    if action == "BUY":
        if pos is None:
            _POSITIONS[sym] = {"shares": req.shares, "avg_cost": price}
        else:
            total_shares = pos["shares"] + req.shares
            pos["avg_cost"] = (pos["avg_cost"] * pos["shares"] + price * req.shares) / total_shares
            pos["shares"] = total_shares
    else:  # SELL
        if pos is None or pos["shares"] < req.shares:
            have = pos["shares"] if pos else 0
            raise UnprocessableError(
                f"Cannot sell {req.shares} shares of {sym!r}; position holds only "
                f"{have}. Reduce the quantity or buy first."
            )
        pos["shares"] -= req.shares
        if pos["shares"] <= 0:
            del _POSITIONS[sym]

    return {"trade": trade, "position": _POSITIONS.get(sym)}


@router.get("/portfolio/history")
async def get_portfolio_history() -> list:
    """Full trade history (most recent first)."""
    return list(reversed(_HISTORY))
