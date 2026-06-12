from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.market import fetch_quote
from ..core import portfolio as svc
from ..models.portfolio import PnL, Portfolio, Trade, TradeSide

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class TradeRequest(BaseModel):
    symbol: str
    side: TradeSide
    quantity: float
    price: Optional[float] = None  # None → use live market price
    fee: float = 0.0
    notes: Optional[str] = None


@router.get("/{pid}", response_model=Portfolio)
async def get_portfolio(pid: int = 1) -> Portfolio:
    return svc.get_portfolio(pid)


@router.get("/{pid}/pnl", response_model=PnL)
async def get_pnl(pid: int = 1) -> PnL:
    portfolio = svc.get_portfolio(pid)
    # Refresh prices for all positions
    symbols = list(portfolio.positions)
    if symbols:
        import asyncio
        quotes = await asyncio.gather(*[fetch_quote(s) for s in symbols])
        svc.update_prices(pid, {q.symbol: q.price for q in quotes})
    return portfolio.pnl()


@router.post("/{pid}/trade", response_model=Trade)
async def place_trade(pid: int = 1, body: TradeRequest = ...) -> Trade:
    price = body.price
    if price is None:
        q = await fetch_quote(body.symbol.upper())
        price = q.price
        if price == 0:
            raise HTTPException(400, f"Could not fetch price for {body.symbol}")
    try:
        return svc.execute_trade(
            pid,
            symbol=body.symbol.upper(),
            side=body.side,
            quantity=body.quantity,
            price=price,
            fee=body.fee,
            notes=body.notes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{pid}/trades", response_model=list[Trade])
async def list_trades(pid: int = 1, limit: int = 50) -> list[Trade]:
    portfolio = svc.get_portfolio(pid)
    return portfolio.trades[-limit:]


@router.delete("/{pid}/position/{symbol}", status_code=204)
async def close_position(pid: int = 1, symbol: str = "", force: bool = False) -> None:
    portfolio = svc.get_portfolio(pid)
    pos = portfolio.positions.get(symbol.upper())
    if not pos:
        raise HTTPException(404, f"No position in {symbol}")
    q = await fetch_quote(symbol.upper())
    if q.price == 0 and not force:
        raise HTTPException(400, "Cannot fetch price; use ?force=true to close at avg_cost")
    price = q.price or pos.avg_cost
    svc.execute_trade(pid, symbol.upper(), TradeSide.SELL, pos.quantity, price)
