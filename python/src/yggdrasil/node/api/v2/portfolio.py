"""Portfolio routes — book state, summary, trades, order placement/cancel."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas.portfolio import (
    CreateOrderRequest,
    Order,
    Portfolio,
    PortfolioSummary,
    Trade,
)

__all__ = ["router"]

router = APIRouter(prefix="/v2/portfolio", tags=["portfolio"])


@router.get("/{portfolio_id}", response_model=Portfolio)
async def get_portfolio(request: Request, portfolio_id: int) -> Portfolio:
    try:
        return await request.app.state.portfolio.get_portfolio(portfolio_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio {portfolio_id}")


@router.get("/{portfolio_id}/summary", response_model=PortfolioSummary)
async def get_summary(request: Request, portfolio_id: int) -> PortfolioSummary:
    try:
        return await request.app.state.portfolio.get_summary(portfolio_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio {portfolio_id}")


@router.get("/{portfolio_id}/trades", response_model=list[Trade])
async def get_trades(request: Request, portfolio_id: int, limit: int = 50) -> list[Trade]:
    try:
        return await request.app.state.portfolio.get_trades(portfolio_id, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio {portfolio_id}")


@router.post("/{portfolio_id}/order", response_model=Order)
async def create_order(
    request: Request, portfolio_id: int, body: CreateOrderRequest
) -> Order:
    try:
        return await request.app.state.portfolio.open_order(portfolio_id, body)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio {portfolio_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown symbol {exc}")


@router.delete("/{portfolio_id}/orders/{order_id}")
async def cancel_order(request: Request, portfolio_id: int, order_id: int) -> dict:
    try:
        cancelled = await request.app.state.portfolio.cancel_order(portfolio_id, order_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown portfolio {portfolio_id}")
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"No open order {order_id}")
    return {"cancelled": True}
