from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas.trading import PositionUpsert, TradeSignalCreate

router = APIRouter(tags=["trading"])


def _svc(request: Request):
    return request.app.state.trading_service


@router.post("/signals")
async def emit_signal(req: TradeSignalCreate, request: Request):
    sig = _svc(request).emit_signal(req)
    return {"signal": sig}


@router.get("/signals")
async def list_signals(request: Request, symbol: str | None = None, limit: int = 50):
    return _svc(request).list_signals(symbol=symbol, limit=limit)


@router.get("/signals/stream")
async def stream_signals(request: Request):
    svc = _svc(request)

    async def _generate():
        async for sig in svc.stream_signals():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(sig.model_dump())}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@router.get("/portfolio")
async def get_portfolio(request: Request):
    return _svc(request).get_portfolio()


@router.put("/portfolio/{symbol}")
async def upsert_position(symbol: str, req: PositionUpsert, request: Request):
    req = req.model_copy(update={"symbol": symbol})
    pos = _svc(request).upsert_position(req)
    return {"position": pos}


@router.delete("/portfolio/{symbol}")
async def delete_position(symbol: str, request: Request):
    deleted = _svc(request).delete_position(symbol)
    return {"deleted": deleted, "symbol": symbol.upper()}
