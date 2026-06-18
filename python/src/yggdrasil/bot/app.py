"""YGG Bot — FastAPI application factory.

Usage::

    from yggdrasil.bot.app import create_app
    app = create_app()               # uses BotSettings from env
    # or:
    app = create_app(BotSettings(port=9000, entsoe_token="..."))

Run directly::

    uvicorn yggdrasil.bot.app:app --reload
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from yggdrasil.exceptions.api import register_api_exception_handlers
from yggdrasil.version import __version__

from .api.core import router as core_router
from .api.market import router as market_router
from .api.signals import router as signals_router
from .api.ai import router as ai_router
from .config import BotSettings
from .ws import broadcast_loop, manager

log = logging.getLogger(__name__)


def create_app(settings: BotSettings | None = None) -> FastAPI:
    settings = settings or BotSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        task = asyncio.create_task(broadcast_loop(settings))
        log.info("ygg-bot %s ready (ws_tick=%.1fs)", __version__, settings.ws_tick_interval)
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    application = FastAPI(
        title="YGG Bot",
        description="Trading + AI backend — ENTSOE day-ahead prices, FX rates, Loki AI.",
        version=__version__,
        lifespan=lifespan,
    )
    # Settings available immediately (before lifespan runs, e.g. in tests)
    application.state.settings = settings

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_api_exception_handlers(application)

    application.include_router(core_router)
    application.include_router(market_router)
    application.include_router(signals_router)
    application.include_router(ai_router)

    @application.websocket("/ws/market")
    async def ws_market(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                # Keep connection alive; client can send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await manager.disconnect(ws)

    @application.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return application


# ASGI-compatible app instance for `uvicorn yggdrasil.bot.app:app`
app = create_app()
