"""FastAPI app factory for the Yggdrasil trading node.

``create_api`` wires settings, CORS, and the v2 routers, then stashes the
shared services on ``app.state`` so route handlers resolve them off the
request without a DI framework. Services are constructed once at app build
time (market is shared into the portfolio book so positions mark against the
same synthetic feed).

The node is read-mostly and async throughout; FastAPI serializes
response-model returns straight to JSON bytes via pydantic, and tabular
endpoints upgrade to Arrow IPC via content negotiation in the transport
layer.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.schemas.base import now_ms
from .api.services import AnalysisService, FsService, MarketDataService, PortfolioService
from .api.v2 import all_routers, ping_router
from .config import Settings

__all__ = ["create_api"]


def create_api(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.node_home.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Yggdrasil Node", version="0.8.57")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    market = MarketDataService()
    app.state.settings = settings
    app.state.started_at = now_ms()
    app.state.market = market
    app.state.portfolio = PortfolioService(market)
    app.state.analysis = AnalysisService(settings.node_home)
    app.state.fs = FsService(settings.node_home)

    app.include_router(ping_router, prefix="/api")
    for router in all_routers():
        app.include_router(router, prefix="/api")

    return app
