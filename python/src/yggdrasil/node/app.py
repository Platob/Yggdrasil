"""FastAPI application factory for ``yggdrasil.node``.

``create_app`` wires together all routers, CORS, and exception handlers.
``serve`` starts uvicorn with the settings from the environment.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from yggdrasil.exceptions.api import register_api_exception_handlers
from yggdrasil.node.config import Settings

_START_TIME = time.time()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _START_TIME
    _START_TIME = time.time()
    yield
    # Close shared httpx clients on shutdown.
    try:
        from yggdrasil.node.api.market import _CLIENT
        if _CLIENT is not None and not _CLIENT.is_closed:
            await _CLIENT.aclose()
    except Exception:
        pass


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the configured :class:`FastAPI` application."""
    cfg = settings or Settings.from_env()

    app = FastAPI(
        title="YGG Node",
        description="Yggdrasil trading backend — market data, signals, portfolio, AI analysis.",
        version="2.0.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_api_exception_handlers(app)

    if cfg.api_key:
        @app.middleware("http")
        async def _require_api_key(request: Request, call_next):
            # Liveness probe stays open so orchestrators can health-check.
            if request.url.path.startswith("/api/ping"):
                return await call_next(request)
            if request.headers.get("X-API-Key") != cfg.api_key:
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing X-API-Key.", "status": 401},
                )
            return await call_next(request)

    from yggdrasil.node.api import health, call, market, trading, ai

    app.include_router(health.router)
    app.include_router(call.router)
    app.include_router(market.router)
    app.include_router(trading.router)
    app.include_router(ai.router)

    # Store settings on app state so middleware and tests can read it.
    app.state.settings = cfg
    app.state.start_time = _START_TIME

    return app


def serve(settings: Settings | None = None) -> None:
    """Start a uvicorn server with the yggdrasil node app."""
    import uvicorn

    cfg = settings or Settings.from_env()
    uvicorn.run(
        create_app(cfg),
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )
