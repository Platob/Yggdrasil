"""Bot app factory.

:func:`create_app` builds the FastAPI bot surface (messenger, functions,
monitor, discovery, ping) mounted under ``/api``. Services are constructed
once and stashed on ``app.state`` so routers stay thin.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .routers import function as function_router
from .routers import hello as hello_router
from .routers import messenger as messenger_router
from .routers import monitor as monitor_router
from .routers import ping as ping_router
from .services.function import FunctionService
from .services.messenger import MessengerService
from .services.monitor import MonitorService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="yggdrasil.node", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.allow_remote else ["http://localhost"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.messenger = MessengerService(settings)
    app.state.function = FunctionService(settings)
    app.state.monitor = MonitorService(settings)

    app.include_router(ping_router.router, prefix="/api")
    app.include_router(messenger_router.router, prefix="/api")
    app.include_router(function_router.router, prefix="/api")
    app.include_router(monitor_router.router, prefix="/api")
    app.include_router(hello_router.router, prefix="/api")

    return app
