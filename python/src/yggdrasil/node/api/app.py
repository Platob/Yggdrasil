"""create_api — assembles the node's REST surface and wires the services.

Services are constructed once and parked on ``app.state`` so routers reach them
through ``request.app.state`` without per-request construction. ``create_app``
(in :mod:`yggdrasil.node.app`) layers the ``/api/call`` remote dispatch on top.
"""
from __future__ import annotations

from fastapi import FastAPI

from yggdrasil.exceptions.api import register_api_exception_handlers


def create_api(settings=None) -> FastAPI:
    from yggdrasil.node.config import Settings

    settings = settings or Settings()

    app = FastAPI(title="Yggdrasil Node API", version="2.0")
    register_api_exception_handlers(app)

    from yggdrasil.node.api.services.audit import AuditLog
    from yggdrasil.node.api.services.fs import FsService
    from yggdrasil.node.api.services.market import MarketService
    from yggdrasil.node.api.services.tabular import TabularService
    from yggdrasil.node.api.services.analysis import AnalysisService
    from yggdrasil.node.services.function import FunctionService
    from yggdrasil.node.services.messenger import MessengerService
    from yggdrasil.node.services.monitor import MonitorService

    fs = FsService(settings)
    app.state.settings = settings
    app.state.fs = fs
    app.state.tabular = TabularService(settings, fs=fs)
    app.state.analysis = AnalysisService(settings, fs=fs)
    app.state.market = MarketService(settings)
    app.state.audit = AuditLog(settings)
    app.state.functions = FunctionService(settings)
    app.state.messenger = MessengerService(settings)
    app.state.monitor = MonitorService(settings, history_size=settings.history_size)

    from yggdrasil.node.api.routers import (
        analysis,
        audit,
        fs as fs_router,
        function,
        health,
        loki,
        market,
        messenger,
        pyfunc,
        tabular,
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(fs_router.router, prefix="/api/fs")
    app.include_router(tabular.router, prefix="/api/v2/tabular")
    app.include_router(analysis.router, prefix="/api/v2/analysis")
    app.include_router(market.router, prefix="/api/v2/market")
    app.include_router(messenger.router, prefix="/api")
    app.include_router(pyfunc.router, prefix="/api/v2")
    app.include_router(function.router, prefix="/api")
    app.include_router(audit.router, prefix="/api/v2")
    app.include_router(loki.router, prefix="/api/v2")

    return app
