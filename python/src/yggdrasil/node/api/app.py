"""v2 API app factory.

:func:`create_api` builds the analysis/v2 FastAPI app using a default
:class:`Settings`. Request counting is done in a middleware so ``/v2/stats``
reports a real number; shared state (start time, audit log, function
registry, env list) lives on ``app.state``.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings
from ..routers import v2 as v2_router
from ..services.function import FunctionService
from .services.audit import AuditLog


def create_api(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="yggdrasil.node.api", version="2.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.started = time.time()
    app.state.requests = 0
    app.state.audit = AuditLog(settings)
    app.state.function = FunctionService(settings)
    app.state.envs = ["default"]

    @app.middleware("http")
    async def _count_requests(request: Request, call_next):
        request.app.state.requests += 1
        return await call_next(request)

    app.include_router(v2_router.router)

    return app
