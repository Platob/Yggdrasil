"""FastAPI sub-application factory for the v2 API.

``create_api()`` builds a FastAPI app pre-loaded with the v2 routers.
``_make_state(settings)`` constructs the shared service instances that are
attached to ``app.state`` so all route handlers reach them via ``Request``.

This is the layer the ``bench_v2_endpoints`` benchmark targets directly.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

from yggdrasil.node.api.routers import analysis, fs, saga, tabular
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.audit import AuditLog
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.saga import SagaService
from yggdrasil.node.api.services.tabular import TabularService
from yggdrasil.node.config import Settings


def create_api(settings: Settings | None = None) -> FastAPI:
    """Build and return the v2 FastAPI sub-application."""
    if settings is None:
        settings = Settings()

    app = FastAPI(title="yggdrasil node API v2", version="2.0.0")
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # shared service instances on app.state
    fs_svc = FsService(settings)
    tabular_svc = TabularService(settings, fs=fs_svc)
    analysis_svc = AnalysisService(settings, fs=fs_svc)
    saga_svc = SagaService(settings)
    audit = AuditLog(settings)

    app.state.settings = settings
    app.state.fs = fs_svc
    app.state.tabular = tabular_svc
    app.state.analysis = analysis_svc
    app.state.saga = saga_svc
    app.state.audit = audit

    # v2 routers
    app.include_router(fs.router)
    app.include_router(tabular.router)
    app.include_router(analysis.router)
    app.include_router(saga.router)

    # -- thin system endpoints -----------------------------------------------
    @app.get("/api/ping")
    def ping():
        return {"status": "ok", "node_id": settings.node_id}

    @app.get("/api/v2/health")
    def health():
        return {"healthy": True, "node_id": settings.node_id}

    @app.get("/api/v2/stats")
    def stats():
        return {
            "node_id": settings.node_id,
            "node_home": str(settings.node_home),
            "uptime_s": round(time.time() - _START_TS, 2),
        }

    @app.get("/api/v2/backend")
    def backend():
        return {
            "node_id": settings.node_id,
            "python": platform.python_version(),
            "os": platform.system(),
        }

    @app.get("/api/v2/backend/summary")
    def backend_summary():
        return {
            "node_id": settings.node_id,
            "platform": f"Python {platform.python_version()} / {platform.system()}",
        }

    @app.get("/api/v2/audit")
    def audit_get(limit: int = 20):
        return {"entries": [e.model_dump() for e in audit.get_entries(limit)]}

    @app.get("/api/v2/pyfunc")
    def pyfunc_list():
        from yggdrasil.node.remote import list_remotes
        return {"functions": list_remotes()}

    @app.get("/api/v2/pyenv")
    def pyenv_list():
        return {"envs": []}

    return app


_START_TS: float = time.time()
