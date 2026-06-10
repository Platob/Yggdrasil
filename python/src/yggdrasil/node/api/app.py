"""The lean v2 API — the hot-path read endpoints, mounted standalone.

:func:`create_api` builds a FastAPI app carrying just the v2 surface the
UI polls constantly (ping / stats / backend / health / audit / pyfunc /
pyenv). It owns a small set of in-memory services on ``app.state`` so it
can run on its own (the bench drives it without the full node app). The
full node app (:mod:`yggdrasil.node.app`) mounts the same routes plus the
mutating surfaces.
"""
from __future__ import annotations

import sys
import time

from fastapi import FastAPI

from yggdrasil import version
from yggdrasil.node.api.services.audit import AuditLog
from yggdrasil.node.config import Settings
from yggdrasil.node.remote import list_all as _remote_list
from yggdrasil.node.services.function import FunctionService
from yggdrasil.node.services.messenger import MessengerService


def create_api(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="ygg-node v2 API")
    started = time.monotonic()

    app.state.settings = settings
    app.state.audit = AuditLog(settings)
    app.state.functions = FunctionService(settings)
    app.state.messenger = MessengerService(settings)

    @app.get("/api/ping")
    async def ping() -> dict:
        return {"status": "ok", "ts": time.time()}

    @app.get("/api/v2/health")
    async def health() -> dict:
        return {"status": "healthy"}

    @app.get("/api/v2/stats")
    async def stats() -> dict:
        msgs = sum(len(c) for c in app.state.messenger._messages.values())
        return {
            "node_id": settings.node_id,
            "uptime_s": time.monotonic() - started,
            "messages": msgs,
            "functions": len(app.state.functions._by_id),
        }

    @app.get("/api/v2/backend")
    async def backend() -> dict:
        return {
            "backend": "ygg-node",
            "version": version.__version__,
            "node_id": settings.node_id,
            "python": sys.version.split()[0],
            "uptime_s": time.monotonic() - started,
        }

    @app.get("/api/v2/backend/summary")
    async def backend_summary() -> dict:
        return {"backend": "ygg-node", "version": version.__version__, "node_id": settings.node_id}

    @app.get("/api/v2/audit")
    async def audit(limit: int = 20) -> dict:
        return {"entries": [e.model_dump() for e in app.state.audit.recent(limit)]}

    @app.get("/api/v2/pyfunc")
    async def pyfunc() -> dict:
        return {"functions": _remote_list()}

    @app.get("/api/v2/pyenv")
    async def pyenv() -> dict:
        return {
            "python": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "executable": sys.executable,
            "platform": sys.platform,
            "ygg_version": version.__version__,
        }

    return app
