"""yggdrasil.node.app — full node application factory.

``create_app(settings)`` composes:

- The v2 API sub-app (``create_api``) for /api/v2/* + /api/ping
- ``/api/messenger*``   — in-memory message bus
- ``/api/function*``    — stored Python function registry
- ``/api/monitor``      — system metrics
- ``/api/call``         — @remote function dispatch (pickle in, Arrow/pickle out)
- ``/api/hello``        — node discovery
- ``/api/v2/saga/``     — all saga catalog + SQL + mount endpoints

``app`` is the importable application object for uvicorn::

    uvicorn yggdrasil.node.app:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response

from yggdrasil.node.api.app import create_api
from yggdrasil.node.config import Settings
from yggdrasil.node.services.function import FunctionService
from yggdrasil.node.services.messenger import MessengerService
from yggdrasil.node.services.monitor import MonitorService
from yggdrasil.node.schemas.function import FunctionCreate
from yggdrasil.node.schemas.messenger import MessageSend
from yggdrasil.node import transport


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the full yggdrasil node application."""
    if settings is None:
        settings = Settings()

    # Start with the v2 API sub-app.
    app = create_api(settings)
    app.title = "yggdrasil node"
    app.version = "1.0.0"

    # additional services
    messenger_svc = MessengerService(settings)
    function_svc = FunctionService(settings)
    monitor_svc = MonitorService(settings)

    app.state.messenger = messenger_svc
    app.state.function = function_svc
    app.state.monitor = monitor_svc

    # ------------------------------------------------------------------
    # /api/ping
    # ------------------------------------------------------------------
    @app.get("/api/ping")
    def ping():
        return {"ok": True, "node_id": settings.node_id}

    # ------------------------------------------------------------------
    # /api/hello  (node discovery — bench_node_integration)
    # ------------------------------------------------------------------
    @app.get("/api/hello")
    def hello():
        return {"node_id": settings.node_id, "version": "1.0.0"}

    @app.get("/api/hello/peers")
    async def hello_peers(request: Request):
        peers = await request.app.state.saga.list_peers()
        return {"node_id": settings.node_id, "peers": peers}

    # ------------------------------------------------------------------
    # /api/monitor
    # ------------------------------------------------------------------
    @app.get("/api/monitor")
    def monitor():
        return monitor_svc.snapshot().model_dump()

    # ------------------------------------------------------------------
    # /api/messenger*
    # ------------------------------------------------------------------
    @app.post("/api/messenger")
    async def messenger_send(body: dict):
        msg = MessageSend(**body)
        result = await messenger_svc.send_message(msg)
        return {"message": result.model_dump()}

    @app.get("/api/messenger/channels")
    async def messenger_channels():
        channels = await messenger_svc.list_channels()
        return {"channels": [c.model_dump() for c in channels]}

    @app.post("/api/messenger/channels")
    async def messenger_create_channel(body: dict):
        name = body.get("name", "general")
        ch = await messenger_svc.create_channel(name)
        return {"channel": ch.model_dump()}

    @app.get("/api/messenger/channels/{channel}/messages")
    async def messenger_messages(channel: str, limit: int = 50):
        msgs = await messenger_svc.get_messages(channel, limit=limit)
        return {"messages": [m.model_dump() for m in msgs]}

    # ------------------------------------------------------------------
    # /api/function*  (legacy v1 endpoints — bench_node_integration)
    # ------------------------------------------------------------------
    @app.get("/api/function")
    async def function_list():
        fns = await function_svc.list()
        return {"functions": [f.model_dump() for f in fns]}

    @app.post("/api/function")
    async def function_create(body: dict):
        req = FunctionCreate(**body)
        resp = await function_svc.create(req)
        return {"function": resp.function.model_dump()}

    @app.get("/api/function/{function_id}")
    async def function_get(function_id: int):
        try:
            resp = await function_svc.get(function_id)
            return {"function": resp.function.model_dump()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.delete("/api/function/{function_id}")
    async def function_delete(function_id: int):
        await function_svc.delete(function_id)
        return {"ok": True}

    @app.post("/api/function/{function_id}/run")
    async def function_run(function_id: int):
        try:
            run = await function_svc.run(function_id)
            return {"run": run.model_dump()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/run/{run_id}")
    async def run_get(run_id: int):
        try:
            run = await function_svc.get_run(run_id)
            return {"run": run.model_dump()}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # /api/call  — @remote function dispatch
    # ------------------------------------------------------------------
    @app.post("/api/call")
    async def call(request: Request):
        ct = request.headers.get("content-type", "")
        raw = await request.body()
        try:
            payload = transport.deserialize_pickle(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"bad payload: {exc}")

        from yggdrasil.node.remote import get_remote
        func_name = payload.get("func", "")
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})
        try:
            fn = get_remote(func_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

        data, content_type = transport.serialize_result(result)
        return Response(content=data, media_type=content_type)

    return app


# ---------------------------------------------------------------------------
# Importable app object for uvicorn
# ---------------------------------------------------------------------------
_settings = Settings()
app = create_app(_settings)
