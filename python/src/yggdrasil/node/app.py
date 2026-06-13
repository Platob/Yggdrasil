"""The YGG node FastAPI app.

``create_app(settings)`` mounts every route group: discovery (/api/ping,
/api/hello), the v2 surface (health/stats/backend/audit/fs/tabular/analysis),
chat, functions, monitor, the pickle /api/call endpoint, saga mounts, and the
market WebSocket streams used for trading demos.
"""
from __future__ import annotations

import asyncio
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from .config import Settings
from .remote import call_remote, get_registry
from .transport import (
    CONTENT_TYPE_ARROW_STREAM,
    deserialize_pickle,
    serialize_pickle,
    serialize_result,
)

from .api.schemas.analysis import (
    AggregateRequest,
    FinanceRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
)
from .api.services.analysis import AnalysisService
from .api.services.audit import AuditLog
from .api.services.fs import FsService
from .api.services.tabular import TabularService
from .schemas.function import FunctionCreate
from .schemas.messenger import MessageSend
from .services.function import FunctionService
from .services.messenger import MessengerService
from .services.monitor import MonitorService

from yggdrasil.version import __version__


@dataclass
class SagaMount:
    alias: str
    target: str
    kind: str  # "local", "s3", "database", "node"
    read_only: bool = True
    comment: str = ""


_START_TIME = time.time()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    settings.node_home.mkdir(parents=True, exist_ok=True)

    fs = FsService(settings)
    tabular = TabularService(settings, fs)
    analysis = AnalysisService(settings, fs)
    audit = AuditLog(settings)
    messenger = MessengerService(settings)
    monitor = MonitorService(settings)
    functions = FunctionService(settings)
    mounts: list[SagaMount] = []
    symbols: dict[str, dict] = {
        "BTCUSD": {"symbol": "BTCUSD", "last": 60000.0},
        "ETHUSD": {"symbol": "ETHUSD", "last": 3000.0},
        "SPY": {"symbol": "SPY", "last": 500.0},
    }

    app = FastAPI(title="YGG Node", version=__version__)
    app.state.settings = settings

    # -- discovery ---------------------------------------------------------
    @app.get("/api/ping")
    async def ping() -> dict:
        return {"pong": True}

    @app.get("/api/hello")
    async def hello() -> dict:
        return {"node_id": settings.node_id, "version": __version__}

    @app.get("/api/hello/peers")
    async def peers() -> dict:
        return {"peers": []}

    # -- v2 health / stats / backend --------------------------------------
    @app.get("/api/v2/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v2/stats")
    async def stats() -> dict:
        snap = monitor.snapshot()
        return {
            "cpu_pct": snap.cpu_pct,
            "mem_pct": snap.mem_pct,
            "mem_mb": snap.mem_mb,
            "load_1m": snap.load_1m,
            "uptime_s": time.time() - _START_TIME,
            "node_id": settings.node_id,
        }

    @app.get("/api/v2/backend")
    async def backend() -> dict:
        return {
            "node_id": settings.node_id,
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "capabilities": {
                "fs": True,
                "tabular": True,
                "analysis": True,
                "messenger": True,
                "functions": True,
                "remote": settings.allow_remote,
                "websocket": True,
            },
        }

    @app.get("/api/v2/backend/summary")
    async def backend_summary() -> dict:
        return {"node_id": settings.node_id, "version": __version__, "status": "ok"}

    @app.get("/api/v2/audit")
    async def get_audit(limit: int = Query(100)) -> dict:
        return {"entries": audit.entries(limit=limit)}

    @app.get("/api/v2/pyfunc")
    async def pyfunc() -> dict:
        return {"functions": sorted(get_registry().keys())}

    @app.get("/api/v2/pyenv")
    async def pyenv() -> dict:
        return {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
        }

    # -- v2 filesystem -----------------------------------------------------
    @app.get("/api/v2/fs/nodes")
    async def fs_nodes() -> dict:
        return {
            "nodes": [{"node_id": settings.node_id, "home": str(settings.node_home)}],
            "mounts": [],
        }

    @app.get("/api/v2/fs/ls")
    async def fs_ls(path: str = Query(""), offset: int = 0, limit: int = 0):
        try:
            return await fs.ls(path, offset=offset, limit=limit)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.get("/api/v2/fs/read")
    async def fs_read(path: str = Query(...)):
        try:
            return await fs.read(path)
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    # -- v2 tabular --------------------------------------------------------
    @app.get("/api/v2/tabular/inspect")
    async def tabular_inspect(path: str = Query(...)):
        try:
            result = await tabular.inspect(path)
            return JSONResponse(result.model_dump())
        except (FileNotFoundError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    # -- v2 analysis -------------------------------------------------------
    @app.post("/api/v2/analysis/aggregate")
    async def analysis_aggregate(req: AggregateRequest):
        return await analysis.aggregate(req)

    @app.post("/api/v2/analysis/series")
    async def analysis_series(req: SeriesRequest):
        return await analysis.series(req)

    @app.post("/api/v2/analysis/ohlc")
    async def analysis_ohlc(req: OhlcRequest):
        return await analysis.ohlc(req)

    @app.post("/api/v2/analysis/pivot")
    async def analysis_pivot(req: PivotRequest):
        return await analysis.pivot(req)

    @app.post("/api/v2/analysis/forecast")
    async def analysis_forecast(req: ForecastRequest):
        return await analysis.forecast(req)

    @app.post("/api/v2/analysis/finance")
    async def analysis_finance(req: FinanceRequest):
        return await analysis.finance(req)

    # -- monitor -----------------------------------------------------------
    @app.get("/api/monitor")
    async def get_monitor() -> dict:
        snap = monitor.snapshot()
        return {
            "ts": snap.ts,
            "cpu_pct": snap.cpu_pct,
            "mem_pct": snap.mem_pct,
            "mem_mb": snap.mem_mb,
            "load_1m": snap.load_1m,
        }

    # -- functions ---------------------------------------------------------
    @app.get("/api/function")
    async def list_functions() -> dict:
        fns = await functions.list()
        return {"functions": [f.model_dump() for f in fns]}

    @app.post("/api/function")
    async def create_function(req: FunctionCreate):
        resp = await functions.create(req)
        audit.log("create", "pyfunc", resp.function.id, detail=f"name={resp.function.name}")
        return resp

    @app.get("/api/function/{fid}")
    async def get_function(fid: int):
        try:
            return await functions.get(fid)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.delete("/api/function/{fid}")
    async def delete_function(fid: int) -> dict:
        await functions.delete(fid)
        audit.log("delete", "pyfunc", fid)
        return {"deleted": fid}

    @app.post("/api/function/{fid}/run")
    async def run_function(fid: int):
        try:
            run = await functions.run(fid)
            return {"run": run.model_dump()}
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    @app.get("/api/run/{run_id}")
    async def get_run(run_id: int):
        try:
            run = await functions.get_run(run_id)
            return {"run": run.model_dump()}
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    # -- messenger ---------------------------------------------------------
    @app.get("/api/messenger/channels")
    async def messenger_channels() -> dict:
        chans = await messenger.list_channels()
        return {"channels": [c.model_dump() for c in chans]}

    @app.post("/api/messenger")
    async def messenger_send(msg: MessageSend):
        sent = await messenger.send_message(msg)
        return sent

    @app.get("/api/messenger/channels/{name}/messages")
    async def messenger_messages(name: str, limit: int = 50):
        try:
            msgs = await messenger.get_messages(name, limit=limit)
            return {"messages": [m.model_dump() for m in msgs]}
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    # -- remote call (pickle wire) ----------------------------------------
    @app.post("/api/call")
    async def call(request: Request):
        if not settings.allow_remote:
            return JSONResponse({"error": "remote calls disabled"}, status_code=403)
        body = await request.body()
        payload = deserialize_pickle(body)
        try:
            result = call_remote(payload["func"], payload.get("args", ()), payload.get("kwargs", {}))
        except Exception as exc:  # surface the error to the caller over pickle
            data = serialize_pickle({"error": type(exc).__name__, "message": str(exc)})
            return Response(content=data, media_type="application/octet-stream", status_code=500)
        data, content_type = serialize_result(result)
        return Response(content=data, media_type=content_type)

    # -- saga mounts -------------------------------------------------------
    @app.get("/api/v2/saga/mount")
    async def list_mounts() -> dict:
        return {"mounts": [m.__dict__ for m in mounts]}

    @app.post("/api/v2/saga/mount")
    async def register_mount(mount: dict) -> dict:
        m = SagaMount(
            alias=mount["alias"],
            target=mount["target"],
            kind=mount.get("kind", "local"),
            read_only=mount.get("read_only", True),
            comment=mount.get("comment", ""),
        )
        mounts.append(m)
        return {"mount": m.__dict__}

    # -- market (trading demo) --------------------------------------------
    @app.get("/api/v2/market/symbols")
    async def market_symbols() -> dict:
        return {"symbols": list(symbols.values())}

    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            t = 0
            while True:
                t += 1
                tick = {
                    "t": t,
                    "ts": time.time(),
                    "value": 100.0 + 5.0 * math.sin(t / 5.0) + random.uniform(-1, 1),
                }
                await ws.send_json(tick)
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    @app.websocket("/api/v2/market/tick")
    async def market_tick(ws: WebSocket) -> None:
        await ws.accept()
        sym = "BTCUSD"
        last = symbols[sym]["last"]
        try:
            while True:
                drift = random.uniform(-0.002, 0.002)
                open_ = last
                last = max(0.01, last * (1.0 + drift))
                high = max(open_, last) * (1.0 + random.uniform(0, 0.001))
                low = min(open_, last) * (1.0 - random.uniform(0, 0.001))
                symbols[sym]["last"] = last
                await ws.send_json({
                    "symbol": sym,
                    "ts": time.time(),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": last,
                    "volume": random.uniform(0.5, 5.0),
                })
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app
