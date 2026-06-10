"""The full node app — every route, services wired onto ``app.state``.

Mounts the lean v2 read surface (:func:`create_api`'s endpoints) plus the
mutating surfaces: the remote-call endpoint (``/api/call``), the messenger,
the pyfunc registry, the confined filesystem, and the tabular/analysis
engine. Tabular results ride the Arrow IPC stream; everything else the
ygg-pickle wire (see :mod:`.transport`).
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from yggdrasil import version
from yggdrasil.exceptions.api import (
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    register_api_exception_handlers,
)
from yggdrasil.node import remote as _remote
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.audit import AuditLog
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.tabular import TabularService
from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.function import FunctionCreate
from yggdrasil.node.schemas.messenger import MessageSend
from yggdrasil.node.services.function import FunctionService
from yggdrasil.node.services.messenger import MessengerService
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    deserialize_pickle,
    is_tabular,
    serialize_result,
    to_arrow_table,
    write_arrow_stream,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        _app.state.audit.close()  # flush the pending audit tail to disk

    app = FastAPI(title="ygg-node", lifespan=lifespan)
    register_api_exception_handlers(app)
    started = time.monotonic()

    fs = FsService(settings)
    app.state.settings = settings
    app.state.audit = AuditLog(settings)
    app.state.functions = FunctionService(settings)
    app.state.messenger = MessengerService(settings)
    app.state.fs = fs
    app.state.analysis = AnalysisService(settings, fs=fs)
    app.state.tabular = TabularService(settings, fs=fs)

    # -- v2 read surface ---------------------------------------------------

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
        return {"functions": _remote.list_all()}

    @app.get("/api/v2/pyenv")
    async def pyenv() -> dict:
        return {
            "python": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "executable": sys.executable,
            "platform": sys.platform,
            "ygg_version": version.__version__,
        }

    # -- remote call -------------------------------------------------------

    @app.post("/api/call")
    async def call(request: Request) -> Response:
        if not settings.allow_remote:
            raise ForbiddenError("Remote calls are disabled (set allow_remote=True).")
        payload = deserialize_pickle(await request.body())
        name = payload.get("func")
        fn = _remote.get(name)
        if fn is None:
            known = ", ".join(_remote.list_all()) or "(none)"
            raise NotFoundError(f"No remote function {name!r}. Registered: {known}.")
        result = fn(*payload.get("args", ()), **payload.get("kwargs", {}))
        if is_tabular(result):
            return StreamingResponse(
                write_arrow_stream(to_arrow_table(result)),
                media_type=CONTENT_TYPE_ARROW_STREAM,
            )
        body, content_type = serialize_result(result)
        return Response(content=body, media_type=content_type)

    # -- messenger ---------------------------------------------------------

    @app.post("/api/messenger")
    async def send(msg: MessageSend) -> dict:
        out = await app.state.messenger.send_message(msg)
        return out.model_dump()

    @app.get("/api/messenger/channels")
    async def channels() -> dict:
        return {"channels": [c.model_dump() for c in await app.state.messenger.list_channels()]}

    @app.get("/api/messenger/channels/{name}/messages")
    async def messages(name: str, limit: int = 100) -> dict:
        msgs = await app.state.messenger.get_messages(name, limit=limit)
        return {"messages": [m.model_dump() for m in msgs], "channel": name, "total": len(msgs)}

    @app.post("/api/messenger/channels")
    async def create_channel(body: dict) -> dict:
        chan = await app.state.messenger.create_channel(body["name"])
        return chan.model_dump()

    # -- functions ---------------------------------------------------------

    @app.post("/api/function")
    async def fn_create(payload: FunctionCreate) -> dict:
        resp = await app.state.functions.create(payload)
        app.state.audit.log("create", "pyfunc", resp.function.id, detail=f"name={payload.name}")
        return resp.model_dump()

    @app.get("/api/function")
    async def fn_list() -> dict:
        return {"functions": [r.model_dump() for r in await app.state.functions.list()]}

    @app.get("/api/function/{fid}")
    async def fn_get(fid: int) -> dict:
        return (await app.state.functions.get(fid)).model_dump()

    @app.delete("/api/function/{fid}")
    async def fn_delete(fid: int) -> dict:
        await app.state.functions.delete(fid)
        app.state.audit.log("delete", "pyfunc", fid)
        return {"deleted": fid}

    # -- filesystem --------------------------------------------------------

    @app.get("/fs/ls")
    async def fs_ls(path: str = "") -> dict:
        return (await app.state.fs.ls(path)).model_dump()

    @app.get("/fs/read")
    async def fs_read(path: str) -> StreamingResponse:
        return StreamingResponse(app.state.fs.read(path), media_type="application/octet-stream")

    @app.post("/fs/mkdir")
    async def fs_mkdir(body: dict) -> dict:
        await app.state.fs.mkdir(body["path"])
        return {"ok": True}

    @app.get("/api/v2/tabular/inspect")
    async def tabular_inspect(path: str) -> dict:
        return (await app.state.tabular.inspect(path)).model_dump()

    return app
