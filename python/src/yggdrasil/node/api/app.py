"""FastAPI app for the yggdrasil node.

One ``Settings`` is built in :func:`create_api` and every service is wired
against it and stashed on ``app.state`` so handlers stay thin — they unpack the
request, call the service, and return the model. Two transports cross the wire:
JSON for control-plane endpoints, and the pickle/Arrow transport for ``/api/call``
(remote functions) and ``/api/v2/saga/sql.arrow`` (streamed tabular results).
"""
from __future__ import annotations

import sys
import time

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .. import remote as remote_mod
from ..config import Settings
from .. import transport
from ..services.function import FunctionService
from ..services.messenger import MessengerService
from ..services.monitor import MonitorService
from .schemas.analysis import ForecastRequest
from .schemas.saga import (
    CatalogCreate,
    ForecastRegisterRequest,
    SchemaCreate,
    SqlRequest,
    TableCreate,
)
from .services.analysis import AnalysisService
from .services.audit import AuditLog
from .services.fs import FsService
from .services.saga import SagaService
from .services.tabular import TabularService


def create_api(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="yggdrasil node", version="2.0")
    app.add_middleware(GZipMiddleware, minimum_size=512)

    st = app.state
    st.settings = settings
    st.started = time.time()
    st.requests = 0
    st.audit = AuditLog(settings)
    st.messenger = MessengerService(settings)
    st.functions = FunctionService(settings)
    st.monitor = MonitorService(settings)
    st.fs = FsService(settings)
    st.tabular = TabularService(settings, st.fs)
    st.analysis = AnalysisService(settings, st.fs)
    st.saga = SagaService(settings)
    st.peers = {}

    @app.middleware("http")
    async def _count(request: Request, call_next):
        app.state.requests += 1
        return await call_next(request)

    # -- discovery ---------------------------------------------------------

    @app.get("/api/ping")
    def ping():
        return {"status": "ok", "ts": time.time()}

    @app.get("/api/hello")
    def hello():
        return {"node_id": settings.node_id, "version": "2.0"}

    @app.get("/api/hello/peers")
    def hello_peers():
        return {"peers": list(st.peers.values())}

    # -- v2 control plane --------------------------------------------------

    @app.get("/api/v2/stats")
    def stats():
        return {"uptime_s": time.time() - st.started, "requests": st.requests}

    @app.get("/api/v2/backend")
    def backend():
        return _detect_backends()

    @app.get("/api/v2/backend/summary")
    def backend_summary():
        backends = _detect_backends()
        online = sum(1 for b in backends if b["status"] == "online")
        return {"count": len(backends), "online": online}

    @app.get("/api/v2/health")
    def health():
        return {"status": "healthy"}

    @app.get("/api/v2/audit")
    def audit(limit: int = Query(20, ge=1, le=1000)):
        return {"entries": st.audit.recent(limit)}

    @app.get("/api/v2/pyfunc")
    def pyfunc():
        return {"functions": sorted(remote_mod._REGISTRY)}

    @app.get("/api/v2/pyenv")
    def pyenv():
        import importlib.metadata as md
        pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
        return {"python": sys.version, "packages": pkgs}

    # -- remote call (pickle transport) ------------------------------------

    @app.post("/api/call")
    async def call(request: Request):
        body = await request.body()
        payload = transport.deserialize_pickle(body)
        func = payload.get("func") or payload.get("func_name")
        result = remote_mod.call(func, payload.get("args"), payload.get("kwargs"))
        data, content_type = transport.serialize_result(result)
        return Response(content=data, media_type=content_type)

    # -- monitor -----------------------------------------------------------

    @app.get("/api/monitor")
    def monitor():
        return st.monitor.snapshot()

    # -- messenger ---------------------------------------------------------

    @app.post("/api/messenger")
    async def messenger_send(payload: dict):
        from ..schemas.messenger import MessageSend
        msg = await st.messenger.send_message(MessageSend(**payload))
        return msg.model_dump()

    @app.get("/api/messenger/channels")
    async def messenger_channels():
        chans = await st.messenger.list_channels()
        return {"channels": [c.model_dump() for c in chans]}

    @app.get("/api/messenger/channels/{channel}/messages")
    async def messenger_messages(channel: str, limit: int = 50):
        msgs = await st.messenger.get_messages(channel, limit=limit)
        return {"messages": [m.model_dump() for m in msgs]}

    # -- functions ---------------------------------------------------------

    @app.get("/api/function")
    async def function_list():
        funcs = await st.functions.list()
        return {"functions": [f.model_dump() for f in funcs]}

    @app.post("/api/function")
    async def function_create(payload: dict):
        from ..schemas.function import FunctionCreate
        resp = await st.functions.create(FunctionCreate(**payload))
        st.audit.log("create", "function", resp.function.id, detail=f"name={resp.function.name}")
        return resp.model_dump()

    @app.get("/api/function/{func_id}")
    async def function_get(func_id: str):
        return (await st.functions.get(func_id)).model_dump()

    @app.delete("/api/function/{func_id}")
    async def function_delete(func_id: str):
        await st.functions.delete(func_id)
        st.audit.log("delete", "function", func_id)
        return {"deleted": func_id}

    @app.post("/api/function/{func_id}/run")
    async def function_run(func_id: str):
        resp = await st.functions.run(func_id)
        st.audit.log("run", "function", func_id, detail=f"status={resp.run.status}")
        return resp.model_dump()

    @app.get("/api/run/{run_id}")
    async def run_get(run_id: str):
        return (await st.functions.get_run(run_id)).model_dump()

    # -- analysis ----------------------------------------------------------

    @app.get("/api/v2/analysis/finance")
    async def analysis_finance(path: str, column: str, ts_column: str | None = None):
        return await st.analysis.finance(path, column, ts_column)

    @app.post("/api/v2/analysis/forecast")
    async def analysis_forecast(payload: dict):
        return (await st.analysis.forecast(ForecastRequest(**payload))).model_dump()

    # -- fs ----------------------------------------------------------------

    @app.get("/api/v2/fs/nodes")
    def fs_nodes():
        return {"root": str(settings.node_home), "node_id": settings.node_id, "mounts": []}

    @app.get("/api/v2/fs/ls")
    async def fs_ls(path: str = "", offset: int = 0, limit: int | None = None):
        return (await st.fs.ls(path, offset=offset, limit=limit)).model_dump()

    @app.get("/api/v2/tabular/inspect")
    async def tabular_inspect(path: str):
        return (await st.tabular.inspect(path)).model_dump()

    # -- saga --------------------------------------------------------------

    @app.post("/api/v2/saga/catalog")
    async def saga_create_catalog(req: CatalogCreate):
        return await st.saga.create_catalog(req)

    @app.get("/api/v2/saga/catalog")
    async def saga_list_catalogs(node: str | None = None):
        return {"catalogs": await st.saga.list_catalogs(), "node_id": settings.node_id}

    @app.post("/api/v2/saga/catalog/{catalog}/schema")
    async def saga_create_schema(catalog: str, req: SchemaCreate):
        return await st.saga.create_schema(catalog, req)

    @app.get("/api/v2/saga/catalog/{catalog}/schema/{schema}/table")
    async def saga_list_tables(catalog: str, schema: str):
        return {"tables": await st.saga.list_tables(catalog, schema)}

    @app.post("/api/v2/saga/catalog/{catalog}/schema/{schema}/table")
    async def saga_create_table(catalog: str, schema: str, req: TableCreate):
        resp = await st.saga.create_table(catalog, schema, req)
        st.audit.log("create", "table", f"{catalog}.{schema}.{req.name}")
        return resp.model_dump()

    @app.get("/api/v2/saga/catalog/{catalog}/schema/{schema}/table/{table}")
    async def saga_get_table(catalog: str, schema: str, table: str):
        return (await st.saga.get_table(catalog, schema, table)).model_dump()

    @app.post("/api/v2/saga/sql")
    async def saga_sql(req: SqlRequest):
        return (await st.saga.execute_sql(req)).model_dump()

    @app.post("/api/v2/saga/sql.arrow")
    async def saga_sql_arrow(req: SqlRequest):
        stream, cleanup = st.saga.execute_sql_arrow(req)

        def _gen():
            try:
                yield from stream
            finally:
                if cleanup:
                    cleanup()

        return StreamingResponse(_gen(), media_type=transport.CONTENT_TYPE_ARROW_STREAM)

    @app.post("/api/v2/saga/explain")
    async def saga_explain(req: SqlRequest):
        return st.saga.explain(req)

    @app.post("/api/v2/saga/register")
    async def saga_register(payload: dict):
        # Flat convenience registration used by the cluster bench.
        await st.saga.create_catalog(CatalogCreate(name=payload["catalog"]))
        await st.saga.create_schema(payload["catalog"], SchemaCreate(name=payload["schema"]))
        resp = await st.saga.create_table(
            payload["catalog"], payload["schema"],
            TableCreate(name=payload["table"], source_url=payload["source_url"]))
        return resp.model_dump()

    @app.post("/api/v2/saga/replicate")
    async def saga_replicate(payload: dict):
        # Metadata-only "replication" in single-node mode: report the bytes a
        # data copy would move so the cluster bench has a number to print.
        info = await st.saga.get_table(payload["catalog"], payload["schema"], payload["table"])
        return {"mode": payload.get("mode", "metadata"),
                "bytes_copied": info.table.statistics.size_bytes,
                "target": payload.get("target")}

    @app.post("/api/v2/saga/forecast")
    async def saga_forecast(req: ForecastRegisterRequest):
        return (await st.saga.register_forecast(req)).model_dump()

    @app.post("/api/v2/saga/mount")
    async def saga_mount_create(payload: dict):
        alias = payload.get("alias") or payload.get("name")
        st.peers.setdefault("_mounts", {})
        return {"alias": alias, "target": payload.get("target")}

    @app.get("/api/v2/saga/mount")
    def saga_mount_list():
        return {"mounts": []}

    @app.get("/api/v2/saga/mount/{alias}/ls")
    async def saga_mount_ls(alias: str, path: str = ""):
        return (await st.fs.ls(path)).model_dump()

    # -- network (cluster bench stubs) -------------------------------------

    @app.post("/api/v2/network/register")
    async def network_register(payload: dict):
        nid = payload["node_id"]
        st.peers[nid] = {"node_id": nid, "host": payload.get("host"), "port": payload.get("port")}
        return {"registered": nid}

    @app.exception_handler(KeyError)
    async def _key_error(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"error": str(exc.args[0] if exc.args else exc)})

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return app


def _detect_backends() -> list[dict]:
    """Probe optional backends; never raises (offline detection)."""
    backends = []
    for name, mod in [("polars", "polars"), ("pyarrow", "pyarrow"),
                      ("duckdb", "duckdb"), ("databricks", "databricks.sdk")]:
        try:
            __import__(mod)
            backends.append({"name": name, "status": "online"})
        except Exception:
            backends.append({"name": name, "status": "offline"})
    return backends
