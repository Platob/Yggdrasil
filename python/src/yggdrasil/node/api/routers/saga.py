"""Saga catalog HTTP surface — catalog/schema/table CRUD + SQL editor.

Reads and SQL accept ``?node=`` to proxy to a linked peer, so the UI can browse
any node's catalog and run a query where the data lives.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ... import transport
from ..deps import get_network_service, get_saga_service
from ..schemas.saga import (
    CatalogCreate,
    CatalogListResponse,
    CatalogResponse,
    CatalogUpdate,
    DiscoverRequest,
    ExplainResult,
    ForecastAssetResult,
    ForecastRegisterRequest,
    MaterializeResult,
    MountCreate,
    MountListing,
    MountListResponse,
    MountResponse,
    MountUpdate,
    OpLogResponse,
    SessionResult,
    WindowRequest,
    PlanEditRequest,
    PlanEditResult,
    PlanGraph,
    RegisterRequest,
    ActivityResponse,
    ReplicateRequest,
    ReplicateResult,
    SagaOverview,
    SearchResponse,
    SchemaCreate,
    SchemaListResponse,
    SchemaResponse,
    SchemaUpdate,
    SqlExportRequest,
    StagedResult,
    SqlRequest,
    SqlResult,
    TableCreate,
    TableListResponse,
    TablePayload,
    TableResponse,
    TableUpdate,
)
from ..services.network import NetworkService
from ..services.saga import SagaService

router = APIRouter(tags=["saga"])


async def _remote(network: NetworkService, saga: SagaService, node: str | None,
                  suffix: str, *, method: str = "GET", params=None, json_body=None):
    """Forward a /saga call to ``node`` when it names a peer; else return None."""
    if not node or node == saga.settings.node_id:
        return None
    return await network.proxy_json(node, method, f"/api/v2/saga{suffix}",
                                    params=params, json_body=json_body)


# -- catalogs ---------------------------------------------------------------

@router.get("/catalog", response_model=CatalogListResponse)
async def list_catalogs(node: str | None = None,
                        saga: SagaService = Depends(get_saga_service),
                        network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, "/catalog")
    return remote if remote is not None else await saga.list_catalogs()


@router.post("/catalog", response_model=CatalogResponse)
async def create_catalog(req: CatalogCreate, node: str | None = None,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, "/catalog", method="POST",
                           json_body=req.model_dump())
    return remote if remote is not None else await saga.create_catalog(req)


@router.get("/catalog/{name}", response_model=CatalogResponse)
async def get_catalog(name: str, node: str | None = None,
                      saga: SagaService = Depends(get_saga_service),
                      network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/catalog/{name}")
    return remote if remote is not None else await saga.get_catalog(name)


@router.patch("/catalog/{name}", response_model=CatalogResponse)
async def update_catalog(name: str, req: CatalogUpdate,
                         saga: SagaService = Depends(get_saga_service)):
    return await saga.update_catalog(name, req)


@router.delete("/catalog/{name}", response_model=CatalogResponse)
async def delete_catalog(name: str, cascade: bool = False,
                         saga: SagaService = Depends(get_saga_service)):
    return await saga.delete_catalog(name, cascade=cascade)


# -- schemas ----------------------------------------------------------------

@router.get("/catalog/{catalog}/schema", response_model=SchemaListResponse)
async def list_schemas(catalog: str, node: str | None = None,
                       saga: SagaService = Depends(get_saga_service),
                       network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/catalog/{catalog}/schema")
    return remote if remote is not None else await saga.list_schemas(catalog)


@router.post("/catalog/{catalog}/schema", response_model=SchemaResponse)
async def create_schema(catalog: str, req: SchemaCreate,
                        saga: SagaService = Depends(get_saga_service)):
    return await saga.create_schema(catalog, req)


@router.get("/catalog/{catalog}/schema/{name}", response_model=SchemaResponse)
async def get_schema(catalog: str, name: str,
                     saga: SagaService = Depends(get_saga_service)):
    return await saga.get_schema(catalog, name)


@router.patch("/catalog/{catalog}/schema/{name}", response_model=SchemaResponse)
async def update_schema(catalog: str, name: str, req: SchemaUpdate,
                        saga: SagaService = Depends(get_saga_service)):
    return await saga.update_schema(catalog, name, req)


@router.delete("/catalog/{catalog}/schema/{name}", response_model=SchemaResponse)
async def delete_schema(catalog: str, name: str, cascade: bool = False,
                        saga: SagaService = Depends(get_saga_service)):
    return await saga.delete_schema(catalog, name, cascade=cascade)


# -- tables -----------------------------------------------------------------

@router.get("/catalog/{catalog}/schema/{schema}/table", response_model=TableListResponse)
async def list_tables(catalog: str, schema: str, node: str | None = None,
                      saga: SagaService = Depends(get_saga_service),
                      network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/catalog/{catalog}/schema/{schema}/table")
    return remote if remote is not None else await saga.list_tables(catalog, schema)


@router.post("/catalog/{catalog}/schema/{schema}/table", response_model=TableResponse)
async def create_table(catalog: str, schema: str, req: TableCreate,
                       saga: SagaService = Depends(get_saga_service)):
    return await saga.create_table(catalog, schema, req)


@router.get("/catalog/{catalog}/schema/{schema}/table/{name}", response_model=TableResponse)
async def get_table(catalog: str, schema: str, name: str, node: str | None = None,
                    saga: SagaService = Depends(get_saga_service),
                    network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/catalog/{catalog}/schema/{schema}/table/{name}")
    return remote if remote is not None else await saga.get_table(catalog, schema, name)


@router.patch("/catalog/{catalog}/schema/{schema}/table/{name}", response_model=TableResponse)
async def update_table(catalog: str, schema: str, name: str, req: TableUpdate,
                       saga: SagaService = Depends(get_saga_service)):
    return await saga.update_table(catalog, schema, name, req)


@router.delete("/catalog/{catalog}/schema/{schema}/table/{name}", response_model=TableResponse)
async def delete_table(catalog: str, schema: str, name: str,
                       saga: SagaService = Depends(get_saga_service)):
    return await saga.delete_table(catalog, schema, name)


@router.post("/catalog/{catalog}/schema/{schema}/table/{name}/refresh", response_model=TableResponse)
async def refresh_table(catalog: str, schema: str, name: str,
                        saga: SagaService = Depends(get_saga_service)):
    entry = await saga.refresh_table(catalog, schema, name)
    return TableResponse(table=entry)


@router.get("/catalog/{catalog}/schema/{schema}/table/{name}/log", response_model=OpLogResponse)
async def read_table_log(catalog: str, schema: str, name: str, limit: int = 200,
                         node: str | None = None,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node,
                           f"/catalog/{catalog}/schema/{schema}/table/{name}/log",
                           params={"limit": limit})
    return remote if remote is not None else await saga.read_log(catalog, schema, name, limit=limit)


@router.post("/register", response_model=TableResponse)
async def register(req: RegisterRequest, saga: SagaService = Depends(get_saga_service)):
    """One-shot: ensure catalog + schema, infer the name, register + profile."""
    return await saga.register(req)


# -- mounts (named aliases over path objects) -------------------------------

@router.get("/mount", response_model=MountListResponse)
async def list_mounts(node: str | None = None,
                      saga: SagaService = Depends(get_saga_service),
                      network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, "/mount")
    return remote if remote is not None else await saga.list_mounts()


@router.post("/mount", response_model=MountResponse)
async def create_mount(req: MountCreate, node: str | None = None,
                       saga: SagaService = Depends(get_saga_service),
                       network: NetworkService = Depends(get_network_service)):
    """Register (upsert by alias) a mount: a named base path/URL that the SQL
    engine and file browser expand on demand — e.g. a Databricks volume, an S3
    prefix, or a remote node path made queryable under one short name."""
    remote = await _remote(network, saga, node, "/mount", method="POST", json_body=req.model_dump())
    return remote if remote is not None else await saga.create_mount(req)


@router.get("/mount/{name}", response_model=MountResponse)
async def get_mount(name: str, node: str | None = None,
                    saga: SagaService = Depends(get_saga_service),
                    network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/mount/{name}")
    return remote if remote is not None else MountResponse(mount=await saga.get_mount(name))


@router.patch("/mount/{name}", response_model=MountResponse)
async def update_mount(name: str, req: MountUpdate, node: str | None = None,
                       saga: SagaService = Depends(get_saga_service),
                       network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/mount/{name}", method="PATCH", json_body=req.model_dump())
    return remote if remote is not None else await saga.update_mount(name, req)


@router.delete("/mount/{name}", response_model=MountResponse)
async def delete_mount(name: str, node: str | None = None,
                       saga: SagaService = Depends(get_saga_service),
                       network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, f"/mount/{name}", method="DELETE")
    return remote if remote is not None else await saga.delete_mount(name)


@router.get("/mount/{name}/ls", response_model=MountListing)
async def list_mount_dir(name: str, subpath: str = "", node: str | None = None,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    """Lazily browse a mount through the Path layer — tabular files are flagged
    so the UI can query/preview them inline."""
    remote = await _remote(network, saga, node, f"/mount/{name}/ls", params={"subpath": subpath})
    return remote if remote is not None else await saga.list_mount(name, subpath)


@router.get("/overview", response_model=SagaOverview)
async def overview(node: str | None = None,
                   saga: SagaService = Depends(get_saga_service),
                   network: NetworkService = Depends(get_network_service)):
    """Catalog-wide monitoring rollup: counts by kind, totals, activity feed,
    and largest/busiest leaderboards — drives the Saga management dashboard."""
    remote = await _remote(network, saga, node, "/overview")
    return remote if remote is not None else await saga.overview()


@router.get("/search", response_model=SearchResponse)
async def search(q: str = "", limit: int = 50, node: str | None = None,
                 saga: SagaService = Depends(get_saga_service),
                 network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node, "/search", params={"q": q, "limit": limit})
    return remote if remote is not None else await saga.search(q, limit=limit)


@router.get("/catalog/{catalog}/schema/{schema}/table/{name}/activity", response_model=ActivityResponse)
async def table_activity(catalog: str, schema: str, name: str, node: str | None = None,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    remote = await _remote(network, saga, node,
                           f"/catalog/{catalog}/schema/{schema}/table/{name}/activity")
    return remote if remote is not None else await saga.activity(catalog, schema, name)


@router.post("/discover", response_model=TableListResponse)
async def discover(req: DiscoverRequest, saga: SagaService = Depends(get_saga_service)):
    return await saga.discover(req)


# -- forecast workflows -----------------------------------------------------

@router.post("/forecast", response_model=ForecastAssetResult)
async def register_forecast(req: ForecastRegisterRequest, node: str | None = None,
                            saga: SagaService = Depends(get_saga_service),
                            network: NetworkService = Depends(get_network_service)):
    """Register (upsert) a forecasting workflow as a FORECAST catalog asset — a
    persisted spec that resolves to a history+forecast view the SQL engine can
    query like a table. `materialize` snapshots it to a managed parquet."""
    if node and node != saga.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/saga/forecast",
                                        json_body=req.model_dump(by_alias=True))
    return await saga.register_forecast(req)


@router.post("/catalog/{catalog}/schema/{schema}/table/{name}/forecast/refresh",
             response_model=ForecastAssetResult)
async def refresh_forecast(catalog: str, schema: str, name: str, node: str | None = None,
                           saga: SagaService = Depends(get_saga_service),
                           network: NetworkService = Depends(get_network_service)):
    """Recompute a forecast workflow (rewrites its materialised snapshot)."""
    remote = await _remote(network, saga, node,
                           f"/catalog/{catalog}/schema/{schema}/table/{name}/forecast/refresh",
                           method="POST")
    return remote if remote is not None else await saga.refresh_forecast(catalog, schema, name)


# -- replication ------------------------------------------------------------

@router.post("/import", response_model=TableResponse)
async def import_table(payload: TablePayload, saga: SagaService = Depends(get_saga_service)):
    """Register a table pushed from a peer (the replication receive side)."""
    return await saga.import_payload(payload)


@router.post("/replicate", response_model=ReplicateResult)
async def replicate(req: ReplicateRequest, saga: SagaService = Depends(get_saga_service)):
    return await saga.replicate(req)


# -- SQL editor -------------------------------------------------------------

@router.post("/sql", response_model=SqlResult)
async def run_sql(req: SqlRequest,
                  saga: SagaService = Depends(get_saga_service),
                  network: NetworkService = Depends(get_network_service)):
    target = saga.compute_node(req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None  # peer runs it locally
        try:
            return await network.proxy_json(target, "POST", "/api/v2/saga/sql", json_body=body)
        except Exception:
            # Connection to the chosen node failed — fall back to running here
            # (works when the fs is shared); otherwise the local run surfaces a
            # clear "table lives on node X" error.
            req = req.model_copy(update={"node": None})
    return await saga.execute_sql(req)


_EXPORT_MEDIA = {
    "csv": "text/csv", "parquet": "application/vnd.apache.parquet",
    "json": "application/json", "ndjson": "application/x-ndjson",
    "arrow": "application/vnd.apache.arrow.stream",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@router.post("/sql.export")
async def export_sql(req: SqlExportRequest,
                     saga: SagaService = Depends(get_saga_service),
                     network: NetworkService = Depends(get_network_service)):
    """Run the query and download the full result in any handled media type
    (csv/parquet/json/ndjson/arrow/xlsx/tsv). Runs where the data lives."""
    media = _EXPORT_MEDIA.get((req.fmt or "csv").lower(), "application/octet-stream")
    sql_req = SqlRequest(sql=req.sql, dialect=req.dialect, catalog=req.catalog,
                         schema=req.schema_, node=req.node)
    target = saga.compute_node(sql_req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None
        return StreamingResponse(
            network.proxy_post_stream(target, "/api/v2/saga/sql.export", body),
            media_type=media)

    tmp_path, name = await saga.export_sql(req)

    def _gen():
        try:
            yield from transport.iter_file_chunks(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        _gen(), media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}"',
                 "X-Accel-Buffering": "no"})


@router.post("/sql.stage", response_model=StagedResult)
async def stage_sql(req: SqlRequest, saga: SagaService = Depends(get_saga_service),
                    network: NetworkService = Depends(get_network_service)):
    """Run where the data lives and write the Arrow result to ``staging_path``."""
    target = saga.compute_node(req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None
        return await network.proxy_json(target, "POST", "/api/v2/saga/sql.stage", json_body=body)
    return await saga.stage_result(req)


@router.post("/sql.session", response_model=SessionResult)
async def sql_session(req: SqlRequest,
                      saga: SagaService = Depends(get_saga_service),
                      network: NetworkService = Depends(get_network_service)):
    """Stage a query result to an Arrow IPC file for lazy windowed scrolling.
    Runs where the data lives; the result's ``node_id`` tells the client which
    node to read windows from / close the session on."""
    target = saga.compute_node(req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None
        return await network.proxy_json(target, "POST", "/api/v2/saga/sql.session", json_body=body)
    return await saga.sql_session(req)


@router.post("/session/window")
async def session_window(req: WindowRequest,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    """A lazily filtered/sorted/exploded slice of a staged session as Arrow IPC.
    ``?node=`` (or req.node) reads the window from the node that staged it."""
    if req.node and req.node != saga.settings.node_id:
        body = req.model_dump()
        body["node"] = None
        return StreamingResponse(
            network.proxy_post_stream(req.node, "/api/v2/saga/session/window", body),
            media_type=transport.CONTENT_TYPE_ARROW_STREAM)
    data, rows, has_more = await saga.window(req)

    return StreamingResponse(
        iter((data,)),
        media_type=transport.CONTENT_TYPE_ARROW_STREAM,
        headers={"X-Window-Rows": str(rows), "X-Has-More": "1" if has_more else "0",
                 "Access-Control-Expose-Headers": "X-Window-Rows, X-Has-More"})


@router.post("/session/close")
async def session_close(path: str, node: str | None = None,
                        saga: SagaService = Depends(get_saga_service),
                        network: NetworkService = Depends(get_network_service)):
    """Clear a staged session file (called when the viewer closes/disconnects)."""
    if node and node != saga.settings.node_id:
        return await network.proxy_json(node, "POST", f"/api/v2/saga/session/close?path={path}")
    return {"closed": saga.close_session(path)}


@router.post("/sql.materialize", response_model=MaterializeResult)
async def materialize_sql(req: SqlRequest,
                         saga: SagaService = Depends(get_saga_service),
                         network: NetworkService = Depends(get_network_service)):
    """Run the query once and write it to a tmp parquet, returning a node path.
    Drives the shared /tabular + /analysis surfaces over a SQL result."""
    target = saga.compute_node(req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None
        return await network.proxy_json(target, "POST", "/api/v2/saga/sql.materialize", json_body=body)
    return await saga.materialize_sql(req)


@router.post("/explain", response_model=ExplainResult)
async def explain_sql(req: SqlRequest, saga: SagaService = Depends(get_saga_service)):
    return saga.explain(req)


@router.post("/plan", response_model=PlanGraph)
async def plan(req: SqlRequest, analyze: bool = False,
               saga: SagaService = Depends(get_saga_service)):
    """Structured execution-plan DAG. With ``?analyze=true`` it runs the query
    in staged prefixes and fills per-operation rows + elapsed times."""
    return await saga.analyze_plan(req) if analyze else saga.build_plan(req)


@router.post("/plan/edit", response_model=PlanEditResult)
async def plan_edit(req: PlanEditRequest, saga: SagaService = Depends(get_saga_service)):
    """Apply structural edits (set limit, drop filter/order/…) and re-emit SQL."""
    return saga.edit_plan(req)


@router.post("/sql.arrow")
async def run_sql_arrow(req: SqlRequest,
                        saga: SagaService = Depends(get_saga_service),
                        network: NetworkService = Depends(get_network_service)):
    """Execute and stream the result as an Arrow IPC stream (zero-copy wire,
    disk-spilled when heavy). Runs where the data lives."""
    target = saga.compute_node(req)
    if target:
        body = req.model_dump(by_alias=True)
        body["node"] = None
        return StreamingResponse(
            network.proxy_post_stream(target, "/api/v2/saga/sql.arrow", body),
            media_type=transport.CONTENT_TYPE_ARROW_STREAM)
    stream, cleanup = saga.execute_sql_arrow(req)

    def _gen():
        try:
            yield from stream
        finally:
            if cleanup:
                cleanup()

    return StreamingResponse(
        _gen(),
        media_type=transport.CONTENT_TYPE_ARROW_STREAM,
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
