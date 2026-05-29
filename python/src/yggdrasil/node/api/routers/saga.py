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
    OpLogResponse,
    ReplicateRequest,
    ReplicateResult,
    SchemaCreate,
    SchemaListResponse,
    SchemaResponse,
    SchemaUpdate,
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


@router.post("/discover", response_model=TableListResponse)
async def discover(req: DiscoverRequest, saga: SagaService = Depends(get_saga_service)):
    return await saga.discover(req)


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


@router.post("/explain", response_model=ExplainResult)
async def explain_sql(req: SqlRequest, saga: SagaService = Depends(get_saga_service)):
    return saga.explain(req)


@router.post("/sql.arrow")
async def run_sql_arrow(req: SqlRequest, saga: SagaService = Depends(get_saga_service)):
    """Execute and stream the result as an Arrow IPC stream (zero-copy wire,
    disk-spilled when heavy)."""
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
