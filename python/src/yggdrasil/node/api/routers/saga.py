"""Saga router — /api/v2/saga/

Catalog CRUD:
  GET/POST /api/v2/saga/catalog
  GET/POST /api/v2/saga/catalog/{catalog}/schema
  GET/POST /api/v2/saga/catalog/{catalog}/schema/{schema}/table

SQL engine:
  POST /api/v2/saga/sql           → SqlResult (JSON)
  POST /api/v2/saga/sql.arrow     → Arrow IPC stream (bytes)
  POST /api/v2/saga/explain       → plan dict
  POST /api/v2/saga/register      → (shortcut — catalog.schema.table in body)

Mounts:
  GET/POST /api/v2/saga/mount                → list / create
  GET      /api/v2/saga/mount/{alias}/ls     → MountLsResult

Network / cluster:
  POST /api/v2/network/register
  GET  /api/v2/network/peers

Replication (cluster bench):
  POST /api/v2/saga/replicate
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from yggdrasil.node.api.schemas.mount import MountCreate
from yggdrasil.node.api.schemas.saga import (
    CatalogCreate,
    ForecastRegisterRequest,
    SchemaCreate,
    SqlRequest,
    TableCreate,
)
from yggdrasil.node import transport

router = APIRouter(tags=["saga"])


def _svc(request: Request):
    return request.app.state.saga


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

@router.get("/api/v2/saga/catalog")
async def list_catalogs(node: str | None = None, svc=Depends(_svc)) -> dict:
    # ``node=`` query param is used by cluster bench to proxy to another node —
    # for now return local catalogs.
    cats = await svc.list_catalogs()
    return {"catalogs": [c.model_dump() for c in cats]}


@router.post("/api/v2/saga/catalog")
async def create_catalog(req: CatalogCreate, svc=Depends(_svc)) -> dict:
    return (await svc.create_catalog(req)).model_dump()


@router.get("/api/v2/saga/catalog/{catalog}/schema")
async def list_schemas(catalog: str, svc=Depends(_svc)) -> dict:
    schemas = await svc.list_schemas(catalog)
    return {"schemas": [s.model_dump() for s in schemas]}


@router.post("/api/v2/saga/catalog/{catalog}/schema")
async def create_schema(catalog: str, req: SchemaCreate, svc=Depends(_svc)) -> dict:
    return (await svc.create_schema(catalog, req)).model_dump()


@router.get("/api/v2/saga/catalog/{catalog}/schema/{schema}/table")
async def list_tables(catalog: str, schema: str, svc=Depends(_svc)) -> dict:
    tables = await svc.list_tables(catalog, schema)
    return {"tables": [t.model_dump() for t in tables]}


@router.post("/api/v2/saga/catalog/{catalog}/schema/{schema}/table")
async def create_table(catalog: str, schema: str, req: TableCreate, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.create_table(catalog, schema, req)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/v2/saga/catalog/{catalog}/schema/{schema}/table/{table}")
async def get_table(catalog: str, schema: str, table: str, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.get_table(catalog, schema, table)).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

@router.post("/api/v2/saga/sql")
async def execute_sql(req: SqlRequest, svc=Depends(_svc)) -> dict:
    try:
        result = await svc.execute_sql(req)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/v2/saga/sql.arrow")
async def execute_sql_arrow(req: SqlRequest, svc=Depends(_svc)) -> StreamingResponse:
    try:
        stream, cleanup = svc.execute_sql_arrow(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async def _gen():
        try:
            for chunk in stream:
                yield chunk
        finally:
            if cleanup:
                cleanup()

    return StreamingResponse(
        _gen(),
        media_type=transport.CONTENT_TYPE_ARROW_STREAM,
    )


@router.post("/api/v2/saga/explain")
async def explain_sql(req: SqlRequest, svc=Depends(_svc)) -> dict:
    try:
        return await svc.explain_sql(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/v2/saga/register")
async def register_table(body: dict, svc=Depends(_svc)) -> dict:
    """Shortcut: POST {source_url, catalog, schema, table} to register a table."""
    try:
        catalog = body.get("catalog", "main")
        schema = body.get("schema", "default")
        name = body.get("table", "")
        source = body.get("source_url", "")
        if not name or not source:
            raise HTTPException(status_code=400, detail="table and source_url required")
        # ensure catalog + schema exist
        from yggdrasil.node.api.schemas.saga import CatalogCreate, SchemaCreate, TableCreate
        await svc.create_catalog(CatalogCreate(name=catalog))
        await svc.create_schema(catalog, SchemaCreate(name=schema))
        result = await svc.create_table(catalog, schema, TableCreate(name=name, source_url=source))
        return result.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/v2/saga/replicate")
async def replicate(body: dict, svc=Depends(_svc)) -> dict:
    catalog = body.get("catalog", "main")
    schema = body.get("schema", "default")
    table = body.get("table", "")
    target = body.get("target", "")
    mode = body.get("mode", "metadata")
    try:
        return await svc.replicate(catalog, schema, table, target, mode)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# FORECAST asset
# ---------------------------------------------------------------------------

@router.post("/api/v2/saga/forecast")
async def register_forecast(req: ForecastRegisterRequest, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.register_forecast(req)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# mounts
# ---------------------------------------------------------------------------

@router.get("/api/v2/saga/mount")
async def list_mounts(svc=Depends(_svc)) -> dict:
    mounts = await svc.list_mounts()
    return {"mounts": [m.model_dump() for m in mounts]}


@router.post("/api/v2/saga/mount")
async def create_mount(req: MountCreate, svc=Depends(_svc)) -> dict:
    try:
        return (await svc.create_mount(req)).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/v2/saga/mount/{alias}/ls")
async def mount_ls(alias: str, path: str = "", svc=Depends(_svc)) -> dict:
    try:
        return (await svc.mount_ls(alias, path)).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# network / cluster
# ---------------------------------------------------------------------------

@router.post("/api/v2/network/register")
async def register_node(body: dict, svc=Depends(_svc)) -> dict:
    node_id = body.get("node_id", "")
    host = body.get("host", "127.0.0.1")
    port = int(body.get("port", 8100))
    return await svc.register_node(node_id, host, port)


@router.get("/api/v2/network/peers")
async def list_peers(svc=Depends(_svc)) -> dict:
    peers = await svc.list_peers()
    return {"peers": peers}
