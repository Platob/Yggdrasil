"""Saga — distributed data catalog service.

A metadata layer: catalogs/schemas/tables are JSON records persisted in the
managed store at ``~/.saga/{node_id}/store.json`` (off the network filesystem);
the table *data* stays on the shared filesystem. SQL runs through the repo's
existing plan engine (``parse_sql`` → ``execute_plan``) over registered tables
and raw file URLs, and results come back as Arrow IPC — streamed from memory
when small, spilled to ``tmp`` and streamed off disk when heavy, or written to
a remote staging NodePath. Tables replicate (metadata or data) to peers, and
every mutation/query is logged per-asset on compressed Arrow IPC.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import tempfile
import time
from functools import partial
from threading import Lock
from typing import Any

import pyarrow as pa
from fastapi.concurrency import run_in_threadpool

from yggdrasil.data.options import CastOptions
from yggdrasil.enums.dialect import Dialect
from yggdrasil.io.tabular.base import Tabular, is_tabular_source
from yggdrasil.plan.execute import execute_plan
from yggdrasil.plan.nodes import InsertNode, MergeNode, PlanNode, SelectNode
from yggdrasil.plan.ops import (
    JoinClause,
    LateralViewItem,
    SubqueryRef,
    TableRef,
    ValuesRef,
)
from yggdrasil.plan.sql_parser import parse_sql

from ...config import Settings
from ... import scratch, transport
from yggdrasil.exceptions.api import BadRequestError, ConflictError, NotFoundError
from ..schemas.saga import (
    CatalogCreate,
    CatalogEntry,
    CatalogListResponse,
    CatalogResponse,
    CatalogUpdate,
    ColumnSpec,
    ColumnStat,
    DiscoverRequest,
    ExplainResult,
    ForecastAssetResult,
    ForecastRegisterRequest,
    ForecastSpec,
    SchemaCreate,
    SchemaEntry,
    SchemaListResponse,
    SchemaResponse,
    SchemaUpdate,
    ActivityResponse,
    OpLogEntry,
    OpLogResponse,
    PlanEditRequest,
    PlanEditResult,
    PlanGraph,
    PlanOp,
    MaterializeResult,
    RegisterRequest,
    ReplicateRequest,
    SessionResult,
    ReplicateResult,
    SearchHit,
    SearchResponse,
    StagedResult,
    SqlColumn,
    SqlRequest,
    SqlResult,
    TableCreate,
    TableEntry,
    TableListResponse,
    TablePayload,
    TableResponse,
    TableStatistics,
    TableUpdate,
)
from .saga_log import OpLog
from ...ids import make_static_id as make_id

_TABULAR_EXTS = {"parquet", "pq", "csv", "tsv", "ndjson", "json", "arrow", "feather",
                 "ipc", "xlsx", "zip", "gz", "orc", "delta"}
# Formats the result-export writer can emit (every one has a tabular encoder).
_EXPORT_FMTS = {"csv", "parquet", "json", "ndjson", "arrow", "xlsx"}
_DIALECTS = {d.value for d in Dialect}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(v: Any) -> Any:
    """A scalar safe to drop into a JSON response (NaN/inf → None, exotic → str)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    return str(v)


class SagaService:
    """Catalog/schema/table registry + SQL executor for one node."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._root = settings.saga_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = self._root / "store.json"
        self._lock = Lock()
        self._catalogs: dict[int, CatalogEntry] = {}
        self._schemas: dict[int, SchemaEntry] = {}
        self._tables: dict[int, TableEntry] = {}
        self._cat_idx: dict[str, int] = {}
        self._sch_idx: dict[str, int] = {}
        self._tbl_idx: dict[str, int] = {}
        self._log = OpLog(settings.saga_log_root)
        self._network = None  # bound after construction; enables replication
        try:
            from yggdrasil.environ.userinfo import UserInfo
            self._user = UserInfo.current().key
        except Exception:
            self._user = ""
        self._load()

    def bind_network(self, network) -> None:
        """Wire the peer mesh in (set after construction to avoid a cycle)."""
        self._network = network

    def _record(self, asset: str, op: str, *, statement: str = "",
                rows: int | None = None, detail: str = "") -> None:
        try:
            self._log.append(asset, op, user=self._user, node=self.settings.node_id,
                             statement=statement, rows=rows, detail=detail)
        except Exception:
            pass  # logging must never break the operation

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._store.exists():
            return
        try:
            doc = json.loads(self._store.read_text())
        except (OSError, ValueError):
            return
        for c in doc.get("catalogs", []):
            try:
                e = CatalogEntry.model_validate(c)
                self._catalogs[e.id] = e
            except Exception:
                continue
        for s in doc.get("schemas", []):
            try:
                e = SchemaEntry.model_validate(s)
                self._schemas[e.id] = e
            except Exception:
                continue
        for t in doc.get("tables", []):
            try:
                e = TableEntry.model_validate(t)
                self._tables[e.id] = e
            except Exception:
                continue
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the name → id indexes. Cheap, and called from the one place
        that mutates state (``_save``), so every lookup stays O(1) without
        scattering index bookkeeping across each create/delete."""
        self._cat_idx = {c.name: c.id for c in self._catalogs.values()}
        self._sch_idx = {f"{s.catalog}.{s.name}": s.id for s in self._schemas.values()}
        self._tbl_idx = {t.full_name: t.id for t in self._tables.values()}

    def _save(self) -> None:
        """Reindex, then atomic-write: dump to a temp file, then rename."""
        self._reindex()
        doc = {
            "catalogs": [c.model_dump() for c in self._catalogs.values()],
            "schemas": [s.model_dump() for s in self._schemas.values()],
            "tables": [t.model_dump(by_alias=True) for t in self._tables.values()],
        }
        fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(doc, fh)
            os.replace(tmp, self._store)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- internal lookups (assume lock held) --------------------------------

    def _catalog_by_name(self, name: str) -> CatalogEntry | None:
        cid = self._cat_idx.get(name)
        return self._catalogs.get(cid) if cid is not None else None

    def _schema_by_name(self, catalog: str, name: str) -> SchemaEntry | None:
        sid = self._sch_idx.get(f"{catalog}.{name}")
        return self._schemas.get(sid) if sid is not None else None

    def _table_by_name(self, catalog: str, schema: str, name: str) -> TableEntry | None:
        return self._table_by_full(f"{catalog}.{schema}.{name}")

    def _table_by_full(self, full_name: str) -> TableEntry | None:
        tid = self._tbl_idx.get(full_name)
        return self._tables.get(tid) if tid is not None else None

    def _require_catalog(self, name: str) -> CatalogEntry:
        c = self._catalog_by_name(name)
        if c is None:
            raise NotFoundError(f"Catalog {name!r} not found")
        return c

    def _require_schema(self, catalog: str, name: str) -> SchemaEntry:
        s = self._schema_by_name(catalog, name)
        if s is None:
            raise NotFoundError(f"Schema {catalog}.{name!r} not found")
        return s

    def _enrich_catalog(self, c: CatalogEntry) -> CatalogEntry:
        n = sum(1 for s in self._schemas.values() if s.catalog == c.name)
        return c.model_copy(update={"schema_count": n})

    def _enrich_schema(self, s: SchemaEntry) -> SchemaEntry:
        n = sum(1 for t in self._tables.values()
                if t.catalog == s.catalog and t.schema_name == s.name)
        return s.model_copy(update={"table_count": n})

    # -- catalog CRUD -------------------------------------------------------

    async def create_catalog(self, req: CatalogCreate) -> CatalogResponse:
        with self._lock:
            existing = self._catalog_by_name(req.name)
            now = _now()
            if existing:
                upd = {"updated_at": now}
                if req.comment:
                    upd["comment"] = req.comment
                if req.owner:
                    upd["owner"] = req.owner
                if req.dialect:
                    upd["dialect"] = req.dialect
                if req.storage_location is not None:
                    upd["storage_location"] = req.storage_location
                if req.properties:
                    upd["properties"] = {**existing.properties, **req.properties}
                entry = existing.model_copy(update=upd)
                self._catalogs[entry.id] = entry
                self._save()
                return CatalogResponse(catalog=self._enrich_catalog(entry))
            cid = make_id(req.name)
            entry = CatalogEntry(
                id=cid,
                name=req.name,
                comment=req.comment,
                owner=req.owner,
                dialect=(req.dialect or self.settings.saga_default_dialect),
                storage_location=(req.storage_location or f"saga/{req.name}"),
                node_id=self.settings.node_id,
                properties=dict(req.properties),
                created_at=now,
                updated_at=now,
            )
            self._catalogs[cid] = entry
            self._save()
            return CatalogResponse(catalog=self._enrich_catalog(entry))

    async def list_catalogs(self) -> CatalogListResponse:
        with self._lock:
            items = [self._enrich_catalog(c) for c in self._catalogs.values()]
        items.sort(key=lambda c: c.name)
        return CatalogListResponse(node_id=self.settings.node_id, catalogs=items)

    async def get_catalog(self, name: str) -> CatalogResponse:
        with self._lock:
            return CatalogResponse(catalog=self._enrich_catalog(self._require_catalog(name)))

    async def update_catalog(self, name: str, req: CatalogUpdate) -> CatalogResponse:
        with self._lock:
            c = self._require_catalog(name)
            upd: dict[str, Any] = {"updated_at": _now()}
            for f in ("comment", "owner", "dialect", "storage_location"):
                v = getattr(req, f)
                if v is not None:
                    upd[f] = v
            if req.properties is not None:
                upd["properties"] = dict(req.properties)
            entry = c.model_copy(update=upd)
            self._catalogs[entry.id] = entry
            self._save()
            return CatalogResponse(catalog=self._enrich_catalog(entry))

    async def delete_catalog(self, name: str, *, cascade: bool = False) -> CatalogResponse:
        with self._lock:
            c = self._require_catalog(name)
            child_schemas = [s for s in self._schemas.values() if s.catalog == name]
            child_tables = [t for t in self._tables.values() if t.catalog == name]
            if (child_schemas or child_tables) and not cascade:
                raise ConflictError(
                    f"Catalog {name!r} is not empty ({len(child_schemas)} schemas, "
                    f"{len(child_tables)} tables). Pass cascade=true to drop it."
                )
            for s in child_schemas:
                self._schemas.pop(s.id, None)
            for t in child_tables:
                self._tables.pop(t.id, None)
            self._catalogs.pop(c.id, None)
            self._save()
            return CatalogResponse(catalog=c)

    # -- schema CRUD --------------------------------------------------------

    async def create_schema(self, catalog: str, req: SchemaCreate) -> SchemaResponse:
        with self._lock:
            self._require_catalog(catalog)
            now = _now()
            existing = self._schema_by_name(catalog, req.name)
            if existing:
                upd: dict[str, Any] = {"updated_at": now}
                if req.comment:
                    upd["comment"] = req.comment
                if req.properties:
                    upd["properties"] = {**existing.properties, **req.properties}
                entry = existing.model_copy(update=upd)
                self._schemas[entry.id] = entry
                self._save()
                return SchemaResponse(schema=self._enrich_schema(entry))
            full = f"{catalog}.{req.name}"
            sid = make_id(full)
            entry = SchemaEntry(
                id=sid, catalog=catalog, name=req.name, full_name=full,
                comment=req.comment, properties=dict(req.properties),
                created_at=now, updated_at=now,
            )
            self._schemas[sid] = entry
            self._save()
            return SchemaResponse(schema=self._enrich_schema(entry))

    async def list_schemas(self, catalog: str) -> SchemaListResponse:
        with self._lock:
            self._require_catalog(catalog)
            items = [self._enrich_schema(s) for s in self._schemas.values()
                     if s.catalog == catalog]
        items.sort(key=lambda s: s.name)
        return SchemaListResponse(node_id=self.settings.node_id, catalog=catalog, schemas=items)

    async def get_schema(self, catalog: str, name: str) -> SchemaResponse:
        with self._lock:
            return SchemaResponse(schema=self._enrich_schema(self._require_schema(catalog, name)))

    async def update_schema(self, catalog: str, name: str, req: SchemaUpdate) -> SchemaResponse:
        with self._lock:
            s = self._require_schema(catalog, name)
            upd: dict[str, Any] = {"updated_at": _now()}
            if req.comment is not None:
                upd["comment"] = req.comment
            if req.properties is not None:
                upd["properties"] = dict(req.properties)
            entry = s.model_copy(update=upd)
            self._schemas[entry.id] = entry
            self._save()
            return SchemaResponse(schema=self._enrich_schema(entry))

    async def delete_schema(self, catalog: str, name: str, *, cascade: bool = False) -> SchemaResponse:
        with self._lock:
            s = self._require_schema(catalog, name)
            child_tables = [t for t in self._tables.values()
                            if t.catalog == catalog and t.schema_name == name]
            if child_tables and not cascade:
                raise ConflictError(
                    f"Schema {catalog}.{name!r} has {len(child_tables)} tables. "
                    "Pass cascade=true to drop it."
                )
            for t in child_tables:
                self._tables.pop(t.id, None)
            self._schemas.pop(s.id, None)
            self._save()
            return SchemaResponse(schema=s)

    # -- table CRUD ---------------------------------------------------------

    async def create_table(self, catalog: str, schema: str, req: TableCreate) -> TableResponse:
        with self._lock:
            self._require_schema(catalog, schema)
            now = _now()
            full = f"{catalog}.{schema}.{req.name}"
            fmt = req.format or (req.source_url.rsplit(".", 1)[-1].lower() if "." in req.source_url else "")
            existing = self._table_by_name(catalog, schema, req.name)
            base = existing or TableEntry(
                id=make_id(full), catalog=catalog, schema=schema, name=req.name,
                full_name=full, created_at=now, updated_at=now,
            )
            entry = base.model_copy(update={
                "source_url": req.source_url,
                "object_type": req.object_type,
                "definition": req.definition or base.definition,
                "node": req.node,
                "table_type": req.table_type,
                "format": fmt,
                "comment": req.comment or base.comment,
                "columns": list(req.columns) if req.columns else base.columns,
                "properties": {**base.properties, **req.properties},
                "updated_at": now,
            })
            self._tables[entry.id] = entry
            self._save()
        self._record(full, "update" if existing else "register",
                     detail=f"{entry.object_type} {entry.source_url or entry.definition[:40]}")
        # Infer outside the lock — only file-backed TABLEs scan; a VIEW's schema
        # comes from running its definition, others carry declared columns.
        if req.infer and not req.node and entry.object_type in ("TABLE", "VIEW") and (req.source_url or req.definition):
            try:
                entry = await self.refresh_table(catalog, schema, req.name)
                return TableResponse(table=entry)
            except Exception:
                pass
        return TableResponse(table=entry)

    async def register(self, req: RegisterRequest) -> TableResponse:
        """One call to land a file in the catalog: ensure the catalog + schema
        exist, infer the table name from the filename, register and profile it.

        This is the easy path the Files page and Excel use — no need to create
        the catalog/schema first."""
        from pathlib import PurePosixPath
        name = req.table or PurePosixPath(req.source_url.split("://")[-1]).stem or "table"
        with self._lock:
            need_cat = self._catalog_by_name(req.catalog) is None
            need_sch = self._schema_by_name(req.catalog, req.schema_) is None
        if need_cat:
            await self.create_catalog(CatalogCreate(name=req.catalog, dialect=req.dialect))
        if need_sch:
            await self.create_schema(req.catalog, SchemaCreate(name=req.schema_))
        return await self.create_table(req.catalog, req.schema_, TableCreate(
            name=name, source_url=req.source_url, node=req.node,
            object_type=req.object_type, definition=req.definition,
            table_type=req.table_type, infer=True,
        ))

    # -- forecast workflows -------------------------------------------------

    async def register_forecast(self, req: ForecastRegisterRequest) -> ForecastAssetResult:
        """Upsert a FORECAST asset and (optionally) materialise it.

        The spec rides the asset's ``definition`` so it persists in store.json
        and replicates like any other asset. Materialising runs the forecast
        once and writes a managed parquet snapshot under the Saga data root; the
        asset then resolves from the snapshot until refreshed (a `live=false`
        optimisation), otherwise it recomputes on every query."""
        # Ensure catalog + schema exist (the easy path, like register()).
        with self._lock:
            need_cat = self._catalog_by_name(req.catalog) is None
            need_sch = self._schema_by_name(req.catalog, req.schema_) is None
        if need_cat:
            await self.create_catalog(CatalogCreate(name=req.catalog))
        if need_sch:
            await self.create_schema(req.catalog, SchemaCreate(name=req.schema_))
        return await run_in_threadpool(partial(self._register_forecast, req))

    def _register_forecast(self, req: ForecastRegisterRequest) -> ForecastAssetResult:
        from .forecast import forecast_frame
        spec = req.spec.model_copy(update={"materialized": req.materialize or req.spec.materialized})
        now = _now()
        full = f"{req.catalog}.{req.schema_}.{req.name}"

        df = self._forecast_source_frame(spec, req.catalog, req.schema_, set())
        out, used, rmse = forecast_frame(
            df, value=spec.column, x=spec.x, keys=list(spec.keys),
            horizon=spec.horizon, model=spec.model, period=spec.period, agg=spec.agg)

        materialized_url: str | None = None
        if spec.materialized:
            self.settings.saga_data_root.mkdir(parents=True, exist_ok=True)
            snap = self.settings.saga_data_root / f"forecast_{make_id(full)}.parquet"
            out.write_parquet(str(snap))
            materialized_url = str(snap)

        cols = [ColumnSpec(name=f.name, dtype=str(f.type), nullable=True)
                for f in out.to_arrow().schema]
        props = {"materialized_url": materialized_url} if materialized_url else {}
        props.update({"model_used": used, "horizon": str(spec.horizon),
                      "materialized_at": now if materialized_url else ""})
        with self._lock:
            self._require_schema(req.catalog, req.schema_)
            existing = self._table_by_name(req.catalog, req.schema_, req.name)
            base = existing or TableEntry(
                id=make_id(full), catalog=req.catalog, schema=req.schema_, name=req.name,
                full_name=full, created_at=now, updated_at=now)
            entry = base.model_copy(update={
                "object_type": "FORECAST",
                "definition": spec.model_dump_json(),
                "node": req.node,
                "source_url": materialized_url or "",
                "comment": req.comment or base.comment,
                "columns": cols,
                "statistics": TableStatistics(row_count=out.height, computed_at=now),
                "properties": {**base.properties, **props},
                "updated_at": now,
            })
            self._tables[entry.id] = entry
            self._save()
        self._record(full, "update" if existing else "register",
                     detail=f"FORECAST {spec.column}~{spec.x or 'idx'} h{spec.horizon} ({used})")
        return ForecastAssetResult(
            node_id=self.settings.node_id, table=entry, model_used=used,
            rmse=round(rmse, 4) if rmse is not None else None, rows=out.height,
            materialized_url=materialized_url,
            sampled=bool(spec.keys) and out.height > 0,
        )

    async def refresh_forecast(self, catalog: str, schema: str, name: str) -> ForecastAssetResult:
        """Recompute a materialised forecast snapshot (or refresh a live one's
        column/stat metadata)."""
        with self._lock:
            t = self._table_by_name(catalog, schema, name)
            if t is None:
                raise NotFoundError(f"Forecast {catalog}.{schema}.{name!r} not found")
            if t.object_type != "FORECAST":
                raise BadRequestError(f"{t.full_name!r} is not a FORECAST asset")
            spec = ForecastSpec.model_validate_json(t.definition)
        return await self.register_forecast(ForecastRegisterRequest(
            catalog=catalog, schema=schema, name=name, spec=spec, node=t.node,
            comment=t.comment, materialize=bool(spec.materialized)))

    async def search(self, q: str, *, limit: int = 50) -> "SearchResponse":
        """Substring search across catalogs/schemas/tables (assets)."""
        ql = (q or "").strip().lower()
        hits: list[SearchHit] = []
        with self._lock:
            cats = list(self._catalogs.values())
            schs = list(self._schemas.values())
            tbls = list(self._tables.values())
        for c in cats:
            if not ql or ql in c.name.lower() or ql in c.comment.lower():
                hits.append(SearchHit(kind="catalog", name=c.name, full_name=c.name, comment=c.comment))
        for s in schs:
            if not ql or ql in s.full_name.lower() or ql in s.comment.lower():
                hits.append(SearchHit(kind="schema", name=s.name, full_name=s.full_name,
                                      catalog=s.catalog, comment=s.comment))
        for t in tbls:
            if not ql or ql in t.full_name.lower() or ql in t.comment.lower():
                hits.append(SearchHit(kind="table", name=t.name, full_name=t.full_name,
                                      object_type=t.object_type, catalog=t.catalog,
                                      schema=t.schema_name, comment=t.comment))
        hits.sort(key=lambda h: (h.kind != "table", h.full_name))
        total = len(hits)
        return SearchResponse(node_id=self.settings.node_id, query=q,
                              hits=hits[:limit], total=total, truncated=total > limit)

    async def activity(self, catalog: str, schema: str, name: str) -> "ActivityResponse":
        """Op-log rollup for an asset — the monitoring dashboard feed."""
        full = f"{catalog}.{schema}.{name}"
        table = await run_in_threadpool(partial(self._log.read, full, limit=2000))
        rows = table.to_pylist()
        counts: dict[str, int] = {}
        per_day: dict[str, int] = {}
        for r in rows:
            counts[r["op"]] = counts.get(r["op"], 0) + 1
            day = (r["ts"] or "")[:10]
            if day:
                per_day[day] = per_day.get(day, 0) + 1
        days = sorted(per_day)[-14:]
        recent = [OpLogEntry(**{k: _json_safe(v) for k, v in rows[i].items()})
                  for i in range(min(20, len(rows)))]
        return ActivityResponse(
            node_id=self.settings.node_id, asset=full, op_counts=counts,
            total_ops=len(rows), last_op_at=(rows[0]["ts"] if rows else None),
            daily=[per_day[d] for d in days], recent=recent,
        )

    async def list_tables(self, catalog: str, schema: str) -> TableListResponse:
        with self._lock:
            self._require_schema(catalog, schema)
            items = [t for t in self._tables.values()
                     if t.catalog == catalog and t.schema_name == schema]
        items.sort(key=lambda t: t.name)
        return TableListResponse(
            node_id=self.settings.node_id, catalog=catalog, schema=schema, tables=items,
        )

    async def get_table(self, catalog: str, schema: str, name: str) -> TableResponse:
        with self._lock:
            t = self._table_by_name(catalog, schema, name)
            if t is None:
                raise NotFoundError(f"Table {catalog}.{schema}.{name!r} not found")
            return TableResponse(table=t)

    async def update_table(self, catalog: str, schema: str, name: str, req: TableUpdate) -> TableResponse:
        with self._lock:
            t = self._table_by_name(catalog, schema, name)
            if t is None:
                raise NotFoundError(f"Table {catalog}.{schema}.{name!r} not found")
            upd: dict[str, Any] = {"updated_at": _now()}
            for f in ("source_url", "object_type", "definition", "node", "table_type", "comment"):
                v = getattr(req, f)
                if v is not None:
                    upd[f] = v
            if req.format is not None:
                upd["format"] = req.format
            if req.columns is not None:
                upd["columns"] = list(req.columns)
            if req.properties is not None:
                upd["properties"] = dict(req.properties)
            entry = t.model_copy(update=upd)
            self._tables[entry.id] = entry
            self._save()
            return TableResponse(table=entry)

    async def delete_table(self, catalog: str, schema: str, name: str) -> TableResponse:
        with self._lock:
            t = self._table_by_name(catalog, schema, name)
            if t is None:
                raise NotFoundError(f"Table {catalog}.{schema}.{name!r} not found")
            self._tables.pop(t.id, None)
            self._save()
        # Dropping the table purges its operation log — no orphan history.
        self._log.purge(t.full_name)
        return TableResponse(table=t)

    async def read_log(self, catalog: str, schema: str, name: str, *, limit: int = 200) -> OpLogResponse:
        full = f"{catalog}.{schema}.{name}"
        table = await run_in_threadpool(partial(self._log.read, full, limit=limit))
        entries = [OpLogEntry(**{k: _json_safe(v) for k, v in row.items()})
                   for row in table.to_pylist()]
        return OpLogResponse(node_id=self.settings.node_id, asset=full, entries=entries)

    # -- statistics / schema inference --------------------------------------

    def _resolve_path(self, source_url: str) -> str:
        """Map a stored source_url to something Tabular.from_ can open.

        URLs (anything with ``://``) pass through. Relative paths are resolved
        against the node home — the same rooting the ``/fs`` and ``/tabular``
        APIs use — so a file's browser path, its ``source_url`` and its preview
        path are all the same string (no translation needed by the UI).
        """
        if "://" in source_url:
            return source_url
        from pathlib import Path as _P
        p = _P(source_url)
        if p.is_absolute():
            return str(p)
        root = self.settings.node_home
        resolved = (root / source_url).resolve()
        if not str(resolved).startswith(str(root.resolve())):
            raise BadRequestError("source_url escapes the node home")
        return str(resolved)

    async def refresh_table(self, catalog: str, schema: str, name: str) -> TableEntry:
        with self._lock:
            t = self._table_by_name(catalog, schema, name)
            if t is None:
                raise NotFoundError(f"Table {catalog}.{schema}.{name!r} not found")
            is_view = t.object_type == "VIEW"
            source = None if is_view else self._resolve_path(t.source_url)
        if is_view:
            cols, stats = await run_in_threadpool(self._infer_view, t)
        else:
            cols, stats = await run_in_threadpool(self._infer, source)
        with self._lock:
            cur = self._tables.get(t.id, t)
            entry = cur.model_copy(update={
                "columns": cols or cur.columns,
                "statistics": stats,
                "updated_at": _now(),
            })
            self._tables[entry.id] = entry
            self._save()
            return entry

    def _infer_view(self, view: TableEntry) -> tuple[list[ColumnSpec], TableStatistics]:
        """Profile a view by running its definition and reading the result schema."""
        tab = self._resolve_view(view, set())
        schema = tab.collect_schema()
        cols = [ColumnSpec(name=f.name, dtype=str(f.dtype), nullable=f.nullable)
                for f in schema.fields]
        rows = tab.read_arrow_table().num_rows
        return cols, TableStatistics(row_count=rows, computed_at=_now())

    def _infer(self, source: str) -> tuple[list[ColumnSpec], TableStatistics]:
        """Read schema + bounded statistics from a tabular source.

        Column names / dtypes / nullability come from the canonical
        ``Tabular.collect_schema()`` (yggdrasil.data owns the type system across
        arrow/polars/spark). polars is just the compute engine for the bounded
        null/distinct/min/max pass.
        """
        import polars as pl

        size_bytes: int | None = None
        try:
            if "://" not in source:
                size_bytes = os.path.getsize(source)
        except OSError:
            pass

        # Canonical schema — one source of truth for field types.
        tab = Tabular.from_(source, default=None)
        if tab is None:
            raise BadRequestError(f"Cannot open {source!r} as a table")
        try:
            ysch = tab.collect_schema()
        except Exception as exc:
            raise BadRequestError(f"Cannot read {source!r} as a table: {exc}")
        columns = [
            ColumnSpec(name=f.name, dtype=str(f.dtype), nullable=f.nullable)
            for f in ysch.fields
        ]
        names = [f.name for f in ysch.fields]

        # Stats pass via polars: lazy scan where possible, else read once.
        ext = source.rsplit(".", 1)[-1].lower() if "." in source else ""
        try:
            if ext in ("parquet", "pq"):
                lf = pl.scan_parquet(source)
            elif ext in ("csv", "tsv"):
                lf = pl.scan_csv(source, separator="\t" if ext == "tsv" else ",")
            elif ext == "ndjson":
                lf = pl.scan_ndjson(source)
            else:
                lf = pl.from_arrow(tab.read_arrow_table()).lazy()
        except Exception:
            return columns, TableStatistics(size_bytes=size_bytes, computed_at=_now())

        pschema = lf.collect_schema()
        try:
            row_count = lf.select(pl.len()).collect(engine="streaming").item()
        except Exception:
            row_count = None

        col_stats: list[ColumnStat] = []
        if names:
            exprs = []
            for n in names:
                if n not in pschema.names():
                    continue
                c = pl.col(n)
                exprs.append(c.null_count().alias(f"{n}|n"))
                exprs.append(c.n_unique().alias(f"{n}|d"))
                if pschema[n].is_numeric() or pschema[n] in (pl.Date, pl.Datetime):
                    exprs.append(c.min().alias(f"{n}|mn"))
                    exprs.append(c.max().alias(f"{n}|mx"))
            try:
                row = lf.select(exprs).collect(engine="streaming")
                for n in names:
                    col_stats.append(ColumnStat(
                        column=n,
                        null_count=_as_int(row[f"{n}|n"][0]) if f"{n}|n" in row.columns else None,
                        distinct_count=_as_int(row[f"{n}|d"][0]) if f"{n}|d" in row.columns else None,
                        min=_json_safe(row[f"{n}|mn"][0]) if f"{n}|mn" in row.columns else None,
                        max=_json_safe(row[f"{n}|mx"][0]) if f"{n}|mx" in row.columns else None,
                    ))
            except Exception:
                col_stats = []

        return columns, TableStatistics(
            row_count=row_count, size_bytes=size_bytes,
            columns=col_stats, computed_at=_now(),
        )

    # -- SQL execution ------------------------------------------------------

    def _resolve_dialect(self, name: str | None) -> Dialect:
        raw = (name or self.settings.saga_default_dialect or "postgres").lower()
        if raw not in _DIALECTS:
            raise BadRequestError(f"unknown dialect {raw!r}; one of {sorted(_DIALECTS)}")
        return Dialect(raw)

    def _qualify(self, ref: TableRef, catalog: str | None, schema: str | None) -> str:
        parts = [p for p in (ref.catalog or catalog, ref.schema or schema, ref.name) if p]
        return ".".join(parts)

    def _collect_refs(self, node: Any, out: list[TableRef]) -> None:
        """Walk a plan/from-item tree collecting every TableRef."""
        if node is None:
            return
        if isinstance(node, TableRef):
            out.append(node)
        elif isinstance(node, SubqueryRef):
            self._collect_refs(node.plan, out)
        elif isinstance(node, JoinClause):
            self._collect_refs(node.left, out)
            self._collect_refs(node.right, out)
        elif isinstance(node, LateralViewItem):
            pass
        elif isinstance(node, ValuesRef):
            pass
        elif isinstance(node, SelectNode):
            for cte in (node.ctes or []):
                self._collect_refs(getattr(cte, "plan", None), out)
            self._collect_refs(node.from_clause, out)
            for sop in (node.set_ops or []):
                self._collect_refs(getattr(sop, "plan", None), out)
        elif isinstance(node, (InsertNode, MergeNode)):
            self._collect_refs(node.target, out)
            self._collect_refs(node.source, out)
        elif hasattr(node, "to_plan_node"):  # SelectPlan inside a CTE/subquery
            try:
                self._collect_refs(node.to_plan_node(), out)
            except Exception:
                pass

    def plan_for(self, req: SqlRequest) -> tuple[PlanNode, Dialect, list[TableRef]]:
        if not req.sql or not req.sql.strip():
            raise BadRequestError("empty SQL statement")
        # Infer the dialect from the default catalog when the request doesn't
        # pin one — a catalog created as databricks parses its own queries.
        chosen = req.dialect
        if chosen is None and req.catalog:
            with self._lock:
                cat = self._catalog_by_name(req.catalog)
            if cat is not None:
                chosen = cat.dialect
        dialect = self._resolve_dialect(chosen)
        try:
            node = parse_sql(req.sql, dialect=dialect)
        except (ValueError, NotImplementedError) as exc:
            raise BadRequestError(f"SQL parse error: {exc}")
        refs: list[TableRef] = []
        self._collect_refs(node, refs)
        return node, dialect, refs

    def compute_node(self, req: SqlRequest) -> str | None:
        """Which node should run this query — explicit, else where the data is,
        and among equally-valid holders the freest one.

        Each referenced registered table has a set of *holders* — the node that
        owns the bytes plus any node carrying a local data replica. The query
        must run on a node that holds *every* referenced table (the intersection
        of holder sets); among those, the freest node is chosen so a busy node
        offloads to a less-loaded replica (compute follows data *and* resources).
        ``None`` means run locally. A query whose tables share no common holder
        spans nodes and needs an explicit ``node=``.
        """
        if req.node:
            return None if req.node == self.settings.node_id else req.node
        me = self.settings.node_id
        _, _, refs = self.plan_for(req)
        holder_sets: list[set[str]] = []
        with self._lock:
            for ref in refs:
                t = self._table_by_full(self._qualify(ref, req.catalog, req.schema_))
                if t is not None:
                    holder_sets.append({t.node or me} | set(t.replicas))
                else:
                    cand = ref.name
                    ext = cand.rsplit(".", 1)[-1].lower() if "." in cand else ""
                    if "/" in cand or "\\" in cand or ext in _TABULAR_EXTS:
                        holder_sets.append({me})   # raw path resolves locally
        if not holder_sets:
            return None
        candidates = set.intersection(*holder_sets)
        if not candidates:
            raise BadRequestError(
                "query spans tables on multiple nodes; pass an explicit node= to "
                "pick where it runs")
        # Resource-aware pick among the data holders.
        if len(candidates) > 1 and getattr(self._network, "least_loaded", None):
            chosen = self._network.least_loaded(
                candidates, offload_threshold=self.settings.saga_offload_load)
            if chosen:
                return None if chosen == me else chosen
        if me in candidates:
            return None
        return next(iter(candidates))

    def _resolve_view(self, view: TableEntry, seen: set[str]) -> Tabular:
        if view.full_name in seen:
            raise BadRequestError(f"view {view.full_name!r} is recursive")
        if not view.definition.strip():
            raise BadRequestError(f"view {view.full_name!r} has no definition")
        try:
            node = parse_sql(view.definition, dialect=self._resolve_dialect(None))
        except (ValueError, NotImplementedError) as exc:
            raise BadRequestError(f"view {view.full_name!r} SQL error: {exc}")
        refs: list[TableRef] = []
        self._collect_refs(node, refs)
        tables = self._build_tables(refs, view.catalog, view.schema_name,
                                    seen | {view.full_name})
        from yggdrasil.arrow.tabular import ArrowTabular
        return ArrowTabular(execute_plan(node, tables).read_arrow_table())

    def _forecast_source_frame(self, spec: ForecastSpec, catalog: str | None,
                               schema: str | None, seen: set[str]):
        """Load the forecast source as a polars DataFrame, projected to just the
        value/x/key columns. The source is a registered table (resolved through
        the same VIEW/TABLE machinery) or a node-rooted file path."""
        import polars as pl
        keep = list(dict.fromkeys(
            [spec.column] + ([spec.x] if spec.x else []) + list(spec.keys)))
        reg = self._table_by_full(self._qualify(TableRef(name=spec.source), catalog, schema)) \
            or self._table_by_full(spec.source)
        if reg is not None:
            tab = self._build_tables([TableRef(name=spec.source)], catalog, schema, seen)[
                self._qualify(TableRef(name=spec.source), catalog, schema)]
            df = pl.from_arrow(tab.read_arrow_table())
        else:
            src = self._resolve_path(spec.source)
            ext = src.rsplit(".", 1)[-1].lower() if "." in src else ""
            if ext in ("parquet", "pq"):
                lf = pl.scan_parquet(src)
            elif ext in ("csv", "tsv"):
                lf = pl.scan_csv(src, separator="\t" if ext == "tsv" else ",")
            elif ext == "ndjson":
                lf = pl.scan_ndjson(src)
            else:
                tab = Tabular.from_(src, default=None)
                if tab is None:
                    raise BadRequestError(f"cannot open forecast source {spec.source!r}")
                lf = pl.from_arrow(tab.read_arrow_table()).lazy()
            present = [c for c in keep if c in lf.collect_schema().names()]
            df = lf.select(present).collect(engine="streaming")
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise BadRequestError(f"forecast source missing columns {missing}")
        return df

    def _resolve_forecast(self, asset: TableEntry, seen: set[str]) -> "Tabular":
        """Resolve a FORECAST asset to its history+forecast table.

        Fast path: if a fresh materialised snapshot exists, scan it. Otherwise
        recompute live from the source (the workflow is a live view by default).
        """
        from yggdrasil.arrow.tabular import ArrowTabular
        if asset.full_name in seen:
            raise BadRequestError(f"forecast {asset.full_name!r} is recursive")
        if not asset.definition.strip():
            raise BadRequestError(f"forecast {asset.full_name!r} has no spec")
        spec = ForecastSpec.model_validate_json(asset.definition)
        snap = asset.properties.get("materialized_url")
        if snap:
            src = self._resolve_path(snap)
            if os.path.exists(src):
                tab = Tabular.from_(src, default=None)
                if tab is not None:
                    return tab
        from .forecast import forecast_frame
        df = self._forecast_source_frame(spec, asset.catalog, asset.schema_name,
                                         seen | {asset.full_name})
        out, _, _ = forecast_frame(
            df, value=spec.column, x=spec.x, keys=list(spec.keys),
            horizon=spec.horizon, model=spec.model, period=spec.period, agg=spec.agg)
        return ArrowTabular(out.to_arrow())

    def _build_tables(self, refs: list[TableRef], catalog: str | None,
                      schema: str | None, seen: set[str] | None = None) -> dict[str, Tabular]:
        """Resolve referenced table names to concrete Tabular sources."""
        seen = seen or set()
        tables: dict[str, Tabular] = {}
        for ref in refs:
            full = self._qualify(ref, catalog, schema)
            with self._lock:
                reg = self._table_by_full(full) or self._table_by_full(ref.name)
            if reg is not None:
                if reg.node and reg.node != self.settings.node_id:
                    raise BadRequestError(
                        f"table {full!r} lives on node {reg.node!r}; run the query "
                        "with node= set to that node"
                    )
                if reg.object_type == "VIEW":
                    # A view resolves to the materialised result of its SQL — its
                    # own table refs resolve recursively against the catalog.
                    tab = self._resolve_view(reg, seen)
                elif reg.object_type == "FORECAST":
                    # A forecast workflow resolves to its history+forecast view.
                    tab = self._resolve_forecast(reg, seen)
                elif reg.object_type != "TABLE":
                    raise BadRequestError(
                        f"{reg.object_type.lower()} {full!r} is not queryable")
                else:
                    src = self._resolve_path(reg.source_url)
                    tab = Tabular.from_(src, default=None)
                    if tab is None:
                        raise BadRequestError(f"cannot open source for table {full!r}")
                # Register under every name shape the executor might look up.
                tables[full] = tab
                tables[reg.name] = tab
                if reg.schema_name:
                    tables[f"{reg.schema_name}.{reg.name}"] = tab
                continue
            # Unregistered but path-shaped (a slash or a tabular extension):
            # resolve it under the node files root so `SELECT * FROM
            # 'data/x.parquet'` reads from the node, not the process CWD.
            cand = ref.name
            ext = cand.rsplit(".", 1)[-1].lower() if "." in cand else ""
            looks_like_path = ("/" in cand or "\\" in cand or ext in _TABULAR_EXTS)
            if looks_like_path and is_tabular_source(cand):
                tab = Tabular.from_(self._resolve_path(cand), default=None)
                if tab is not None:
                    tables[cand] = tab
            # Anything else falls through to the executor's own resolution.
        return tables

    async def execute_sql(self, req: SqlRequest) -> SqlResult:
        return await run_in_threadpool(self._execute_sql, req)

    def _execute_sql(self, req: SqlRequest) -> SqlResult:
        node, dialect, refs = self.plan_for(req)
        limit = req.limit if req.limit and req.limit > 0 else self.settings.saga_sql_preview_rows
        # Bound the work: push a preview LIMIT into a LIMIT-less top SELECT so
        # the executor slices at the source instead of materialising everything.
        if isinstance(node, SelectNode) and node.limit is None:
            node.limit = limit + 1
        tables = self._build_tables(refs, req.catalog, req.schema_)
        t0 = time.perf_counter()
        try:
            result = execute_plan(node, tables)
            # Bound the materialised read with CastOptions.row_limit (+1 to
            # detect "there's more") instead of reading everything then slicing.
            table = result.read_arrow_table(options=CastOptions(row_limit=limit + 1))
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        elapsed = (time.perf_counter() - t0) * 1000.0

        truncated = table.num_rows > limit
        if truncated:
            table = table.slice(0, limit)
        cols = _sql_columns(table.schema)
        pydict = table.to_pydict()
        col_data = [pydict[c.name] for c in cols]
        rows = [[_json_safe(col_data[ci][ri]) for ci in range(len(cols))]
                for ri in range(table.num_rows)]
        try:
            plan_sql = node.to_sql(dialect=dialect)
        except Exception:
            plan_sql = ""
        ref_names = sorted({self._qualify(r, req.catalog, req.schema_) for r in refs})
        # One log row per touched *registered* asset — queries against raw URLs
        # aren't catalog assets, so they don't accrue history.
        with self._lock:
            known = {t.full_name for t in self._tables.values()}
        for asset in ref_names:
            if asset in known:
                self._record(asset, "query", statement=req.sql, rows=table.num_rows,
                             detail=f"{round(elapsed, 1)}ms")
        return SqlResult(
            node_id=self.settings.node_id, columns=cols, rows=rows,
            row_count=table.num_rows, truncated=truncated,
            elapsed_ms=round(elapsed, 2), plan_sql=plan_sql,
            referenced_tables=ref_names,
        )

    def explain(self, req: SqlRequest) -> ExplainResult:
        node, dialect, refs = self.plan_for(req)
        try:
            plan_sql = node.to_sql(dialect=dialect)
        except Exception:
            plan_sql = ""
        return ExplainResult(
            node_id=self.settings.node_id, dialect=dialect.value,
            plan=repr(node), plan_sql=plan_sql,
            referenced_tables=sorted({self._qualify(r, req.catalog, req.schema_) for r in refs}),
            statement=req.sql,
        )

    # -- execution plan graph ----------------------------------------------

    _AGG = {"COUNT", "SUM", "AVG", "MEAN", "MIN", "MAX", "STDDEV", "VARIANCE",
            "MEDIAN", "APPROX_COUNT_DISTINCT", "COLLECT_LIST", "COLLECT_SET"}

    def _is_aggregate(self, node: SelectNode) -> bool:
        from yggdrasil.execution.expr.nodes import Alias, FunctionCall
        if node.group_by:
            return True
        for p in (node.projections or []):
            e = p.expr if isinstance(p, Alias) else p
            if isinstance(e, FunctionCall) and e.name.upper() in self._AGG:
                return True
        return False

    def _plan_ops(self, node: PlanNode, refs: list[TableRef],
                  catalog: str | None, schema: str | None, dialect: Dialect) -> list:
        """Logical operations of a query as a DAG (scans feed a linear pipeline;
        joins/unions branch)."""
        from yggdrasil.execution.expr.nodes import Alias, Column, FunctionCall, SortOrder, Star

        ops: list[PlanOp] = []
        scan_ids: list[str] = []
        for i, ref in enumerate(refs or []):
            sid = f"scan{i}"
            ops.append(PlanOp(id=sid, op="scan", title="Scan",
                              detail=self._qualify(ref, catalog, schema)))
            scan_ids.append(sid)
        if not scan_ids:
            ops.append(PlanOp(id="scan0", op="scan", title="Scan", detail="—"))
            scan_ids = ["scan0"]

        if len(scan_ids) > 1:
            ops.append(PlanOp(id="join", op="join", title="Join",
                              detail=f"{len(scan_ids)} inputs", inputs=list(scan_ids)))
            cur = "join"
        else:
            cur = scan_ids[0]

        if not isinstance(node, SelectNode):
            return ops

        def chain(op_id: str, op: str, title: str, detail: str) -> None:
            nonlocal cur
            ops.append(PlanOp(id=op_id, op=op, title=title, detail=detail, inputs=[cur]))
            cur = op_id

        def esql(expr) -> str:
            try:
                return expr.to_sql(dialect=dialect)
            except Exception:
                return str(expr)[:80]

        if node.where is not None:
            chain("filter", "filter", "Filter", esql(node.where))
        if self._is_aggregate(node):
            keys = ", ".join(c.name for c in (node.group_by or []) if isinstance(c, Column))
            measures = ", ".join(
                (p.expr if isinstance(p, Alias) else p).name + "()"
                for p in (node.projections or [])
                if isinstance((p.expr if isinstance(p, Alias) else p), FunctionCall))
            detail = (f"by {keys} · {measures}" if keys else measures) or "aggregate"
            chain("aggregate", "aggregate", "Aggregate", detail)
        if node.having is not None:
            chain("having", "having", "Having", esql(node.having))
        if getattr(node, "distinct", False):
            chain("distinct", "distinct", "Distinct", "")
        names = [
            (p.name if isinstance(p, Alias)
             else p.name if isinstance(p, Column) else None)
            for p in (node.projections or [])
            if not isinstance((p.expr if isinstance(p, Alias) else p), Star)
        ]
        names = [n for n in names if n]
        if names and not self._is_aggregate(node):
            chain("project", "project", "Project", ", ".join(names[:8]))
        if node.order_by:
            keys = ", ".join(
                f"{o.expr.name}{'' if getattr(o,'ascending',True) else ' desc'}"
                for o in node.order_by
                if isinstance(o, SortOrder) and isinstance(o.expr, Column))
            chain("sort", "sort", "Sort", keys)
        if node.limit is not None or node.offset:
            d = []
            if node.limit is not None:
                d.append(f"LIMIT {node.limit}")
            if node.offset:
                d.append(f"OFFSET {node.offset}")
            chain("limit", "limit", "Limit", " ".join(d))
        return ops

    def build_plan(self, req: SqlRequest) -> PlanGraph:
        node, dialect, refs = self.plan_for(req)
        ops = self._plan_ops(node, refs, req.catalog, req.schema_, dialect)
        try:
            plan_sql = node.to_sql(dialect=dialect)
        except Exception:
            plan_sql = ""
        return PlanGraph(node_id=self.settings.node_id, dialect=dialect.value,
                         statement=req.sql, plan_sql=plan_sql, ops=ops)

    async def analyze_plan(self, req: SqlRequest) -> PlanGraph:
        return await run_in_threadpool(self._analyze_plan, req)

    def _analyze_plan(self, req: SqlRequest) -> PlanGraph:
        """Run the query in staged prefixes through the real engine, timing each
        and recording rows out — an EXPLAIN ANALYZE the editor can visualise."""
        import dataclasses

        from yggdrasil.execution.expr.nodes import Star

        node, dialect, refs = self.plan_for(req)
        graph = self.build_plan(req)
        if not isinstance(node, SelectNode):
            return graph
        tables = self._build_tables(refs, req.catalog, req.schema_)

        def run(n) -> tuple[int, float]:
            t0 = time.perf_counter()
            tbl = execute_plan(n, dict(tables)).read_arrow_table()
            return tbl.num_rows, (time.perf_counter() - t0) * 1000.0

        cap = self.settings.analysis_max_rows
        # Measured stages → which op id they land on.
        scan = dataclasses.replace(
            node, where=None, group_by=None, having=None, qualify=None,
            order_by=None, limit=cap, offset=None, distinct=False,
            projections=[Star()], set_ops=None)
        stages: list[tuple[str, Any]] = [(graph.ops[0].id if graph.ops else "scan0", scan)]
        if node.where is not None:
            stages.append(("filter", dataclasses.replace(scan, where=node.where)))
        if self._is_aggregate(node):
            stages.append(("aggregate", dataclasses.replace(
                node, order_by=None, limit=None, offset=None, distinct=False)))
        last_id = graph.ops[-1].id if graph.ops else "scan0"
        stages.append((last_id, node))

        by_id = {o.id: o for o in graph.ops}
        sampled = False
        total = 0.0
        # Each stage is timed independently through the real engine (so a stage's
        # number is its own wall time, not a fragile delta); rows-out is exact.
        for op_id, n in stages:
            try:
                rows, ms = run(n)
            except Exception:
                continue
            total = ms  # the final (full-query) stage is the headline total
            if rows >= cap:
                sampled = True
            op = by_id.get(op_id)
            if op is not None:
                op.rows = rows
                op.elapsed_ms = round(ms, 2)
        graph.analyzed = True
        graph.total_ms = round(total, 2)
        graph.sampled = sampled
        return graph

    def edit_plan(self, req: PlanEditRequest) -> PlanEditResult:
        """Apply structural edits to a parsed query and re-emit SQL — the plan
        player's 'modify live' path."""
        import dataclasses

        dialect = self._resolve_dialect(req.dialect)
        try:
            node = parse_sql(req.sql, dialect=dialect)
        except (ValueError, NotImplementedError) as exc:
            raise BadRequestError(f"SQL parse error: {exc}")
        if not isinstance(node, SelectNode):
            raise BadRequestError("only SELECT statements can be edited")
        for e in req.edits:
            if e.op == "set_limit":
                node = dataclasses.replace(node, limit=e.value)
            elif e.op == "set_offset":
                node = dataclasses.replace(node, offset=e.value)
            elif e.op == "drop_limit":
                node = dataclasses.replace(node, limit=None, offset=None)
            elif e.op == "drop_filter":
                node = dataclasses.replace(node, where=None)
            elif e.op == "drop_group":
                node = dataclasses.replace(node, group_by=None, having=None)
            elif e.op == "drop_order":
                node = dataclasses.replace(node, order_by=None)
            elif e.op == "drop_distinct":
                node = dataclasses.replace(node, distinct=False)
            else:
                raise BadRequestError(f"unknown plan edit {e.op!r}")
        try:
            sql = node.to_sql(dialect=dialect)
        except Exception as exc:
            raise BadRequestError(f"cannot re-emit edited plan: {exc}")
        return PlanEditResult(node_id=self.settings.node_id, sql=sql, plan_sql=sql)

    def execute_sql_arrow(self, req: SqlRequest):
        """Run the query and return (iterator-of-ipc-bytes, cleanup-callable).

        Streams the result **batch by batch** straight into the Arrow IPC
        encoder, so peak memory stays near one chunk regardless of result size
        (bench: ~3.7 MB streamed vs ~26 MB to buffer a 1M-row result whole). A
        source that already spilled to disk is read from its parts, never
        re-concatenated. The schema is learned by peeking the first batch — no
        upfront full materialisation.
        """
        import itertools

        node, _dialect, refs = self.plan_for(req)
        # Honor a display limit on a LIMIT-less SELECT so a preview fetch stays
        # bounded without the caller rewriting the SQL.
        if req.limit and req.limit > 0 and isinstance(node, SelectNode) and node.limit is None:
            node.limit = req.limit
        tables = self._build_tables(refs, req.catalog, req.schema_)
        # CastOptions.row_size → Arrow IPC chunk size.
        chunk = req.batch_rows if (req.batch_rows and req.batch_rows > 0) else 65536
        try:
            result = execute_plan(node, tables)
            batches = iter(result.read_arrow_batches(CastOptions(row_size=chunk)))
            first = next(batches, None)
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        if first is None:  # empty result — still emit the schema frame
            schema = result.read_arrow_table().schema
            return transport.iter_arrow_ipc_stream(iter(()), schema), None
        return transport.iter_arrow_ipc_stream(itertools.chain([first], batches), first.schema), None

    # -- discovery ----------------------------------------------------------

    async def discover(self, req: DiscoverRequest) -> TableListResponse:
        """Scan a folder under the node home and register tabular files.

        ``path`` is node-home-relative (same as the /fs browser); registered
        ``source_url`` values are too, so they open directly in the SQL engine
        and the tabular preview alike.
        """
        from pathlib import Path as _P

        root = self.settings.node_home
        base = (root / req.path.lstrip("/")).resolve() if req.path else root
        if not str(base).startswith(str(root.resolve())):
            raise BadRequestError("path escapes the node home")
        if not base.exists() or not base.is_dir():
            raise NotFoundError(f"directory not found: {req.path!r}")

        found: list[_P] = []
        it = base.rglob("*") if req.recursive else base.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lstrip(".").lower() in _TABULAR_EXTS:
                found.append(p)

        for p in found:
            rel = str(p.relative_to(root))
            name = p.stem
            await self.create_table(req.catalog, req.schema_, TableCreate(
                name=name, source_url=rel, node=req.node, table_type="EXTERNAL",
                infer=req.infer,
            ))
        return await self.list_tables(req.catalog, req.schema_)

    # -- staging (remote result placement) ----------------------------------

    async def stage_result(self, req: SqlRequest) -> StagedResult:
        return await run_in_threadpool(self._stage_result, req)

    def _stage_result(self, req: SqlRequest) -> StagedResult:
        """Run the query and write the Arrow result to ``req.staging_path``.

        The staging path is a NodePath URL (``npfs://host:port/stg-…`` or local).
        Lets the node that holds the data compute the result and park it next to
        whoever asked, instead of streaming bytes back through this node.
        """
        if not req.staging_path:
            raise BadRequestError("staging_path is required")
        node, dialect, refs = self.plan_for(req)
        tables = self._build_tables(refs, req.catalog, req.schema_)
        t0 = time.perf_counter()
        try:
            table = execute_plan(node, tables).read_arrow_table()
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        data = transport.write_arrow_stream_bytes(table)
        written = self._stage_write(req.staging_path, data)
        return StagedResult(
            node_id=self.settings.node_id, staging_path=written,
            columns=_sql_columns(table.schema),
            row_count=table.num_rows, bytes=len(data),
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    def _stage_write(self, staging_path: str, data: bytes) -> str:
        """Write Arrow bytes to a staging NodePath; return where they landed.

        A remote ``npfs://host:port/path`` goes through the peer's fs API. A
        local or scheme-less path is resolved against *this* node's files root
        (NodePath's own local rooting uses the process-global settings, which
        isn't what an injected-settings node wants)."""
        if staging_path.startswith("npfs://") and not staging_path.startswith("npfs:///"):
            from yggdrasil.node.path import NodePath
            dest = NodePath.from_url(staging_path)
            if not dest.is_local:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                dest.write_bytes(data)
                return staging_path
            rest = staging_path.split("://", 1)[1]
            rel = rest.split("/", 1)[1] if "/" in rest else ""
        else:
            rel = staging_path.split("://", 1)[1] if "://" in staging_path else staging_path
        target = (self.settings.files_root / rel.lstrip("/")).resolve()
        if not str(target).startswith(str(self.settings.files_root.resolve())):
            raise BadRequestError("staging_path escapes the node files root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    # -- materialize (run once → a path the path-based APIs can analyse) -----

    async def materialize_sql(self, req: SqlRequest):
        return await run_in_threadpool(self._materialize_sql, req)

    def _materialize_sql(self, req: SqlRequest):
        """Run the query and write the result to a tmp parquet, returning a
        node-home-relative path. Lets the existing path-based /tabular and
        /analysis surfaces drive analytics over a SQL result — so Saga and the
        Files browser share the exact same tabular display + analyze + export."""
        from pathlib import Path as _P

        import pyarrow.parquet as pq

        node, _, refs = self.plan_for(req)
        tables = self._build_tables(refs, req.catalog, req.schema_)
        t0 = time.perf_counter()
        try:
            table = execute_plan(node, tables).read_arrow_table()
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        spill = scratch.new_path(self.settings.tmp_root, "tmp",
                                 ttl_seconds=self.settings.tmp_ttl, suffix="materialize.parquet")
        pq.write_table(table, str(spill))
        rel = str(_P(spill).relative_to(self.settings.node_home))
        return MaterializeResult(
            node_id=self.settings.node_id, path=rel,
            columns=_sql_columns(table.schema), row_count=table.num_rows,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    # -- staged session (lazy windowed display of a heavy result) -----------

    async def sql_session(self, req: SqlRequest):
        return await run_in_threadpool(self._sql_session, req)

    def _sql_session(self, req: SqlRequest):
        """Run the query once and stage the result to an Arrow IPC file, so the
        client can scroll a huge result by fetching small windows instead of
        decoding the whole thing in the browser. The file is memory-mapped on
        every window read (zero-copy) and reclaimed by the tmp janitor (or
        /session/close) — cleared when the viewer is done or disconnects."""
        import polars as pl

        node, _, refs = self.plan_for(req)
        tables = self._build_tables(refs, req.catalog, req.schema_)
        t0 = time.perf_counter()
        try:
            table = execute_plan(node, tables).read_arrow_table()
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        spill = scratch.new_path(self.settings.tmp_root, "tmp",
                                 ttl_seconds=self.settings.tmp_ttl, suffix="session.arrow")
        pl.from_arrow(table).write_ipc(str(spill))   # Arrow IPC file (scan_ipc-able)
        from pathlib import Path as _P
        rel = str(_P(spill).relative_to(self.settings.node_home))
        return SessionResult(
            node_id=self.settings.node_id, path=rel,
            columns=_sql_columns(table.schema), row_count=table.num_rows,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        )

    async def window(self, req):
        return await run_in_threadpool(self._window, req)

    def _window(self, req) -> tuple[bytes, int, bool]:
        """A lazily-transformed slice of a staged session as Arrow IPC bytes.

        Filters/sort/projection and nested explode/unnest are applied with
        polars **lazy** over the mmap'd Arrow file — only the requested window is
        ever materialised, so memory stays bounded no matter how big the result.
        Returns (ipc_bytes, rows_in_window, has_more)."""
        import polars as pl

        src = self._resolve_path(req.path)
        try:
            lf = pl.scan_ipc(src)
        except Exception as exc:
            raise BadRequestError(f"session not found or unreadable: {exc}")
        lf = _pl_apply_filters(lf, req.filters)
        for t in req.transforms:
            try:
                lf = lf.explode(t.column) if t.op == "explode" else lf.unnest(t.column)
            except Exception as exc:
                raise BadRequestError(f"transform {t.op}({t.column}) failed: {exc}")
        if req.columns:
            lf = lf.select(req.columns)
        if req.sort:
            lf = lf.sort(req.sort, descending=req.descending, nulls_last=True)
        # Pull one extra row to tell the client whether to keep scrolling.
        out = lf.slice(max(0, req.offset), max(1, req.limit) + 1).collect(engine="streaming")
        has_more = out.height > req.limit
        if has_more:
            out = out.head(req.limit)
        return transport.write_arrow_stream_bytes(out.to_arrow()), out.height, has_more

    def close_session(self, path: str) -> bool:
        """Delete a staged session file (called on viewer close)."""
        try:
            target = self._resolve_path(path)
        except BadRequestError:
            return False
        # Only ever unlink inside the node's tmp area.
        if not str(target).startswith(str(self.settings.tmp_root.resolve())):
            return False
        try:
            os.unlink(target)
            return True
        except OSError:
            return False

    # -- export (download the full media type) -----------------------------

    async def export_sql(self, req):
        return await run_in_threadpool(self._export_sql, req)

    def _export_sql(self, req):
        """Run the query (full result, optionally row-capped) and write it to a
        tmp file in the requested media type. Returns (path, download_name).

        Leverages the project's MediaType registry + Path writer, so every
        tabular format it handles — csv/parquet/json/ndjson/arrow/xlsx/tsv —
        comes for free. The caller streams the file then unlinks it.
        """
        from yggdrasil.enums.media_type import MediaType
        from yggdrasil.path import Path as YggPath

        node, _, refs = self.plan_for(SqlRequest(
            sql=req.sql, dialect=req.dialect, catalog=req.catalog, schema=req.schema_))
        if isinstance(node, SelectNode) and node.limit is None and req.max_rows:
            node.limit = req.max_rows
        tables = self._build_tables(refs, req.catalog, req.schema_)
        try:
            table = execute_plan(node, tables).read_arrow_table()
        except (BadRequestError, NotFoundError):
            raise
        except Exception as exc:
            raise BadRequestError(f"query failed: {exc}")
        if req.max_rows and table.num_rows > req.max_rows:
            table = table.slice(0, req.max_rows)

        fmt = (req.fmt or "csv").lower()
        if fmt not in _EXPORT_FMTS:
            raise BadRequestError(f"unsupported export format {fmt!r}; one of {sorted(_EXPORT_FMTS)}")
        ext = {"ipc": "arrow"}.get(fmt, fmt)
        media = MediaType.from_(fmt, default=None) or MediaType.from_("csv")
        spill = str(scratch.new_path(
            self.settings.tmp_root, "tmp", ttl_seconds=self.settings.tmp_ttl,
            suffix=f"saga-export.{ext}"))
        with YggPath.from_(spill).open("wb", media_type=media) as bio:
            bio.write_arrow_table(table)
        ref = sorted({self._qualify(r, req.catalog, req.schema_) for r in refs})
        with self._lock:
            known = {t.full_name for t in self._tables.values()}
        for a in ref:
            if a in known:
                self._record(a, "export", statement=req.sql, rows=table.num_rows, detail=fmt)
        base = (ref[0].split(".")[-1] if ref else "result")
        from pathlib import Path as _P
        return _P(spill), f"{base}.{ext}"

    # -- replication --------------------------------------------------------

    def export_payload(self, catalog: str, schema: str, table: str) -> TablePayload:
        with self._lock:
            cat = self._require_catalog(catalog)
            self._require_schema(catalog, schema)
            t = self._table_by_name(catalog, schema, table)
            if t is None:
                raise NotFoundError(f"Table {catalog}.{schema}.{table!r} not found")
            return TablePayload(catalog=catalog, schema=schema, table=t,
                                catalog_dialect=cat.dialect)

    async def import_payload(self, payload: TablePayload) -> TableResponse:
        """Register a table (and its catalog/schema) received from a peer."""
        if self._catalog_by_name(payload.catalog) is None:
            await self.create_catalog(CatalogCreate(name=payload.catalog,
                                                    dialect=payload.catalog_dialect))
        if self._schema_by_name(payload.catalog, payload.schema_) is None:
            await self.create_schema(payload.catalog, SchemaCreate(name=payload.schema_))
        t = payload.table
        return await self.create_table(payload.catalog, payload.schema_, TableCreate(
            name=t.name, source_url=t.source_url, node=t.node,
            table_type=t.table_type, format=t.format, comment=t.comment,
            columns=list(t.columns), infer=False, properties=dict(t.properties),
        ))

    async def replicate(self, req: ReplicateRequest) -> ReplicateResult:
        """Replicate a table's metadata (and optionally data) onto a peer.

        ``metadata`` re-registers the table on the target pointing at the same
        source (use when the filesystem is shared). ``data`` copies the file to
        the target's replica area first, so the target reads it locally.
        """
        if self._network is None:
            raise BadRequestError("peer network not available on this node")
        payload = self.export_payload(req.catalog, req.schema_, req.table)
        target_http = self._network.peer_url(req.target)
        if target_http is None:
            raise BadRequestError(f"target {req.target!r} is not a linked peer")
        full = payload.table.full_name
        bytes_copied = 0
        target_source_url = payload.table.source_url

        if req.mode == "data":
            from pathlib import Path as _P

            from yggdrasil.node.path import NodePath
            local = self._resolve_path(payload.table.source_url)
            data = _P(local).read_bytes()
            safe = full.replace(".", "_")
            ext = payload.table.format or "bin"
            # node-home-relative, matching _resolve_path's rooting on the target.
            target_source_url = f"saga-replicas/{safe}.{ext}"
            dest = NodePath(target_source_url, node_url=target_http)
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            dest.write_bytes(data)
            bytes_copied = len(data)
            payload = payload.model_copy(update={
                "table": payload.table.model_copy(update={
                    "source_url": target_source_url, "node": None}),
            })
        elif req.mode == "metadata":
            # The target keeps a pointer to where the data lives, so queries on
            # the target route compute back to this node (compute follows data).
            payload = payload.model_copy(update={
                "table": payload.table.model_copy(update={
                    "node": payload.table.node or self.settings.node_id}),
            })
        else:
            raise BadRequestError(f"unknown replicate mode {req.mode!r}")

        await self._network.proxy_json(req.target, "POST", "/api/v2/saga/import",
                                       json_body=payload.model_dump(by_alias=True))
        # Only a *data* replica makes the target a holder the query router can
        # offload to; a metadata copy just points back here, so it isn't one.
        if req.mode == "data":
            with self._lock:
                cur = self._tables.get(payload.table.id) or self._table_by_name(
                    req.catalog, req.schema_, req.table)
                if cur is not None and req.target not in cur.replicas:
                    self._tables[cur.id] = cur.model_copy(update={
                        "replicas": sorted(set(cur.replicas) | {req.target}),
                        "updated_at": _now()})
                    self._save()
        self._record(full, "replicate", detail=f"-> {req.target} ({req.mode}, {bytes_copied}B)")
        return ReplicateResult(
            source_node=self.settings.node_id, target_node=req.target,
            full_name=full, mode=req.mode, bytes_copied=bytes_copied,
            target_source_url=target_source_url,
        )


def _pl_apply_filters(lf, filters):
    """Push session-window row filters into a polars LazyFrame."""
    import polars as pl
    for f in filters:
        col = pl.col(f.column)
        if f.op == "is_null":
            lf = lf.filter(col.is_null())
        elif f.op == "not_null":
            lf = lf.filter(col.is_not_null())
        elif f.op == "contains":
            lf = lf.filter(col.cast(pl.Utf8).str.contains(str(f.value), literal=True))
        elif f.op == "in":
            vals = f.value if isinstance(f.value, list) else [f.value]
            lf = lf.filter(col.is_in(vals))
        elif f.op in ("==", "!=", ">", ">=", "<", "<="):
            v = f.value
            lf = lf.filter({"==": col == v, "!=": col != v, ">": col > v,
                            ">=": col >= v, "<": col < v, "<=": col <= v}[f.op])
        else:
            raise BadRequestError(f"unknown filter op {f.op!r}")
    return lf


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _dtype_label(arrow_type: "pa.DataType") -> str:
    """One-line canonical dtype name via yggdrasil.data's type system.

    Routes pyarrow types through ``DataType.from_arrow_type`` so the SQL grid
    labels match the catalog's inferred column dtypes exactly, collapsing the
    pretty-printer's whitespace for nested types onto a single line.
    """
    from yggdrasil.data.types.base import DataType
    return " ".join(str(DataType.from_arrow_type(arrow_type)).split())


def _sql_columns(schema: "pa.Schema") -> list[SqlColumn]:
    return [SqlColumn(name=f.name, dtype=_dtype_label(f.type)) for f in schema]
