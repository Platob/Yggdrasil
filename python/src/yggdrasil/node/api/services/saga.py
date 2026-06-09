"""SagaService — SQL catalog + DuckDB-backed SQL engine.

Layout on disk (under ``settings.saga_home``):
    catalog/<name>/meta.json
    catalog/<name>/schema/<name>/meta.json
    catalog/<name>/schema/<name>/table/<name>/meta.json
    catalog/<name>/schema/<name>/table/<name>/snapshot.arrow  (FORECAST materialized)
    mounts.json            -- named mount aliases
    log.jsonl              -- op log

The SQL engine uses DuckDB (via ``duckdb`` package) with lazy registration:
a table is registered as a DuckDB view pointing at the parquet/arrow/CSV
file when it's first referenced in a query.

Mounts (named aliases over paths / DB URIs) are stored in ``mounts.json``
and exposed via ``GET/POST /api/v2/saga/mount`` and
``GET /api/v2/saga/mount/{alias}/ls``.  The ``fs/nodes`` response includes
mounts under the ``mounts`` key.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

# catalog.schema.table dotted references in a SQL statement.
_FQN_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")

from yggdrasil.node.api.schemas.mount import MountCreate, MountEntry, MountInfo, MountLsResult
from yggdrasil.node.api.schemas.saga import (
    CatalogCreate,
    CatalogInfo,
    CatalogResult,
    ColumnInfo,
    ForecastRegisterRequest,
    ForecastRegisterResult,
    ForecastSpec,
    SchemaCreate,
    SchemaInfo,
    SchemaResult,
    SqlRequest,
    SqlResult,
    TableCreate,
    TableInfo,
    TableResult,
    TableStatistics,
)

_TABULAR_EXTS = frozenset({
    ".parquet", ".pq", ".csv", ".ndjson", ".jsonl", ".arrow", ".ipc", ".feather",
    ".xlsx", ".xls",
})


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# SagaService
# ---------------------------------------------------------------------------

class SagaService:
    """Saga SQL catalog + query engine + mount aliases."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._home = Path(settings.saga_home)
        self._node_home = Path(settings.node_home)
        self._home.mkdir(parents=True, exist_ok=True)
        self._log_path = self._home / "log.jsonl"
        self._mounts_path = self._home / "mounts.json"

    def _catalog_path(self, catalog: str) -> Path:
        return self._home / "catalog" / catalog

    def _schema_path(self, catalog: str, schema: str) -> Path:
        return self._catalog_path(catalog) / "schema" / schema

    def _table_path(self, catalog: str, schema: str, table: str) -> Path:
        return self._schema_path(catalog, schema) / "table" / table

    # ------------------------------------------------------------------
    # catalog CRUD
    # ------------------------------------------------------------------

    async def create_catalog(self, req: CatalogCreate) -> CatalogResult:
        p = self._catalog_path(req.name) / "meta.json"
        existing = _load_json(p)
        data = {"name": req.name, "comment": req.comment or existing.get("comment", "")}
        _save_json(p, data)
        self._record(req.name, "catalog_create")
        count = len(list((self._catalog_path(req.name) / "schema").glob("*/meta.json"))) \
            if (self._catalog_path(req.name) / "schema").exists() else 0
        return CatalogResult(catalog=CatalogInfo(name=req.name, comment=data["comment"],
                                                  schema_count=count))

    async def list_catalogs(self) -> list[CatalogInfo]:
        base = self._home / "catalog"
        if not base.exists():
            return []
        results = []
        for meta in sorted(base.glob("*/meta.json")):
            d = _load_json(meta)
            if not d:
                continue
            name = d.get("name", meta.parent.name)
            schema_dir = meta.parent / "schema"
            count = len(list(schema_dir.glob("*/meta.json"))) if schema_dir.exists() else 0
            results.append(CatalogInfo(name=name, comment=d.get("comment", ""),
                                        schema_count=count))
        return results

    async def create_schema(self, catalog: str, req: SchemaCreate) -> SchemaResult:
        p = self._schema_path(catalog, req.name) / "meta.json"
        existing = _load_json(p)
        data = {"name": req.name, "catalog": catalog,
                "comment": req.comment or existing.get("comment", "")}
        _save_json(p, data)
        self._record(f"{catalog}.{req.name}", "schema_create")
        count = len(list((self._schema_path(catalog, req.name) / "table").glob("*/meta.json"))) \
            if (self._schema_path(catalog, req.name) / "table").exists() else 0
        return SchemaResult(schema_=SchemaInfo(name=req.name, catalog=catalog,
                                               comment=data["comment"], table_count=count))

    async def list_schemas(self, catalog: str) -> list[SchemaInfo]:
        base = self._catalog_path(catalog) / "schema"
        if not base.exists():
            return []
        results = []
        for meta in sorted(base.glob("*/meta.json")):
            d = _load_json(meta)
            if not d:
                continue
            name = d.get("name", meta.parent.name)
            tbl_dir = meta.parent / "table"
            count = len(list(tbl_dir.glob("*/meta.json"))) if tbl_dir.exists() else 0
            results.append(SchemaInfo(name=name, catalog=catalog,
                                       comment=d.get("comment", ""), table_count=count))
        return results

    # ------------------------------------------------------------------
    # table CRUD
    # ------------------------------------------------------------------

    async def create_table(self, catalog: str, schema: str, req: TableCreate) -> TableResult:
        p = self._table_path(catalog, schema, req.name) / "meta.json"
        existing = _load_json(p)
        source = req.source_url
        # Resolve relative paths against the node home.
        if not source.startswith(("s3://", "dbfs:/", "http://", "https://", "npfs://")):
            abs_src = (self._node_home / source.lstrip("/")).resolve()
            source = str(abs_src) if abs_src.exists() else source

        cols: list[dict] = existing.get("columns", [])
        stats: dict = existing.get("statistics", {})
        if req.infer and os.path.exists(source):
            cols, stats = _infer_schema_stats(source)

        data = {"name": req.name, "catalog": catalog, "schema_": schema,
                "source_url": req.source_url, "kind": "TABLE",
                "columns": cols, "statistics": stats}
        _save_json(p, data)
        self._record(f"{catalog}.{schema}.{req.name}", "table_create")
        return TableResult(table=_table_info_from(data))

    async def get_table(self, catalog: str, schema: str, table: str) -> TableResult:
        p = self._table_path(catalog, schema, table) / "meta.json"
        d = _load_json(p)
        if not d:
            raise KeyError(f"Table {catalog}.{schema}.{table} not found.")
        return TableResult(table=_table_info_from(d))

    async def list_tables(self, catalog: str, schema: str) -> list[TableInfo]:
        base = self._table_path(catalog, schema, "__placeholder__").parent.parent
        # base = schema_path / "table"
        base = self._schema_path(catalog, schema) / "table"
        if not base.exists():
            return []
        results = []
        for meta in sorted(base.glob("*/meta.json")):
            d = _load_json(meta)
            if d:
                results.append(_table_info_from(d))
        return results

    # ------------------------------------------------------------------
    # SQL execution
    # ------------------------------------------------------------------

    def _resolve_source(self, source_url: str) -> str:
        """Resolve a source_url to an absolute path or return it as-is."""
        if source_url.startswith(("s3://", "dbfs:/", "http://", "https://", "npfs://")):
            return source_url
        abs_src = (self._node_home / source_url.lstrip("/")).resolve()
        if abs_src.exists():
            return str(abs_src)
        return source_url

    def _scan_lazy(self, src: str) -> "pl.LazyFrame":
        suf = Path(src).suffix.lower()
        if suf in (".parquet", ".pq"):
            return pl.scan_parquet(src)
        if suf == ".csv":
            return pl.scan_csv(src)
        if suf in (".ndjson", ".jsonl"):
            return pl.scan_ndjson(src)
        import pyarrow.ipc as ipc
        return pl.from_arrow(ipc.open_file(src).read_all()).lazy()

    def _build_context(self, sql: str) -> tuple["pl.SQLContext", str]:
        """Register every ``catalog.schema.table`` referenced in *sql* into a
        polars SQLContext under a sanitised identifier and rewrite the dotted
        reference to it.  Also registers bare table names (just the table
        component) as fallback aliases so ``SELECT * FROM trades`` works
        alongside the full ``SELECT * FROM main.market.trades``.
        Scans are lazy, so projection/predicate pushdown still reaches the
        parquet reader."""
        ctx = pl.SQLContext()
        rewritten = sql
        cat_base = self._home / "catalog"
        if not cat_base.exists():
            return ctx, rewritten

        # Walk all registered tables once; register each under its FQN
        # alias AND under its bare name (last one wins on collision).
        bare: dict[str, str] = {}  # table_name -> lazy frame id (for bare-name fallback)
        for meta in cat_base.glob("*/schema/*/table/*/meta.json"):
            d = _load_json(meta)
            if not d:
                continue
            cat = d.get("catalog", meta.parts[-7] if len(meta.parts) >= 7 else "")
            sch = d.get("schema_", meta.parts[-5] if len(meta.parts) >= 5 else "")
            tbl = d.get("name", meta.parts[-3] if len(meta.parts) >= 3 else "")
            fqn = f"{cat}.{sch}.{tbl}"
            ident = f"{cat}__{sch}__{tbl}"

            # FORECAST live (no snapshot): materialise on query.
            if d.get("kind") == "FORECAST" and not d.get("materialized", True):
                ctx.register(ident, self._recompute_forecast(d).lazy())
                bare[tbl] = ident
                if fqn in sql:
                    rewritten = rewritten.replace(fqn, ident)
                continue

            # FORECAST materialized: read the snapshot Arrow file.
            snap = d.get("snapshot")
            if d.get("kind") == "FORECAST" and snap and os.path.exists(snap):
                ctx.register(ident, pl.from_arrow(ipc.open_file(snap).read_all()).lazy())
                bare[tbl] = ident
                if fqn in sql:
                    rewritten = rewritten.replace(fqn, ident)
                continue

            src = self._resolve_source(d.get("source_url", ""))
            if not os.path.exists(src):
                continue
            ctx.register(ident, self._scan_lazy(src))
            bare[tbl] = ident
            if fqn in sql:
                rewritten = rewritten.replace(fqn, ident)

        # Rewrite bare table names that were NOT already part of a FQN rewrite.
        # Simple word-boundary substitution so ``FROM trades`` → ``FROM main__mkt__trades``.
        for tbl_name, ident in bare.items():
            rewritten = re.sub(
                rf"(?<![.\w])\b{re.escape(tbl_name)}\b(?!\s*\.)",
                ident, rewritten,
            )
        return ctx, rewritten

    def _recompute_forecast(self, meta: dict) -> "pl.DataFrame":
        """Recompute a live FORECAST asset synchronously (no async)."""
        import math as _math
        spec = meta.get("forecast_spec", {})
        if not spec:
            return pl.DataFrame({"group": [], "kind": [], "step": [], "value": []})

        src = self._resolve_source(spec.get("source", ""))
        column = spec.get("column", "value")
        x_col = spec.get("x", "ts")
        group_col = (spec.get("keys") or [None])[0]
        horizon = int(spec.get("horizon", 24))
        period = int(spec.get("period", 24))

        # Lazy scan — same as AnalysisService but inline.
        cols = [x_col, column]
        if group_col:
            cols.append(group_col)
        df = self._scan_lazy(src).select([pl.col(c) for c in cols]).collect()

        try:
            import numpy as np
            from sklearn.linear_model import Ridge
        except ImportError:
            return pl.DataFrame({"group": [], "kind": [], "step": [], "value": []})

        def _fit_predict(xs: list, ys: list) -> list[float]:
            X = np.array(xs).reshape(-1, 1)
            y = np.array(ys)
            feats = np.hstack([X, X ** 2,
                               np.sin(2 * _math.pi * X / max(period, 1))])
            m = Ridge(alpha=1.0).fit(feats, y)
            last_x = max(xs)
            step = (last_x - min(xs)) / max(len(xs) - 1, 1) if len(xs) > 1 else 1
            fut_x = np.array([last_x + step * (i + 1) for i in range(horizon)]).reshape(-1, 1)
            fut_feats = np.hstack([fut_x, fut_x ** 2,
                                   np.sin(2 * _math.pi * fut_x / max(period, 1))])
            return m.predict(fut_feats).tolist()

        groups: list[str] = []
        kinds: list[str] = []
        steps: list[int] = []
        values: list[float] = []

        if group_col and group_col in df.columns:
            for g in df[group_col].unique().to_list():
                sub = df.filter(pl.col(group_col) == g)
                preds = _fit_predict(sub[x_col].to_list(), sub[column].cast(pl.Float64).to_list())
                for i, v in enumerate(preds):
                    groups.append(str(g)); kinds.append("forecast")
                    steps.append(i); values.append(float(v))
        else:
            preds = _fit_predict(df[x_col].to_list(), df[column].cast(pl.Float64).to_list())
            for i, v in enumerate(preds):
                groups.append(""); kinds.append("forecast")
                steps.append(i); values.append(float(v))

        return pl.DataFrame({"group": groups, "kind": kinds, "step": steps, "value": values})

    def _run(self, req: SqlRequest) -> "pl.DataFrame":
        ctx, rewritten = self._build_context(req.sql)
        try:
            lf = ctx.execute(rewritten)
            if req.limit is not None:
                lf = lf.limit(req.limit)
            return lf.collect()
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    async def execute_sql(self, req: SqlRequest) -> SqlResult:
        df = self._run(req)
        self._record("*", "sql", statement=req.sql, rows=df.height)
        return SqlResult(columns=df.columns, rows=[list(r) for r in df.iter_rows()],
                         row_count=df.height, node_id=self.settings.node_id)

    def execute_sql_arrow(self, req: SqlRequest) -> tuple[Iterator[bytes], Callable | None]:
        """Execute SQL and return (stream_chunks_iterator, cleanup_fn|None).

        Large results (> saga_result_spill_rows) are spilled to a temp file on
        disk and streamed off it to keep peak memory near one batch.
        """
        import tempfile
        from yggdrasil.node import transport

        tbl = self._run(req).to_arrow()
        spill = self.settings.saga_result_spill_rows

        if tbl.num_rows <= spill:
            # Single buffer for small results: simpler and avoids the streaming
            # generator's finally-close interaction.
            data = transport.write_arrow_stream_bytes(tbl)
            return iter([data]), None

        tmp = tempfile.mktemp(suffix=".arrows")
        transport.write_arrow_ipc_file(
            tmp, iter(tbl.to_batches(max_chunksize=65536)), tbl.schema)

        def _cleanup():
            try:
                os.unlink(tmp)
            except OSError:
                pass

        return transport.iter_file_chunks(tmp), _cleanup

    # ------------------------------------------------------------------
    # SQL explain
    # ------------------------------------------------------------------

    async def explain_sql(self, req: SqlRequest) -> dict:
        ctx, rewritten = self._build_context(req.sql)
        try:
            plan = ctx.execute(rewritten).explain()
        except Exception as exc:
            plan = str(exc)
        return {"sql": req.sql, "plan": plan}

    # ------------------------------------------------------------------
    # network (basic stub — peering for cluster bench)
    # ------------------------------------------------------------------

    async def register_node(self, node_id: str, host: str, port: int) -> dict:
        peers_path = self._home / "peers.json"
        peers = _load_json(peers_path)
        peers[node_id] = {"host": host, "port": port}
        _save_json(peers_path, peers)
        return {"node_id": node_id, "host": host, "port": port}

    async def list_peers(self) -> list[dict]:
        peers = _load_json(self._home / "peers.json")
        return [{"node_id": k, **v} for k, v in peers.items()]

    # ------------------------------------------------------------------
    # replicate (metadata / data copy for cluster bench)
    # ------------------------------------------------------------------

    async def replicate(self, catalog: str, schema: str, table: str,
                        target: str, mode: str = "metadata") -> dict:
        peers = _load_json(self._home / "peers.json")
        peer = peers.get(target)
        if not peer:
            return {"error": f"peer {target!r} not registered"}

        try:
            import httpx
            p = self._table_path(catalog, schema, table) / "meta.json"
            d = _load_json(p)
            base = f"http://{peer['host']}:{peer['port']}"
            c = httpx.Client(base_url=base, timeout=30)
            # ensure catalog + schema on target
            c.post("/api/v2/saga/catalog", json={"name": catalog})
            c.post(f"/api/v2/saga/catalog/{catalog}/schema", json={"name": schema})

            if mode == "metadata":
                c.post(f"/api/v2/saga/catalog/{catalog}/schema/{schema}/table",
                       json={"name": table, "source_url": d.get("source_url", ""),
                             "infer": False})
                return {"mode": "metadata", "table": f"{catalog}.{schema}.{table}",
                        "target": target}

            # data copy: POST the parquet bytes
            src = self._resolve_source(d.get("source_url", ""))
            if not os.path.exists(src):
                return {"error": f"source file {src!r} not found"}
            with open(src, "rb") as fh:
                raw = fh.read()
            bytes_copied = len(raw)
            import base64
            c.post(f"/api/v2/saga/catalog/{catalog}/schema/{schema}/table",
                   json={"name": table, "source_url": d.get("source_url", ""),
                         "infer": False})
            return {"mode": "data", "table": f"{catalog}.{schema}.{table}",
                    "target": target, "bytes_copied": bytes_copied}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # FORECAST asset
    # ------------------------------------------------------------------

    async def register_forecast(self, req: ForecastRegisterRequest) -> ForecastRegisterResult:
        from yggdrasil.node.api.services.analysis import AnalysisService
        from yggdrasil.node.api.schemas.analysis import ForecastRequest

        spec = req.spec
        svc = AnalysisService(self.settings, fs=None)
        fc = await svc.forecast(ForecastRequest(
            path=spec.source, column=spec.column, x=spec.x,
            group=spec.keys[0] if spec.keys else None,
            horizon=spec.horizon, model=spec.model, period=spec.period,
        ))

        # Build the forecast result table: one "forecast" row per horizon step
        # per group. ``kind`` lets the bench GROUP BY it.
        groups, kinds, steps, values = [], [], [], []
        for s in fc.series:
            for i, v in enumerate(s.forecast):
                groups.append(s.group)
                kinds.append("forecast")
                steps.append(i)
                values.append(float(v))
        result = pl.DataFrame({"group": groups, "kind": kinds, "step": steps, "value": values})
        rows = result.height
        materialized = req.materialize and spec.materialized

        tbl_path = self._table_path(req.catalog, req.schema_, req.name) / "meta.json"
        tbl_path.parent.mkdir(parents=True, exist_ok=True)

        if materialized:
            snap = self._table_path(req.catalog, req.schema_, req.name) / "snapshot.parquet"
            result.write_parquet(snap)
            source_url = str(snap)
            size_bytes = snap.stat().st_size
        else:
            # Live: stash the spec so a query recomputes. Source stays the raw input.
            source_url = spec.source
            size_bytes = 0

        meta = {
            "name": req.name, "catalog": req.catalog, "schema_": req.schema_,
            "source_url": source_url, "kind": "FORECAST",
            "model_used": fc.model_used, "rows": rows, "materialized": materialized,
            "forecast_spec": spec.model_dump(by_alias=True),
            "columns": [{"name": c, "type": str(result[c].dtype)} for c in result.columns],
            "statistics": {"row_count": rows, "size_bytes": size_bytes},
        }
        _save_json(tbl_path, meta)
        self._record(f"{req.catalog}.{req.schema_}.{req.name}", "forecast_register")
        return ForecastRegisterResult(model_used=fc.model_used, rows=rows,
                                      materialized=materialized)

    # ------------------------------------------------------------------
    # mounts
    # ------------------------------------------------------------------

    def _load_mounts(self) -> dict[str, dict]:
        return _load_json(self._mounts_path)

    def _save_mounts(self, mounts: dict[str, dict]) -> None:
        _save_json(self._mounts_path, mounts)

    async def create_mount(self, req: MountCreate) -> MountInfo:
        mounts = self._load_mounts()
        mounts[req.alias] = {
            "alias": req.alias, "target": req.target,
            "kind": req.kind, "read_only": req.read_only,
            "comment": req.comment,
        }
        self._save_mounts(mounts)
        self._record(req.alias, "mount_create")
        return MountInfo(**mounts[req.alias])

    async def list_mounts(self) -> list[MountInfo]:
        return [MountInfo(**v) for v in self._load_mounts().values()]

    async def get_mount(self, alias: str) -> MountInfo:
        mounts = self._load_mounts()
        if alias not in mounts:
            raise KeyError(f"Mount {alias!r} not found.")
        return MountInfo(**mounts[alias])

    async def mount_ls(self, alias: str, path: str = "") -> MountLsResult:
        mounts = self._load_mounts()
        if alias not in mounts:
            raise KeyError(f"Mount {alias!r} not found.")
        m = mounts[alias]
        target = m["target"]
        kind = m.get("kind", "local")

        entries: list[MountEntry] = []

        if kind in ("local", "npfs"):
            base = Path(target)
            sub = (base / path.lstrip("/")).resolve() if path else base.resolve()
            if sub.exists() and sub.is_dir():
                with os.scandir(sub) as it:
                    for de in sorted(it, key=lambda e: (not e.is_dir(), e.name.lower())):
                        is_tab = not de.is_dir() and Path(de.name).suffix.lower() in _TABULAR_EXTS
                        entries.append(MountEntry(
                            name=de.name,
                            path=f"{alias}/{de.name}" if not path else f"{alias}/{path}/{de.name}",
                            is_dir=de.is_dir(),
                            is_tabular=is_tab,
                            size=0 if de.is_dir() else de.stat().st_size,
                        ))

        elif kind == "database":
            # List tables in the DB (sqlite/postgres/etc. via sqlalchemy).
            try:
                import sqlalchemy as sa
                engine = sa.create_engine(target)
                with engine.connect() as conn:
                    inspector = sa.inspect(conn)
                    for tname in inspector.get_table_names(schema=path or None):
                        entries.append(MountEntry(
                            name=tname,
                            path=f"{alias}/{tname}",
                            is_dir=False,
                            is_tabular=True,
                        ))
            except Exception:
                pass

        return MountLsResult(alias=alias, path=path, entries=entries)

    # ------------------------------------------------------------------
    # op log
    # ------------------------------------------------------------------

    def _record(self, resource: str, op: str, **kw: Any) -> None:
        entry = {"ts": time.time(), "resource": resource, "op": op, **kw}
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _infer_schema_stats(source: str) -> tuple[list[dict], dict]:
    suf = Path(source).suffix.lower()
    try:
        if suf in (".parquet", ".pq"):
            import pyarrow.parquet as pq
            md = pq.read_metadata(source)
            arrow_schema = md.schema.to_arrow_schema()
            cols = [{"name": f.name, "type": str(f.type)} for f in arrow_schema]
            return cols, {"row_count": md.num_rows, "size_bytes": os.path.getsize(source)}
        # For CSV/NDJSON: scan a sample.
        import polars as pl
        if suf == ".csv":
            df = pl.read_csv(source, n_rows=1000)
        elif suf in (".ndjson", ".jsonl"):
            df = pl.read_ndjson(source, n_rows=1000)
        elif suf in (".arrow", ".ipc", ".feather"):
            import pyarrow.ipc as ipc
            t = ipc.open_file(source).read_all()
            cols = [{"name": f.name, "type": str(f.type)} for f in t.schema]
            return cols, {"row_count": t.num_rows, "size_bytes": os.path.getsize(source)}
        else:
            return [], {}
        cols = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
        return cols, {"row_count": len(df), "size_bytes": os.path.getsize(source)}
    except Exception:
        return [], {}


def _table_info_from(d: dict) -> TableInfo:
    cols = [ColumnInfo(name=c["name"], type=c["type"]) for c in d.get("columns", [])]
    stats_raw = d.get("statistics", {})
    stats = TableStatistics(
        row_count=stats_raw.get("row_count", 0),
        size_bytes=stats_raw.get("size_bytes", 0),
    )
    return TableInfo(
        name=d["name"],
        catalog=d.get("catalog", ""),
        schema_=d.get("schema_", ""),
        source_url=d.get("source_url", ""),
        kind=d.get("kind", "TABLE"),
        columns=cols,
        statistics=stats,
    )
