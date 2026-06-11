"""Saga — a lightweight Unity-Catalog-like SQL layer over node-local files.

A three-level namespace (catalog → schema → table) where each table points at a
source file (parquet/arrow). Metadata persists as JSON under ``saga_home``; the
data never moves. SQL runs on DuckDB when installed (it scans the source files
directly) and falls back to polars' SQL context otherwise — both register every
known table as ``catalog.schema.table`` so a query names tables the Unity way.

Results stream as Arrow IPC: small results encode in memory and iterate as
chunks; results past ``saga_result_spill_rows`` spill to a temp file and stream
off disk (with a cleanup callback) so a huge SELECT never lands in RAM whole.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator

import polars as pl
import pyarrow as pa
from pydantic import BaseModel

from ... import transport
from ..schemas.saga import (
    CatalogCreate,
    ForecastRegisterRequest,
    SchemaCreate,
    SqlRequest,
    TableCreate,
)

try:
    import duckdb as _duckdb
except ImportError:
    _duckdb = None


# --- response models -------------------------------------------------------

class TableStatistics(BaseModel):
    row_count: int
    col_count: int
    size_bytes: int


class TableColumn(BaseModel):
    name: str
    type: str


class TableInfo(BaseModel):
    catalog: str
    schema_name: str
    name: str
    source_url: str
    kind: str = "table"
    comment: str | None = None
    columns: list[TableColumn] = []
    statistics: TableStatistics = TableStatistics(row_count=0, col_count=0, size_bytes=0)


class TableResponse(BaseModel):
    table: TableInfo


class SqlResult(BaseModel):
    columns: list[str]
    rows: list[list]
    row_count: int
    node_id: str | None = None


class ForecastRegisterResult(BaseModel):
    name: str
    kind: str
    model_used: str
    rows: int
    materialized: bool


# --- service ---------------------------------------------------------------

_FQN = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


class SagaService:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._saga_home = Path(settings.saga_home)
        self._saga_home.mkdir(parents=True, exist_ok=True)
        self._node_home = Path(settings.node_home)
        self._meta: dict = self._load_meta()
        self._log: list[dict] = []

    # -- metadata persistence ----------------------------------------------

    def _meta_path(self) -> Path:
        return self._saga_home / "meta.json"

    def _load_meta(self) -> dict:
        p = self._saga_home / "meta.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

    def _save_meta(self) -> None:
        self._meta_path().write_text(json.dumps(self._meta, separators=(",", ":")))

    def _record(self, table_fqn: str, action: str, **kwargs) -> None:
        self._log.append({"ts": time.time(), "table": table_fqn, "action": action, **kwargs})

    # -- catalogs / schemas / tables ---------------------------------------

    async def create_catalog(self, req: CatalogCreate):
        cat = self._meta.setdefault(req.name, {"_comment": req.comment, "schemas": {}})
        if req.comment is not None:
            cat["_comment"] = req.comment
        self._save_meta()
        self._record(req.name, "create_catalog")
        return {"name": req.name, "comment": cat.get("_comment")}

    async def list_catalogs(self):
        return [{"name": name, "comment": cat.get("_comment"),
                 "schema_count": len(cat.get("schemas", {}))}
                for name, cat in self._meta.items()]

    async def create_schema(self, catalog: str, req: SchemaCreate):
        cat = self._meta.get(catalog)
        if cat is None:
            raise KeyError(
                f"No catalog {catalog!r}. Known: {sorted(self._meta)}. "
                f"Create it first with create_catalog."
            )
        sch = cat["schemas"].setdefault(req.name, {"_comment": req.comment, "tables": {}})
        if req.comment is not None:
            sch["_comment"] = req.comment
        self._save_meta()
        self._record(f"{catalog}.{req.name}", "create_schema")
        return {"catalog": catalog, "name": req.name}

    async def list_schemas(self, catalog: str):
        cat = self._meta.get(catalog)
        if cat is None:
            raise KeyError(f"No catalog {catalog!r}. Known: {sorted(self._meta)}.")
        return [{"name": n, "comment": s.get("_comment"), "table_count": len(s.get("tables", {}))}
                for n, s in cat["schemas"].items()]

    async def create_table(self, catalog: str, schema: str, req: TableCreate):
        sch = self._schema(catalog, schema, create=True)
        entry = {
            "name": req.name,
            "source_url": req.source_url,
            "kind": "table",
            "comment": req.comment,
        }
        if req.infer:
            entry["statistics"], entry["columns"] = self._infer(req.source_url)
        sch["tables"][req.name] = entry
        self._save_meta()
        self._record(f"{catalog}.{schema}.{req.name}", "create_table", infer=req.infer)
        return await self.get_table(catalog, schema, req.name)

    async def list_tables(self, catalog: str, schema: str):
        sch = self._schema(catalog, schema)
        return [{"name": n, "source_url": t["source_url"], "kind": t.get("kind", "table")}
                for n, t in sch["tables"].items()]

    async def get_table(self, catalog: str, schema: str, table: str) -> TableResponse:
        sch = self._schema(catalog, schema)
        t = sch["tables"].get(table)
        if t is None:
            raise KeyError(
                f"No table {catalog}.{schema}.{table!r}. "
                f"Known in {catalog}.{schema}: {sorted(sch['tables'])}."
            )
        stats = t.get("statistics")
        cols = t.get("columns")
        if stats is None or cols is None:
            stats, cols = self._infer(t["source_url"])
            t["statistics"], t["columns"] = stats, cols
        return TableResponse(table=TableInfo(
            catalog=catalog,
            schema_name=schema,
            name=table,
            source_url=t["source_url"],
            kind=t.get("kind", "table"),
            comment=t.get("comment"),
            columns=[TableColumn(**c) for c in cols],
            statistics=TableStatistics(**stats),
        ))

    def _schema(self, catalog: str, schema: str, create: bool = False) -> dict:
        cat = self._meta.get(catalog)
        if cat is None:
            if not create:
                raise KeyError(f"No catalog {catalog!r}. Known: {sorted(self._meta)}.")
            cat = self._meta.setdefault(catalog, {"_comment": None, "schemas": {}})
        sch = cat["schemas"].get(schema)
        if sch is None:
            if not create:
                raise KeyError(
                    f"No schema {catalog}.{schema!r}. Known: {sorted(cat['schemas'])}."
                )
            sch = cat["schemas"].setdefault(schema, {"_comment": None, "tables": {}})
        return sch

    def _resolve_source(self, source_url: str) -> Path:
        # Sources are relative to node_home; an absolute path is taken as-is.
        p = Path(source_url)
        return p if p.is_absolute() else self._node_home / source_url

    def _infer(self, source_url: str) -> tuple[dict, list[dict]]:
        path = self._resolve_source(source_url)
        lf = _scan(path)
        schema = lf.collect_schema()
        cols = [{"name": n, "type": str(t)} for n, t in schema.items()]
        rows = lf.select(pl.len()).collect().item()
        size = path.stat().st_size if path.exists() else 0
        return ({"row_count": int(rows), "col_count": len(cols), "size_bytes": size}, cols)

    # -- SQL ---------------------------------------------------------------

    def _all_tables(self) -> dict[str, Path]:
        """Map ``catalog.schema.table`` (and bare ``table``) to its source path."""
        out: dict[str, Path] = {}
        for cat, c in self._meta.items():
            for sch, s in c.get("schemas", {}).items():
                for name, t in s.get("tables", {}).items():
                    if t.get("kind") == "forecast" and t.get("materialized_path"):
                        path = Path(t["materialized_path"])
                    else:
                        path = self._resolve_source(t["source_url"])
                    out[f"{cat}.{sch}.{name}"] = path
                    out.setdefault(name, path)
        return out

    def _query_table(self, req: SqlRequest) -> pa.Table:
        tables = self._all_tables()
        referenced = {
            f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            for m in _FQN.finditer(req.sql)
            if f"{m.group(1)}.{m.group(2)}.{m.group(3)}" in tables
        }
        live_forecasts = self._live_forecasts_in(req.sql)

        if _duckdb is not None:
            con = _duckdb.connect()
            try:
                for fqn in referenced:
                    view = fqn.replace(".", "__")
                    con.execute(f'CREATE VIEW "{view}" AS SELECT * FROM read_parquet(?)'
                                if str(tables[fqn]).endswith((".parquet", ".pq"))
                                else f'CREATE VIEW "{view}" AS SELECT * FROM "{tables[fqn]}"',
                                [str(tables[fqn])])
                for fqn, df in live_forecasts.items():
                    con.register(fqn.replace(".", "__"), df.to_arrow())
                sql = _rewrite_fqns(req.sql, referenced | set(live_forecasts))
                if req.limit is not None and not re.search(r"\blimit\b", sql, re.I):
                    sql = f"SELECT * FROM ({sql}) _q LIMIT {int(req.limit)}"
                return con.execute(sql).fetch_arrow_table()
            finally:
                con.close()

        # Polars SQL fallback.
        ctx = pl.SQLContext()
        for fqn in referenced:
            ctx.register(fqn.replace(".", "__"), _scan(tables[fqn]))
        for fqn, df in live_forecasts.items():
            ctx.register(fqn.replace(".", "__"), df.lazy())
        sql = _rewrite_fqns(req.sql, referenced | set(live_forecasts))
        lf = ctx.execute(sql)
        if req.limit is not None:
            lf = lf.limit(req.limit)
        return lf.collect().to_arrow()

    def _live_forecasts_in(self, sql: str) -> dict[str, pl.DataFrame]:
        """Recompute any non-materialized forecast table referenced in ``sql``."""
        out: dict[str, pl.DataFrame] = {}
        for cat, c in self._meta.items():
            for sch, s in c.get("schemas", {}).items():
                for name, t in s.get("tables", {}).items():
                    if t.get("kind") != "forecast" or t.get("materialized"):
                        continue
                    fqn = f"{cat}.{sch}.{name}"
                    if fqn in sql or name in sql:
                        out[fqn] = self._compute_forecast(t["spec"])[1]
        return out

    async def execute_sql(self, req: SqlRequest) -> SqlResult:
        tbl = self._query_table(req)
        self._record("", "query", statement=req.sql, rows=tbl.num_rows)
        return SqlResult(
            columns=tbl.column_names,
            rows=[list(r.values()) for r in tbl.to_pylist()],
            row_count=tbl.num_rows,
            node_id=self.settings.node_id,
        )

    def execute_sql_arrow(self, req: SqlRequest) -> tuple[Iterator[bytes], Callable | None]:
        tbl = self._query_table(req)
        self._record("", "query.arrow", statement=req.sql, rows=tbl.num_rows)
        spill = getattr(self.settings, "saga_result_spill_rows", 1_000_000)
        if tbl.num_rows < spill:
            # Small enough: encode once, hand back a single-chunk iterator.
            return transport.write_arrow_stream(tbl), None
        # Spill to a temp file batch-by-batch, then stream it off disk.
        fd, tmp = tempfile.mkstemp(suffix=".arrows", dir=str(self._saga_home))
        os.close(fd)
        transport.write_arrow_ipc_file(tmp, iter(tbl.to_batches(max_chunksize=65536)), tbl.schema)
        return transport.iter_file_chunks(tmp), lambda: os.unlink(tmp)

    def explain(self, req: SqlRequest) -> dict:
        tables = self._all_tables()
        referenced = sorted({
            f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            for m in _FQN.finditer(req.sql)
            if f"{m.group(1)}.{m.group(2)}.{m.group(3)}" in tables
        })
        plan = None
        try:
            for fqn in referenced:
                pass
            plan = pl.SQLContext({
                fqn.replace(".", "__"): _scan(tables[fqn]) for fqn in referenced
            }).execute(_rewrite_fqns(req.sql, set(referenced))).explain()
        except Exception as exc:
            plan = f"<unavailable: {type(exc).__name__}: {exc}>"
        return {"sql": req.sql, "tables": referenced, "engine": "duckdb" if _duckdb else "polars", "plan": plan}

    # -- forecast assets ---------------------------------------------------

    def _compute_forecast(self, spec: dict) -> tuple[str, pl.DataFrame]:
        """Fit + roll a forecast for a saga FORECAST asset.

        Returns ``(model_used, frame)`` where frame has columns
        ``[*keys, x, yhat, kind]`` — ``kind`` is "actual" then "forecast".
        """
        from .analysis import _forecast_one, _pick_model

        path = self._resolve_source(spec["source"])
        keys = spec.get("keys", [])
        cols = list(dict.fromkeys([spec["column"], spec["x"], *keys]))
        df = _scan(path).select(cols).sort(spec["x"]).collect()
        fit, model_used = _pick_model(spec.get("model", "auto"))
        period = spec.get("period")
        horizon = spec.get("horizon", 24)

        frames = []
        groups = ([(tuple(str(v) for v in k), g) for k, g in df.group_by(keys, maintain_order=True)]
                  if keys else [((), df)])
        for key, g in groups:
            y = g.get_column(spec["column"]).to_numpy()
            x = g.get_column(spec["x"]).to_numpy()
            _, fut_x, fut_y, _ = _forecast_one(x, y, horizon, period, fit)
            keymap = dict(zip(keys, key))
            actual = pl.DataFrame({
                **{k: [v] * len(x) for k, v in keymap.items()},
                spec["x"]: x.tolist(),
                "yhat": y.tolist(),
                "kind": ["actual"] * len(x),
            })
            forecast = pl.DataFrame({
                **{k: [v] * len(fut_x) for k, v in keymap.items()},
                spec["x"]: fut_x.tolist(),
                "yhat": fut_y.tolist(),
                "kind": ["forecast"] * len(fut_x),
            })
            frames.append(pl.concat([actual, forecast], how="vertical_relaxed"))
        out = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
        return model_used, out

    async def register_forecast(self, req: ForecastRegisterRequest) -> ForecastRegisterResult:
        sch = self._schema(req.catalog, req.schema_, create=True)
        spec = req.spec.model_dump()
        materialize = req.materialize and req.spec.materialized
        model_used = "auto"
        rows = 0
        entry: dict = {
            "name": req.name,
            "source_url": req.spec.source,
            "kind": "forecast",
            "spec": spec,
            "materialized": materialize,
        }
        if materialize:
            model_used, frame = self._compute_forecast(spec)
            rows = frame.height
            out_path = self._saga_home / f"_fc_{req.catalog}_{req.schema_}_{req.name}.parquet"
            frame.write_parquet(str(out_path))
            entry["materialized_path"] = str(out_path)
            entry["model_used"] = model_used
        else:
            # Live: store the spec only; queries recompute on the fly. Record
            # which backend would be used so the response is still informative.
            from .analysis import _pick_model
            _, model_used = _pick_model(req.spec.model)
            entry["model_used"] = model_used
        sch["tables"][req.name] = entry
        self._save_meta()
        self._record(f"{req.catalog}.{req.schema_}.{req.name}", "register_forecast",
                     materialized=materialize)
        return ForecastRegisterResult(
            name=req.name, kind="forecast", model_used=model_used,
            rows=rows, materialized=materialize,
        )


# --- module helpers --------------------------------------------------------

def _scan(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.scan_parquet(str(path))
    if suffix in (".arrow", ".arrows", ".ipc", ".feather"):
        return pl.scan_ipc(str(path))
    if suffix == ".csv":
        return pl.scan_csv(str(path))
    raise ValueError(
        f"Can't scan {path.name!r}: unsupported extension {suffix!r}. "
        f"Expected .parquet/.arrow/.csv."
    )


def _rewrite_fqns(sql: str, fqns: set[str]) -> str:
    """Rewrite ``catalog.schema.table`` references to the flattened view name.

    DuckDB/polars don't model a 3-level namespace here; we register each table
    under ``catalog__schema__table`` and rewrite the SQL to match.
    """
    out = sql
    for fqn in sorted(fqns, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(fqn)}\b", fqn.replace(".", "__"), out)
    return out
