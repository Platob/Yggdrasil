"""Saga client — drive the distributed SQL engine as resources-as-code.

The mirror of ``fn.py`` for data instead of compute: where ``@function``
registers code and returns a callable handle, these helpers register catalog
assets (mounts, tables, catalogs, forecasts) and run SQL, talking to a node's
``/api/v2/saga`` API over plain stdlib HTTP — no server stack, no pydantic, no
extra deps to import.

Usage::

    from yggdrasil.node import sql, mount, register, forecast

    # Run SQL against a node-rooted file, no registration needed
    res = sql("SELECT category, sum(amount) FROM 'sales/2024.parquet' GROUP BY 1")
    for row in res:            # iterate dict rows
        print(row)
    res.rows                   # list[list]
    res.columns                # ["category", "sum(amount)"]
    df = res.to_polars()       # optional, if polars is installed

    # Mount a Databricks volume / S3 prefix / remote node under a short alias,
    # then query through it — the alias expands everywhere a source resolves.
    vol = mount("prod_vol", "/Volumes/main/sales/raw")
    vol.ls("2024")                                   # lazy browse
    vol.sql("SELECT count(*) FROM 'prod_vol/2024/jan.parquet'")

    # One-shot register a file as a catalog table (infers schema + stats)
    register("prod_vol/2024/jan.parquet", catalog="main", schema="default")

    # Scope to a catalog so unqualified names resolve
    main = catalog("main", dialect="duckdb")
    main.sql("SELECT * FROM jan LIMIT 10")

    # Register a forecast workflow and query its history+forecast view
    forecast("sales_fc", source="jan", column="amount", x="date", horizon=24)
    sql("SELECT * FROM main.default.sales_fc")

Every helper accepts ``node_url=`` to target a remote node; it defaults to the
local node (``BOT_API_URL`` or ``http://127.0.0.1:$YGG_NODE_PORT``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator

__all__ = [
    "sql", "mount", "register", "table", "catalog", "forecast", "finance", "overview",
    "mounts", "SqlResult", "Mount", "Catalog",
]


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, no external deps) — self-contained like path.py
# ---------------------------------------------------------------------------


def _base_url(node_url: str | None = None) -> str:
    """Resolve a node's base URL: explicit arg, else ``BOT_API_URL``, else the
    local node from ``YGG_NODE_PORT`` (default 8100)."""
    if node_url:
        return node_url.rstrip("/")
    env = os.environ.get("BOT_API_URL")
    if env:
        return env.rstrip("/")
    port = os.environ.get("YGG_NODE_PORT", "8100")
    return f"http://127.0.0.1:{port}"


def _request(method: str, url: str, data: dict | None = None) -> Any:
    """Issue an HTTP request, returning parsed JSON (or {} on 204). Surfaces the
    server's error ``detail`` as a RuntimeError so failures read cleanly."""
    body = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"saga {method} {url} failed [{exc.code}]: {detail}") from None


def _q(value: str) -> str:
    return urllib.parse.quote(value, safe="")


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


class SqlResult:
    """A SQL query result: a column header + materialized rows, plus the plan.

    Iterating yields dict rows; ``.rows`` is the raw row matrix. ``to_polars`` /
    ``to_arrow`` / ``to_pandas`` convert on demand when the library is present.
    """

    __slots__ = ("columns", "column_types", "rows", "row_count", "truncated",
                 "elapsed_ms", "plan_sql", "referenced_tables", "node_id")

    def __init__(self, payload: dict) -> None:
        cols = payload.get("columns", [])
        self.columns: list[str] = [c["name"] for c in cols]
        self.column_types: list[tuple[str, str]] = [(c["name"], c.get("dtype", "")) for c in cols]
        self.rows: list[list[Any]] = payload.get("rows", [])
        self.row_count: int = payload.get("row_count", len(self.rows))
        self.truncated: bool = payload.get("truncated", False)
        self.elapsed_ms: float = payload.get("elapsed_ms", 0.0)
        self.plan_sql: str = payload.get("plan_sql", "")
        self.referenced_tables: list[str] = payload.get("referenced_tables", [])
        self.node_id: str = payload.get("node_id", "")

    def __iter__(self) -> Iterator[dict]:
        for r in self.rows:
            yield dict(zip(self.columns, r))

    def dicts(self) -> list[dict]:
        return list(self)

    def __len__(self) -> int:
        return len(self.rows)

    def to_polars(self):
        import polars as pl
        return pl.DataFrame(self.rows, schema=self.columns, orient="row")

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_arrow(self):
        return self.to_polars().to_arrow()

    def __repr__(self) -> str:
        more = "+" if self.truncated else ""
        return (f"SqlResult({self.row_count}{more} rows, "
                f"{len(self.columns)} cols {self.columns}, {self.elapsed_ms:.0f}ms)")


def sql(query: str, *, catalog: str | None = None, schema: str | None = None,
        dialect: str | None = None, limit: int | None = None,
        node: str | None = None, node_url: str | None = None) -> SqlResult:
    """Run a SQL query on a node and return the result.

    Unqualified table names resolve against ``catalog``/``schema``; quoted
    file paths and mount aliases (``'sales/x.parquet'``, ``'prod_vol/...'``)
    resolve directly. ``node`` proxies the run to a peer where the data lives.
    """
    payload = {"sql": query, "dialect": dialect, "catalog": catalog,
               "schema": schema, "node": node, "limit": limit}
    resp = _request("POST", f"{_base_url(node_url)}/api/v2/saga/sql", payload)
    return SqlResult(resp)


# ---------------------------------------------------------------------------
# Mounts — named aliases over path objects, queryable + browsable
# ---------------------------------------------------------------------------


class Mount:
    """Handle to a registered mount. ``ls`` browses lazily; ``sql`` runs a
    query (the alias resolves inside the SQL on the same node)."""

    __slots__ = ("name", "target", "kind", "_node_url")

    def __init__(self, name: str, target: str, kind: str, node_url: str | None) -> None:
        self.name = name
        self.target = target
        self.kind = kind
        self._node_url = node_url

    def ls(self, subpath: str = "") -> list[dict]:
        """List children under the mount; each entry flags ``is_tabular``."""
        url = f"{_base_url(self._node_url)}/api/v2/saga/mount/{_q(self.name)}/ls?subpath={_q(subpath)}"
        return _request("GET", url).get("entries", [])

    def sql(self, query: str, **kwargs: Any) -> SqlResult:
        kwargs.setdefault("node_url", self._node_url)
        return sql(query, **kwargs)

    def __truediv__(self, sub: str) -> str:
        """``mount / "2024/jan.parquet"`` → the alias-qualified ref for SQL."""
        return f"{self.name}/{sub.lstrip('/')}"

    def __repr__(self) -> str:
        return f"Mount({self.name!r} -> {self.target!r}, kind={self.kind!r})"


def mount(name: str, target: str, *, comment: str = "", read_only: bool = True,
          node_url: str | None = None) -> Mount:
    """Register (upsert by alias) a mount: a named base path/URL the SQL engine
    and file browser expand on demand — a Databricks volume, an S3 prefix, an
    ``npfs://`` node path, a node-home folder, or a live-database connection URI
    (``postgres://`` / ``mysql://`` / ``sqlite://`` / ``mssql://`` …, queried via
    connectorx as ``SELECT ... FROM 'alias/schema.table'``)."""
    payload = {"name": name, "target": target, "comment": comment,
               "read_only": read_only}
    resp = _request("POST", f"{_base_url(node_url)}/api/v2/saga/mount", payload)
    m = resp["mount"]
    return Mount(m["name"], m["target"], m.get("kind", "local"), node_url)


def mounts(node_url: str | None = None) -> list[Mount]:
    """List every registered mount on the node."""
    resp = _request("GET", f"{_base_url(node_url)}/api/v2/saga/mount")
    return [Mount(m["name"], m["target"], m.get("kind", "local"), node_url)
            for m in resp.get("mounts", [])]


def overview(node_url: str | None = None) -> dict:
    """Catalog-wide monitoring rollup: asset counts by kind, totals (rows,
    bytes, ops), a recent-activity feed across all assets, and largest/busiest
    leaderboards. The one call behind the Saga management dashboard."""
    return _request("GET", f"{_base_url(node_url)}/api/v2/saga/overview")


# ---------------------------------------------------------------------------
# Tables / catalogs
# ---------------------------------------------------------------------------


def register(source_url: str, *, catalog: str = "main", schema: str = "default",
             table: str | None = None, node: str | None = None,
             dialect: str | None = None, node_url: str | None = None) -> dict:
    """One-shot: ensure the catalog + schema exist, register ``source_url`` as a
    table (name inferred from the filename), infer its schema + statistics.
    Returns the table entry. ``source_url`` may be a mount ref (``alias/sub``)."""
    payload = {"source_url": source_url, "catalog": catalog, "schema": schema,
               "table": table, "node": node, "dialect": dialect}
    resp = _request("POST", f"{_base_url(node_url)}/api/v2/saga/register", payload)
    return resp["table"]


def table(catalog: str, schema: str, name: str, *, source_url: str = "",
          object_type: str = "TABLE", definition: str = "", node: str | None = None,
          infer: bool = True, node_url: str | None = None) -> dict:
    """Register (upsert) a table under an existing catalog/schema. Use
    ``object_type="VIEW"`` with ``definition`` (SQL) for a view. Returns the
    table entry."""
    payload = {"name": name, "source_url": source_url, "object_type": object_type,
               "definition": definition, "node": node, "infer": infer}
    url = f"{_base_url(node_url)}/api/v2/saga/catalog/{_q(catalog)}/schema/{_q(schema)}/table"
    return _request("POST", url, payload)["table"]


class Catalog:
    """Handle to a catalog: scopes ``sql`` and table registration to it so
    unqualified names resolve, without repeating catalog/schema everywhere."""

    __slots__ = ("name", "dialect", "schema", "_node_url")

    def __init__(self, name: str, dialect: str | None, schema: str, node_url: str | None) -> None:
        self.name = name
        self.dialect = dialect
        self.schema = schema
        self._node_url = node_url

    def using(self, schema: str) -> "Catalog":
        """Return a handle scoped to a different default schema."""
        return Catalog(self.name, self.dialect, schema, self._node_url)

    def sql(self, query: str, **kwargs: Any) -> SqlResult:
        kwargs.setdefault("catalog", self.name)
        kwargs.setdefault("schema", self.schema)
        kwargs.setdefault("dialect", self.dialect)
        kwargs.setdefault("node_url", self._node_url)
        return sql(query, **kwargs)

    def register(self, source_url: str, *, table: str | None = None, **kwargs: Any) -> dict:
        kwargs.setdefault("catalog", self.name)
        kwargs.setdefault("schema", self.schema)
        kwargs.setdefault("node_url", self._node_url)
        return register(source_url, table=table, **kwargs)

    def table(self, name: str, **kwargs: Any) -> dict:
        kwargs.setdefault("node_url", self._node_url)
        return table(self.name, self.schema, name, **kwargs)

    def __repr__(self) -> str:
        return f"Catalog({self.name!r}, schema={self.schema!r}, dialect={self.dialect!r})"


def catalog(name: str, *, dialect: str | None = None, comment: str = "",
            schema: str = "default", node_url: str | None = None) -> Catalog:
    """Register (upsert) a catalog and ensure its default schema exists. Returns
    a Catalog handle that scopes ``sql``/``register`` to it."""
    base = _base_url(node_url)
    _request("POST", f"{base}/api/v2/saga/catalog",
             {"name": name, "dialect": dialect, "comment": comment})
    # Ensure the default schema exists so unqualified names resolve immediately.
    try:
        _request("POST", f"{base}/api/v2/saga/catalog/{_q(name)}/schema", {"name": schema})
    except RuntimeError:
        pass  # already exists
    return Catalog(name, dialect, schema, node_url)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def forecast(name: str, *, source: str, column: str, x: str | None = None,
             keys: list[str] | None = None, horizon: int = 24, model: str = "auto",
             period: int | None = None, agg: str = "mean", catalog: str = "main",
             schema: str = "default", materialize: bool = False,
             node: str | None = None, node_url: str | None = None) -> dict:
    """Register a forecast workflow as a catalog asset and return its result.

    ``source`` is a registered table name or a file/mount path; ``column`` is the
    value to forecast over the time/order column ``x`` (row index if omitted).
    ``keys`` forecasts one series per key (e.g. per ticker). The asset is then
    queryable like a view: ``SELECT * FROM <catalog>.<schema>.<name>``.
    """
    payload = {
        "catalog": catalog, "schema": schema, "name": name, "node": node,
        "materialize": materialize,
        "spec": {"source": source, "column": column, "x": x,
                 "keys": list(keys or []), "horizon": horizon, "model": model,
                 "period": period, "agg": agg, "materialized": materialize},
    }
    return _request("POST", f"{_base_url(node_url)}/api/v2/saga/forecast", payload)


# ---------------------------------------------------------------------------
# Finance — risk/return analytics over a price series
# ---------------------------------------------------------------------------


def finance(path: str, column: str, *, order_by: str | None = None, window: int = 20,
            limit: int = 2000, periods_per_year: int = 252, risk_free: float = 0.0,
            node: str | None = None, node_url: str | None = None) -> dict:
    """Risk/return analytics over a price series in a node-rooted (or mount)
    file. Returns per-row series (value, pct_change, cum_return, rolling
    mean/vol, EMA, drawdown) plus a ``metrics`` summary (total return, CAGR,
    annualized return/vol, Sharpe, Sortino, max drawdown, Calmar).

    ``periods_per_year`` annualizes the scalar metrics (252 daily / 52 weekly /
    12 monthly); ``risk_free`` is the per-year rate for Sharpe/Sortino.
    """
    payload = {"path": path, "column": column, "order_by": order_by,
               "window": window, "limit": limit,
               "periods_per_year": periods_per_year, "risk_free": risk_free}
    # The analysis endpoints take ?node= as a query param, not in the body.
    url = f"{_base_url(node_url)}/api/v2/analysis/finance"
    if node:
        url += f"?node={_q(node)}"
    return _request("POST", url, payload)
