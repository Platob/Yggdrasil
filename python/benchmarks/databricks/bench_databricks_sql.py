"""Benchmark the in-process hot paths on the Databricks SQL resource layer.

Targets the dict-like navigation + factory + identity surface that every
Databricks SQL call goes through:

  client.catalogs["main.sales.orders"]                  -> Table
  client.catalogs.parse_location("main.sales.orders")   -> (cat, sch, tbl)
  table.full_name()                                     -> "cat.sch.tbl"
  table.full_name(safe=True)                            -> "`cat`.`sch`.`tbl`"
  Table(service=..., catalog_name=..., schema_name=..., table_name=...)
  parse_databricks_field("STRING")                      -> Field("", StringType())

None of these issue network calls — they're pure string + dataclass +
type-map work — but every workspace-touching method calls one of them on
the boundary (DDL builders, tag operations, cache keys, repr / logging,
URL packing for the FastAPI bridge, etc.).

Usage::

    python benchmarks/databricks/bench_databricks_sql.py
    python benchmarks/databricks/bench_databricks_sql.py --repeat 7
    python benchmarks/databricks/bench_databricks_sql.py --only catalogs_get,table_full_name_safe

A/B comparison (n=20_000, repeat=7, best us/op, lower is better)::

                            BEFORE       AFTER     delta
    catalogs_get_catalog     8.49 us     1.57 us    -82%
    catalogs_get_schema      1.40 us     1.43 us     +2%
    catalogs_get_table      15.00 us    10.96 us    -27%
    catalogs_parse_location  0.49 us     0.50 us     +2%
    schemas_parse_location   0.43 us     0.44 us     +2%
    columns_parse_location   0.52 us     0.51 us     -2%
    table_init               9.30 us     5.26 us    -43%
    table_full_name          0.10 us     0.09 us    -10%
    table_full_name_safe     0.19 us     0.19 us     0%
    table_column_full_name   0.16 us     0.16 us     0%
    safe_table_name_short    0.07 us     0.07 us     0%
    safe_table_name_long    34.49 us    31.37 us     -9%
    quote_ident              0.09 us     0.09 us     0%
    quote_qualified_ident    0.97 us     0.98 us     +1%
    sql_literal_number       0.33 us     0.34 us     +3%
    sql_literal_string       1.00 us     0.96 us     -4%
    parse_field_string       3.78 us     3.90 us     +3%

The wins concentrate on the catalog / table factory and ``Table.__init__``
paths — both used to call ``client.base_url.to_string()`` which parses
a fresh ``URL`` (~6 us) on every invocation:

* ``DatabricksClient.base_url`` now caches the parsed URL after first
  use; the host doesn't change after ``__post_init__``.
* ``Catalogs._cache_key`` / ``Schemas._cache_key`` /
  ``Tables._cache_key`` build their host scope from the dataclass
  ``client.host`` field directly — no URL round-trip needed.
* ``Table.__init__`` benefits indirectly through the cached
  ``base_url`` (used to derive the table URL host).
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Callable
from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.catalog.catalogs import Catalogs
from yggdrasil.databricks.column.columns import Columns
from yggdrasil.databricks.schema.schemas import Schemas
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.tables import Tables
from yggdrasil.databricks.sql.sql_utils import (
    quote_ident,
    quote_qualified_ident,
    safe_table_name,
    sql_literal,
)
from yggdrasil.databricks.sql.types import parse_databricks_field


# ---------------------------------------------------------------------------
# Local fixtures — a client wired with mock SDK handles so the bench never
# crosses the wire. ``DatabricksTestCase`` does the same thing, but pulling
# in unittest from a bench script is heavier than necessary.
# ---------------------------------------------------------------------------


def _clear_env() -> None:
    for key in list(os.environ):
        if key.startswith(("DATABRICKS_", "ARM_", "GOOGLE_")):
            os.environ.pop(key, None)


def _make_client() -> DatabricksClient:
    client = DatabricksClient(
        host="https://bench.databricks.example",
        token="fake-pat-not-a-secret",
        auth_type="pat",
    )
    # Pre-populate the SDK slots so the lazy properties don't try to build
    # a real Config / WorkspaceClient on access.
    object.__setattr__(client, "_workspace_client", MagicMock())
    object.__setattr__(client, "_workspace_config", MagicMock())
    return client


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _scenario_catalogs_get_catalog(n: int) -> Callable[[], None]:
    cats = Catalogs(client=_make_client())

    def run() -> None:
        for _ in range(n):
            cats["main"]
    return run


def _scenario_catalogs_get_schema(n: int) -> Callable[[], None]:
    cats = Catalogs(client=_make_client())

    def run() -> None:
        for _ in range(n):
            cats["main.sales"]
    return run


def _scenario_catalogs_get_table(n: int) -> Callable[[], None]:
    cats = Catalogs(client=_make_client())

    def run() -> None:
        for _ in range(n):
            cats["main.sales.orders"]
    return run


def _scenario_catalogs_parse_location(n: int) -> Callable[[], None]:
    cats = Catalogs(client=_make_client())
    name = "main.sales.orders"

    def run() -> None:
        for _ in range(n):
            cats.parse_location(name)
    return run


def _scenario_schemas_parse_location(n: int) -> Callable[[], None]:
    schemas = Schemas(client=_make_client())
    name = "main.sales"

    def run() -> None:
        for _ in range(n):
            schemas.parse_location(name)
    return run


def _scenario_columns_parse_location(n: int) -> Callable[[], None]:
    cols = Columns(client=_make_client())
    name = "main.sales.orders.price"

    def run() -> None:
        for _ in range(n):
            cols.parse_location(name)
    return run


def _scenario_table_init(n: int) -> Callable[[], None]:
    tables = Tables(client=_make_client())

    def run() -> None:
        for _ in range(n):
            Table(
                service=tables,
                catalog_name="main",
                schema_name="sales",
                table_name="orders",
            )
    return run


def _scenario_table_full_name(n: int) -> Callable[[], None]:
    tables = Tables(client=_make_client())
    t = Table(
        service=tables,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )

    def run() -> None:
        for _ in range(n):
            t.full_name()
    return run


def _scenario_table_full_name_safe(n: int) -> Callable[[], None]:
    tables = Tables(client=_make_client())
    t = Table(
        service=tables,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )

    def run() -> None:
        for _ in range(n):
            t.full_name(safe=True)
    return run


def _scenario_table_column_full_name(n: int) -> Callable[[], None]:
    tables = Tables(client=_make_client())
    t = Table(
        service=tables,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )

    def run() -> None:
        for _ in range(n):
            t.column_full_name("price")
    return run


def _scenario_safe_table_name_short(n: int) -> Callable[[], None]:
    name = "orders_daily_v2"

    def run() -> None:
        for _ in range(n):
            safe_table_name(name)
    return run


def _scenario_safe_table_name_long(n: int) -> Callable[[], None]:
    # Way over the 255-char cap so the truncate+digest path runs.
    name = "_".join(f"layer-{i}" for i in range(80))

    def run() -> None:
        for _ in range(n):
            safe_table_name(name)
    return run


def _scenario_quote_ident(n: int) -> Callable[[], None]:
    name = "orders"

    def run() -> None:
        for _ in range(n):
            quote_ident(name)
    return run


def _scenario_quote_qualified_ident(n: int) -> Callable[[], None]:
    name = "main.sales.orders"

    def run() -> None:
        for _ in range(n):
            quote_qualified_ident(name)
    return run


def _scenario_sql_literal_number(n: int) -> Callable[[], None]:
    value = "42"

    def run() -> None:
        for _ in range(n):
            sql_literal(value)
    return run


def _scenario_sql_literal_string(n: int) -> Callable[[], None]:
    value = "alice's table"

    def run() -> None:
        for _ in range(n):
            sql_literal(value)
    return run


def _scenario_parse_field_string(n: int) -> Callable[[], None]:
    type_text = "STRING"

    def run() -> None:
        for _ in range(n):
            parse_databricks_field(type_text)
    return run


SCENARIOS: dict[str, Callable[[int], Callable[[], None]]] = {
    "catalogs_get_catalog": _scenario_catalogs_get_catalog,
    "catalogs_get_schema": _scenario_catalogs_get_schema,
    "catalogs_get_table": _scenario_catalogs_get_table,
    "catalogs_parse_location": _scenario_catalogs_parse_location,
    "schemas_parse_location": _scenario_schemas_parse_location,
    "columns_parse_location": _scenario_columns_parse_location,
    "table_init": _scenario_table_init,
    "table_full_name": _scenario_table_full_name,
    "table_full_name_safe": _scenario_table_full_name_safe,
    "table_column_full_name": _scenario_table_column_full_name,
    "safe_table_name_short": _scenario_safe_table_name_short,
    "safe_table_name_long": _scenario_safe_table_name_long,
    "quote_ident": _scenario_quote_ident,
    "quote_qualified_ident": _scenario_quote_qualified_ident,
    "sql_literal_number": _scenario_sql_literal_number,
    "sql_literal_string": _scenario_sql_literal_string,
    "parse_field_string": _scenario_parse_field_string,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], repeat: int, n: int) -> dict:
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) / n)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "samples": samples,
    }


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:>26s}  "
        f"best={r['best']*1e6:8.2f} us  "
        f"median={r['median']*1e6:8.2f} us  "
        f"mean={r['mean']*1e6:8.2f} us"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000, help="Calls per sample.")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of scenarios to run.",
    )
    args = ap.parse_args()

    _clear_env()

    names = list(SCENARIOS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise SystemExit(
                f"Unknown scenario(s): {sorted(unknown)}. "
                f"Available: {sorted(names)}"
            )
        names = [n for n in names if n in wanted]

    print(f"# n={args.n} repeat={args.repeat}")
    print(f"# {'label':>26s}  {'best':>12s}  {'median':>14s}  {'mean':>12s}")
    for name in names:
        fn = SCENARIOS[name](args.n)
        # Warm-up sample so the first timed pass isn't biased by lazy imports
        # / one-shot caches inside the resource layer.
        fn()
        print(_fmt_row(_time_one(name, fn, args.repeat, args.n)))


if __name__ == "__main__":
    main()
