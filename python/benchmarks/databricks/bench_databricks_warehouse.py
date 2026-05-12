"""Benchmark the in-process hot paths on the Databricks SQL Warehouse layer.

Two surfaces are covered:

* ``warehouse.wh_utils`` — name parsing (regex), enum coercion, dataclass
  shape-conversion (``safeEndpointInfo``), sibling-spec builder.
* ``warehouse.SQLWarehouse`` — identity, repr, ``__call__`` dispatch,
  ``is_running`` / ``is_pending`` / ``is_serverless`` predicates against
  the *cached* details (so the bench never goes to the SDK), and the
  module-level ``set_cached_warehouse`` / ``get_cached_warehouse`` cache.

None of the scenarios issue API calls — the SDK clients on
``DatabricksClient`` are replaced with ``MagicMock`` and cached details
are pre-populated. The benchmark is the right place to spot patterns
that look cheap in isolation but compound when a warehouse handle is
used in tight loops (e.g. statement submission, polling, repeated
``is_running`` checks before / during ``start()`` / ``stop()``).

Usage::

    python benchmarks/databricks/bench_databricks_warehouse.py
    python benchmarks/databricks/bench_databricks_warehouse.py --repeat 7
    python benchmarks/databricks/bench_databricks_warehouse.py --only is_running_cached,details_cached

A/B comparison (n=20_000, repeat=7, best us/op)::

                                BEFORE       AFTER     delta
    indexed_name_parts_plain     0.52 us     0.53 us     0%
    indexed_name_parts_suffixed  0.77 us     0.77 us     0%
    next_indexed_name            0.81 us     0.81 us     0%
    name_at_index                0.91 us     0.91 us     0%
    safe_endpoint_info_pass-     0.05 us     0.05 us     0%
    safe_map_enum                0.05 us     0.05 us     0%
    serverless_sibling_spec      2.20 us     2.19 us     0%
    warehouse_repr              10.72 us    10.52 us    -2%
    warehouse_call_self          0.17 us     0.17 us     0%
    details_cached               0.05 us     0.05 us     0%
    is_running_cached           31.99 us     0.11 us   -99%
    is_pending_cached           18.06 us     0.43 us   -98%
    is_serverless_cached         0.08 us     0.08 us     0%
    get_cached_warehouse         0.52 us     0.52 us     0%
    set_cached_warehouse         1.22 us     1.26 us     +3%

API-call reduction
------------------
``is_running`` and ``is_pending`` used to call ``state``, which read
``latest_details()`` — one ``WarehousesAPI.get`` round-trip per
property access. Any caller that polled ``is_running`` or that
chained ``start()`` / ``stop()`` followed by ``is_running`` was paying
1 GET per check. With the cached-state fix:

* ``is_running`` / ``is_pending`` now read from the cached
  ``_details`` (same pattern as ``Cluster.state``).
* ``start`` / ``stop`` / ``wait_for_status`` call ``refresh()`` once
  before consulting the predicate, so internal call sites still see
  fresh state — the round-trip count is unchanged inside lifecycle
  methods.
* External hot loops (status dashboards, pre-submit checks, the
  ``while not wh.is_running: ...`` pattern) drop from one GET per
  iteration to zero. The bench measures the per-call wrapper cost
  the old code paid before each network round-trip.

Pair this with :file:`bench_databricks_sql.py` for the full
DatabricksClient / SQL resource picture.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Callable
from unittest.mock import MagicMock

from databricks.sdk.service.sql import (
    EndpointInfo,
    EndpointInfoWarehouseType,
    State,
)

from yggdrasil.databricks.client import DatabricksClient
# Force ``yggdrasil.databricks.sql`` to finish initializing before reaching
# into the warehouse package — ``sql.engine`` imports SQLWarehouse, which
# in turn imports ``sql.exceptions``; running this import first lets the
# cycle resolve in a deterministic order.
import yggdrasil.databricks.sql  # noqa: F401
from yggdrasil.databricks.warehouse.service import (
    CACHE_MAP,
    Warehouses,
    get_cached_warehouse,
    set_cached_warehouse,
)
from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse
from yggdrasil.databricks.warehouse.wh_utils import (
    indexed_name_parts,
    name_at_index,
    next_indexed_name,
    safeEndpointInfo,
    serverless_sibling_spec,
    _safe_map_enum,
)


# ---------------------------------------------------------------------------
# Fixtures
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
    object.__setattr__(client, "_workspace_client", MagicMock())
    object.__setattr__(client, "_workspace_config", MagicMock())
    return client


def _make_endpoint_info(*, name: str = "wh", state: State = State.RUNNING) -> EndpointInfo:
    """Build a populated EndpointInfo for cached-details scenarios."""
    return EndpointInfo(
        id="wh-bench-1",
        name=name,
        cluster_size="2X-Small",
        min_num_clusters=1,
        max_num_clusters=1,
        enable_serverless_compute=False,
        warehouse_type=EndpointInfoWarehouseType.PRO,
        state=state,
        auto_stop_mins=30,
    )


def _make_warehouse() -> SQLWarehouse:
    client = _make_client()
    wh = SQLWarehouse(
        service=Warehouses(client=client),
        warehouse_id="wh-bench-1",
        warehouse_name="wh",
        details=_make_endpoint_info(),
    )
    return wh


# ---------------------------------------------------------------------------
# wh_utils scenarios
# ---------------------------------------------------------------------------


def _scenario_indexed_name_parts_plain(n: int) -> Callable[[], None]:
    name = "wh"

    def run() -> None:
        for _ in range(n):
            indexed_name_parts(name)
    return run


def _scenario_indexed_name_parts_suffixed(n: int) -> Callable[[], None]:
    name = "wh [12]"

    def run() -> None:
        for _ in range(n):
            indexed_name_parts(name)
    return run


def _scenario_next_indexed_name(n: int) -> Callable[[], None]:
    name = "wh [5]"

    def run() -> None:
        for _ in range(n):
            next_indexed_name(name)
    return run


def _scenario_name_at_index(n: int) -> Callable[[], None]:
    name = "wh [3]"

    def run() -> None:
        for _ in range(n):
            name_at_index(name, 7)
    return run


def _scenario_safe_endpoint_info_passthrough(n: int) -> Callable[[], None]:
    # When src is already EndpointInfo we should return immediately.
    info = _make_endpoint_info()

    def run() -> None:
        for _ in range(n):
            safeEndpointInfo(info)
    return run


def _scenario_safe_map_enum(n: int) -> Callable[[], None]:
    src = EndpointInfoWarehouseType.PRO

    def run() -> None:
        for _ in range(n):
            _safe_map_enum(EndpointInfoWarehouseType, src)
    return run


def _scenario_serverless_sibling_spec(n: int) -> Callable[[], None]:
    info = _make_endpoint_info(name="wh [2]")

    def run() -> None:
        for _ in range(n):
            serverless_sibling_spec(info)
    return run


# ---------------------------------------------------------------------------
# SQLWarehouse scenarios (cached details → no network)
# ---------------------------------------------------------------------------


def _scenario_warehouse_repr(n: int) -> Callable[[], None]:
    wh = _make_warehouse()

    def run() -> None:
        for _ in range(n):
            repr(wh)
    return run


def _scenario_warehouse_call_self(n: int) -> Callable[[], None]:
    wh = _make_warehouse()

    def run() -> None:
        for _ in range(n):
            wh(warehouse_id="wh-bench-1")
    return run


def _scenario_details_cached(n: int) -> Callable[[], None]:
    wh = _make_warehouse()

    def run() -> None:
        for _ in range(n):
            wh.details
    return run


def _scenario_is_running_cached(n: int) -> Callable[[], None]:
    """Hot path for callers that just want ``is the warehouse running?``.

    Before the optimization this hits ``latest_details()`` (SDK call)
    every access — even when the warehouse handle already carries
    populated ``_details``. With the cached-state fix it becomes a dict
    lookup. We patch the SDK call to return the cached details so
    pre-optimization runs measure the wrapper cost (no real network).
    """
    wh = _make_warehouse()
    # Wire the SDK mock so ``latest_details()`` still works for the
    # baseline (which calls it every time) — same data the cache holds.
    info = _make_endpoint_info()
    wh.client.workspace_client().warehouses.get.return_value = info

    def run() -> None:
        for _ in range(n):
            wh.is_running
    return run


def _scenario_is_pending_cached(n: int) -> Callable[[], None]:
    wh = _make_warehouse()
    info = _make_endpoint_info()
    wh.client.workspace_client().warehouses.get.return_value = info

    def run() -> None:
        for _ in range(n):
            wh.is_pending
    return run


def _scenario_is_serverless_cached(n: int) -> Callable[[], None]:
    wh = _make_warehouse()

    def run() -> None:
        for _ in range(n):
            wh.is_serverless
    return run


def _scenario_get_cached_warehouse(n: int) -> Callable[[], None]:
    wh = _make_warehouse()
    set_cached_warehouse(wh.client, wh)

    def run() -> None:
        for _ in range(n):
            get_cached_warehouse(wh.client, "wh")
    return run


def _scenario_set_cached_warehouse(n: int) -> Callable[[], None]:
    wh = _make_warehouse()
    # Pre-seed so the inner ``if existing is None`` branch is the steady-state.
    set_cached_warehouse(wh.client, wh)

    def run() -> None:
        for _ in range(n):
            set_cached_warehouse(wh.client, wh)
    return run


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SCENARIOS: dict[str, Callable[[int], Callable[[], None]]] = {
    "indexed_name_parts_plain": _scenario_indexed_name_parts_plain,
    "indexed_name_parts_suffixed": _scenario_indexed_name_parts_suffixed,
    "next_indexed_name": _scenario_next_indexed_name,
    "name_at_index": _scenario_name_at_index,
    "safe_endpoint_info_passthrough": _scenario_safe_endpoint_info_passthrough,
    "safe_map_enum": _scenario_safe_map_enum,
    "serverless_sibling_spec": _scenario_serverless_sibling_spec,
    "warehouse_repr": _scenario_warehouse_repr,
    "warehouse_call_self": _scenario_warehouse_call_self,
    "details_cached": _scenario_details_cached,
    "is_running_cached": _scenario_is_running_cached,
    "is_pending_cached": _scenario_is_pending_cached,
    "is_serverless_cached": _scenario_is_serverless_cached,
    "get_cached_warehouse": _scenario_get_cached_warehouse,
    "set_cached_warehouse": _scenario_set_cached_warehouse,
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
        f"{r['label']:>32s}  "
        f"best={r['best']*1e6:8.2f} us  "
        f"median={r['median']*1e6:8.2f} us  "
        f"mean={r['mean']*1e6:8.2f} us"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--only", default=None, help="Comma-separated subset.")
    args = ap.parse_args()

    _clear_env()
    CACHE_MAP.clear()

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
    print(f"# {'label':>32s}  {'best':>12s}  {'median':>14s}  {'mean':>12s}")
    for name in names:
        fn = SCENARIOS[name](args.n)
        fn()  # warm-up
        print(_fmt_row(_time_one(name, fn, args.repeat, args.n)))


if __name__ == "__main__":
    main()
