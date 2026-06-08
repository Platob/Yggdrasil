"""Saga catalog + SQL engine micro-benchmarks.

Measures the hot paths of the Saga service in-process (no HTTP): catalog CRUD,
schema/statistics inference, SQL execution latency vs row count, Arrow-IPC
result throughput (in-memory vs disk-spilled), and the per-op log overhead.

Usage::

    PYTHONPATH=src python benchmarks/saga/bench_saga.py
    PYTHONPATH=src python benchmarks/saga/bench_saga.py --rows 2000000
"""
from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.saga import (
    CatalogCreate, SchemaCreate, SqlRequest, TableCreate,
)
from yggdrasil.node.api.services.saga import SagaService
from yggdrasil.node.config import Settings


def _svc(home: Path) -> SagaService:
    return SagaService(Settings(node_id="bench", node_home=home,
                                saga_home=home / ".saga", front_home=home))


def _timeit(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0  # ms/op


def _gen_parquet(path: Path, rows: int) -> None:
    import random
    syms = [f"S{i:03d}" for i in range(200)]
    pq.write_table(pa.table({
        "sym": [random.choice(syms) for _ in range(rows)],
        "px": [round(random.uniform(1, 500), 2) for _ in range(rows)],
        "qty": [random.randint(1, 1000) for _ in range(rows)],
        "ts": list(range(rows)),
    }), str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500_000)
    args = ap.parse_args()
    run = asyncio.run

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        svc = _svc(home)
        data = home / "data"
        data.mkdir(parents=True, exist_ok=True)
        src = data / "trades.parquet"
        print(f"generating {args.rows:,}-row parquet …")
        _gen_parquet(src, args.rows)
        size_mb = src.stat().st_size / 1024 / 1024

        print("\n=== Catalog CRUD (ms/op) ===")
        run(svc.create_catalog(CatalogCreate(name="main")))
        run(svc.create_schema("main", SchemaCreate(name="market")))
        print(f"  create_catalog (upsert) : {_timeit(lambda: run(svc.create_catalog(CatalogCreate(name='main'))), 50):.3f}")
        print(f"  list_catalogs           : {_timeit(lambda: run(svc.list_catalogs()), 200):.3f}")

        print("\n=== Register + inference ===")
        t0 = time.perf_counter()
        run(svc.create_table("main", "market", TableCreate(
            name="trades", source_url="data/trades.parquet", infer=True)))
        print(f"  register + infer schema/stats ({args.rows:,} rows): {(time.perf_counter()-t0)*1000:.1f} ms")
        tb = run(svc.get_table("main", "market", "trades")).table
        print(f"  inferred row_count={tb.statistics.row_count:,} cols={len(tb.columns)} size={tb.statistics.size_bytes/1024/1024:.1f} MB")

        print("\n=== SQL latency (ms) ===")
        for label, sql in [
            ("count(*)", "SELECT count(*) AS n FROM main.market.trades"),
            ("filter+limit", "SELECT * FROM main.market.trades WHERE px > 250 LIMIT 1000"),
            ("group_by sum", "SELECT sym, sum(qty) AS q FROM main.market.trades GROUP BY sym ORDER BY q LIMIT 20"),
        ]:
            ms = _timeit(lambda s=sql: run(svc.execute_sql(SqlRequest(sql=s))), 5)
            print(f"  {label:16s}: {ms:8.1f}")

        print("\n=== Arrow IPC result throughput ===")
        for label, spill_rows in [("in-memory", 10_000_000), ("disk-spill", 1)]:
            svc.settings = svc.settings.__class__(  # tweak spill threshold
                node_id="bench", node_home=home, saga_home=home / ".saga",
                front_home=home, saga_result_spill_rows=spill_rows)
            t0 = time.perf_counter()
            stream, cleanup = svc.execute_sql_arrow(SqlRequest(
                sql="SELECT * FROM main.market.trades"))
            nbytes = sum(len(c) for c in stream)
            if cleanup:
                cleanup()
            dt = time.perf_counter() - t0
            print(f"  {label:11s}: {nbytes/1024/1024:7.1f} MB in {dt*1000:7.1f} ms "
                  f"= {nbytes/1024/1024/dt:7.1f} MB/s")

        print("\n=== Op-log append overhead (ms/op) ===")
        print(f"  log.append: {_timeit(lambda: svc._record('main.market.trades', 'query', statement='SELECT 1', rows=1), 100):.3f}")

        print("\n=== Forecast workflow (FORECAST asset) ===")
        from yggdrasil.node.api.schemas.saga import ForecastRegisterRequest, ForecastSpec
        spec = ForecastSpec(source="data/trades.parquet", column="px", x="ts",
                            keys=["sym"], horizon=24, model="auto")
        t0 = time.perf_counter()
        fc = run(svc.register_forecast(ForecastRegisterRequest(
            catalog="main", schema="market", name="px_fc", spec=spec, materialize=True)))
        print(f"  register + materialise ({fc.model_used}, {fc.rows:,} rows): {(time.perf_counter()-t0)*1000:.1f} ms")
        # query the materialised snapshot (fast path) vs a live recompute
        mat_ms = _timeit(lambda: run(svc.execute_sql(SqlRequest(
            sql="SELECT kind, count(*) AS n FROM main.market.px_fc GROUP BY kind"))), 5)
        run(svc.register_forecast(ForecastRegisterRequest(
            catalog="main", schema="market", name="px_live", spec=spec.model_copy(update={"materialized": False}))))
        live_ms = _timeit(lambda: run(svc.execute_sql(SqlRequest(
            sql="SELECT kind, count(*) AS n FROM main.market.px_live GROUP BY kind"))), 3)
        print(f"  query materialised snapshot : {mat_ms:8.1f} ms")
        print(f"  query live (recompute)      : {live_ms:8.1f} ms")
        print(f"  ==> {live_ms/mat_ms:5.1f}x  (snapshot avoids re-fitting on every query)")

        print(f"\nsource file: {size_mb:.1f} MB parquet")


if __name__ == "__main__":
    main()
