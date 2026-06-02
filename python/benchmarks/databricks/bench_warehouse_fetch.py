"""Benchmark the live external-link *fetch* path of a SQL-warehouse result.

Unlike the other Databricks benches (which mock the SDK and measure
in-process hot paths), this one drives a **real** SQL warehouse: it runs
a row-generating query, lets the result come back over ``EXTERNAL_LINKS``
as Arrow IPC chunks, and times how fast
:meth:`WarehouseStatementResult._read_arrow_batches` materializes them.

It exists to expose — and then prove the fix for — the download
concurrency of the chunk fetch:

* ``Job.make(fetch_batches, url)`` used to hand the pool a *generator
  function*.  ``Job.run()`` only *constructs* the generator (no I/O), so
  every chunk's HTTP GET + IPC decode actually ran later, serially, in the
  consumer thread.  The ``max_workers`` pool did no real download work.
* The fix materializes each chunk inside the worker, so N chunks download
  concurrently, and resolves external-link URLs from the manifest's chunk
  indices in parallel instead of walking the ``next_chunk`` linked list.

Reported per size: wall time, rows/s, MB/s, chunk count.

A/B (Serverless Starter warehouse, repeat=3, best of N, higher MB/s better)::

       rows       MB  chunks   BEFORE s   AFTER s    BEFORE MB/s  AFTER MB/s  speedup
     200000     17.1       1       1.02      0.79          16.8        21.7     1.3x
    1000000     85.9       8       5.86      1.83          14.6        47.0     3.2x
    3000000    259.9      16      15.87      4.13          16.4        63.0     3.8x

Before, throughput is flat (~15-17 MB/s) no matter how many chunks — the
download was serial. After, MB/s scales with the chunk count because the
chunks transfer in parallel; the single-chunk case still gains from
materializing in the worker (one ``read_all`` vs. a per-batch consumer loop).

Skipped unless ``DATABRICKS_HOST`` (+ credentials) is set.

Usage::

    DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
        python benchmarks/databricks/bench_warehouse_fetch.py
    python benchmarks/databricks/bench_warehouse_fetch.py --repeat 5 --rows 200000,1000000
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult
from yggdrasil.data.options import CastOptions


# Wide-ish rows so a modest row count still spans several external-link
# chunks (the default 32 MiB flush would otherwise buffer a small result
# into a single batch and hide the streaming behaviour).
def _query(rows: int) -> str:
    return (
        "SELECT id, "
        "CAST(id AS STRING) AS s, "
        "rpad(CAST(id AS STRING), 64, 'x') AS pad "
        f"FROM range(0, {rows})"
    )


def _drain(result: WarehouseStatementResult) -> tuple[int, int]:
    """Fully materialize the result's chunk stream; return (rows, bytes)."""
    rows = 0
    nbytes = 0
    options = CastOptions()
    for batch in result._read_arrow_batches(options):
        rows += batch.num_rows
        nbytes += batch.nbytes
    return rows, nbytes


def _bench_size(engine, rows: int, repeat: int) -> dict:
    samples: list[float] = []
    total_rows = 0
    total_bytes = 0
    chunks = 0
    for _ in range(repeat):
        result = engine.execute(_query(rows), engine="api")
        assert isinstance(result, WarehouseStatementResult)
        result.wait()
        chunks = result.manifest.total_chunk_count or 0
        start = time.perf_counter()
        got_rows, got_bytes = _drain(result)
        samples.append(time.perf_counter() - start)
        total_rows, total_bytes = got_rows, got_bytes

    best = min(samples)
    return {
        "rows": total_rows,
        "mb": total_bytes / 1e6,
        "chunks": chunks,
        "best_s": best,
        "median_s": statistics.median(samples),
        "rows_per_s": total_rows / best if best else 0.0,
        "mb_per_s": (total_bytes / 1e6) / best if best else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--rows",
        type=str,
        default="200000,1000000,3000000",
        help="Comma-separated row counts to sweep.",
    )
    args = parser.parse_args()

    if not os.environ.get("DATABRICKS_HOST", "").strip():
        print("DATABRICKS_HOST not set — skipping live warehouse fetch bench.")
        return 0

    client = DatabricksClient()
    engine = client.sql(
        catalog_name=os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev"),
        schema_name=os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "ygg_integration"),
    )

    sizes = [int(x) for x in args.rows.split(",") if x.strip()]
    print(f"\nWarehouse external-link fetch (repeat={args.repeat}, best of N)\n")
    header = f"{'rows':>10} {'MB':>8} {'chunks':>7} {'best_s':>8} {'rows/s':>12} {'MB/s':>8}"
    print(header)
    print("-" * len(header))
    for rows in sizes:
        r = _bench_size(engine, rows, args.repeat)
        print(
            f"{r['rows']:>10} {r['mb']:>8.1f} {r['chunks']:>7} "
            f"{r['best_s']:>8.2f} {r['rows_per_s']:>12,.0f} {r['mb_per_s']:>8.1f}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
