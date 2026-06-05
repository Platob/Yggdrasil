"""Frontend-facing Saga benchmark: the HTTP round-trips the UI actually makes.

Spawns a node, registers a table, then times the same calls the /saga page and
the Excel task pane issue — catalog list, SQL (JSON grid) and SQL (Arrow IPC
stream) — and compares JSON-grid vs Arrow-IPC payload size, which is what drives
the editor's perceived reactivity.

Usage::  PYTHONPATH=src python benchmarks/saga/bench_saga_http.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def _percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def _time(fn, n=20):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts), _percentile(ts, 0.95)


def main() -> None:
    port = 8155
    home = tempfile.mkdtemp()
    env = {**os.environ, "YGG_NODE_SEED_DEFAULTS": "0", "YGG_NODE_PORT": str(port),
           "YGG_NODE_HOME": f"{home}/node", "YGG_NODE_SAGA_HOME": f"{home}/saga",
           "YGG_NODE_NODE_ID": "bench"}
    os.makedirs(f"{home}/node/data", exist_ok=True)
    pq.write_table(pa.table({
        "sym": [f"S{i%200}" for i in range(200_000)],
        "px": [float(i % 500) for i in range(200_000)],
        "qty": [i % 1000 for i in range(200_000)],
    }), f"{home}/node/data/trades.parquet")

    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "yggdrasil.node.app:app",
                             "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                            env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{base}/api/ping", timeout=1).status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        c = httpx.Client(base_url=base, timeout=30)
        c.post("/api/v2/saga/catalog", json={"name": "main"})
        c.post("/api/v2/saga/catalog/main/schema", json={"name": "market"})
        c.post("/api/v2/saga/catalog/main/schema/market/table",
               json={"name": "trades", "source_url": "data/trades.parquet"})

        print("=== HTTP latency (ms: mean / p95) ===")
        for label, fn in [
            ("GET  catalog list", lambda: c.get("/api/v2/saga/catalog")),
            ("GET  table list", lambda: c.get("/api/v2/saga/catalog/main/schema/market/table")),
            ("POST sql group_by (JSON)", lambda: c.post("/api/v2/saga/sql", json={
                "sql": "SELECT sym, sum(qty) AS q FROM main.market.trades GROUP BY sym ORDER BY q LIMIT 50"})),
            ("POST explain", lambda: c.post("/api/v2/saga/explain", json={
                "sql": "SELECT * FROM main.market.trades WHERE px > 100"})),
        ]:
            mean, p95 = _time(fn)
            print(f"  {label:30s}: {mean:7.1f} / {p95:7.1f}")

        print("\n=== result payload: JSON grid vs Arrow IPC (10k rows) ===")
        sql = "SELECT sym, px, qty FROM main.market.trades LIMIT 10000"
        rj = c.post("/api/v2/saga/sql", json={"sql": sql, "limit": 10000})
        json_bytes = len(rj.content)
        ra = c.post("/api/v2/saga/sql.arrow", json={"sql": sql})
        arrow_bytes = len(ra.content)
        rows = ipc.open_stream(ra.content).read_all().num_rows
        # Note: sizes are post-gzip-decompression (the middleware gzips both).
        # Arrow's win is zero-copy *typed* decode + streaming + disk spill on
        # large results, not raw size on a tiny narrow result like this one.
        print(f"  JSON  : {json_bytes/1024:8.1f} KB")
        print(f"  Arrow : {arrow_bytes/1024:8.1f} KB  ({rows} rows, typed columnar / zero-copy)")
        c.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
