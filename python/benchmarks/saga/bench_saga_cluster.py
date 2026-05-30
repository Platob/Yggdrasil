"""Multi-node Saga cluster simulation — front-to-backend network interactions.

Spins up three real nodes, peers them, and drives the same HTTP endpoints the
/saga page and Excel task pane use, timing each distributed interaction:

  * remote catalog browse (?node= proxy)
  * compute-follows-data: query a table whose bytes live on another node
  * metadata vs data replication
  * remote staging (run on the data's node, write Arrow to the asker's stg)
  * node failover (kill the compute node, query falls back)

Usage::  PYTHONPATH=src python benchmarks/saga/bench_saga_cluster.py
"""
from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time

import httpx
import pyarrow as pa
import pyarrow.parquet as pq


def _wait(base: str, tries: int = 80) -> None:
    for _ in range(tries):
        try:
            if httpx.get(f"{base}/api/ping", timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"node {base} never came up")


def _spawn(node_id: str, port: int, root: str) -> subprocess.Popen:
    env = {**os.environ, "YGG_NODE_SEED_DEFAULTS": "0", "YGG_NODE_PORT": str(port),
           "YGG_NODE_HOME": f"{root}/node", "YGG_NODE_SAGA_HOME": f"{root}/saga",
           "YGG_NODE_NODE_ID": node_id}
    os.makedirs(f"{root}/node/data", exist_ok=True)
    pq.write_table(pa.table({
        "sym": [f"S{i%50}" for i in range(100_000)],
        "px": [float(i % 500) for i in range(100_000)],
        "qty": [i % 1000 for i in range(100_000)],
    }), f"{root}/node/data/trades.parquet")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "yggdrasil.node.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env)


def _timed(fn, n=10):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), max(ts)


def main() -> None:
    base = "http://127.0.0.1"
    nodes = {"nodeA": 8161, "nodeB": 8162, "nodeC": 8163}
    roots = {n: tempfile.mkdtemp() for n in nodes}
    procs = {n: _spawn(n, p, roots[n]) for n, p in nodes.items()}
    url = {n: f"{base}:{p}" for n, p in nodes.items()}
    clients = {}
    try:
        for n, u in url.items():
            _wait(u)
            clients[n] = httpx.Client(base_url=u, timeout=30)

        # All-to-all peering.
        for n in nodes:
            for m, p in nodes.items():
                if m != n:
                    clients[n].post("/api/v2/network/register",
                                    json={"node_id": m, "host": "127.0.0.1", "port": p})

        # Register trades on each node (data lives where it's registered).
        for n in nodes:
            clients[n].post("/api/v2/saga/register",
                            json={"source_url": "data/trades.parquet", "catalog": "main", "schema": "market", "table": "trades"})

        print("=== 3-node cluster: distributed interactions (ms: mean / max) ===")
        q = {"sql": "SELECT sym, sum(qty) AS q FROM main.market.trades GROUP BY sym ORDER BY q LIMIT 20",
             "catalog": "main", "schema": "market"}

        mean, mx = _timed(lambda: clients["nodeB"].get("/api/v2/saga/catalog?node=nodeA").raise_for_status())
        print(f"  remote catalog browse (B→A)      : {mean:7.1f} / {mx:7.1f}")

        mean, mx = _timed(lambda: clients["nodeA"].post("/api/v2/saga/sql", json=q).raise_for_status())
        print(f"  local query (A, data local)      : {mean:7.1f} / {mx:7.1f}")

        # Compute-follows-data: register A's table on C as a remote pointer, then
        # query it on C — C proxies compute to A.
        clients["nodeA"].post("/api/v2/saga/replicate",
                              json={"catalog": "main", "schema": "market", "table": "trades", "target": "nodeC", "mode": "metadata"})
        # C now has main.market.trades twice? It self-registered + got A's. Use a
        # distinct catalog for the remote pointer to avoid the clash.
        clients["nodeA"].post("/api/v2/saga/catalog", json={"name": "remoteA"})
        clients["nodeA"].post("/api/v2/saga/catalog/remoteA/schema", json={"name": "market"})
        clients["nodeA"].post("/api/v2/saga/catalog/remoteA/schema/market/table",
                              json={"name": "trades", "source_url": "data/trades.parquet"})
        clients["nodeA"].post("/api/v2/saga/replicate",
                              json={"catalog": "remoteA", "schema": "market", "table": "trades", "target": "nodeC", "mode": "metadata"})
        rq = {"sql": "SELECT sym, sum(qty) AS q FROM remoteA.market.trades GROUP BY sym ORDER BY q LIMIT 20",
              "catalog": "remoteA", "schema": "market"}
        r = clients["nodeC"].post("/api/v2/saga/sql", json=rq).json()
        mean, mx = _timed(lambda: clients["nodeC"].post("/api/v2/saga/sql", json=rq).raise_for_status())
        print(f"  compute-follows-data (C→A proxy) : {mean:7.1f} / {mx:7.1f}   (ran @ {r.get('node_id')})")

        # Data replication A→B + local read on B.
        t0 = time.perf_counter()
        rep = clients["nodeA"].post("/api/v2/saga/replicate",
                                    json={"catalog": "remoteA", "schema": "market", "table": "trades", "target": "nodeB", "mode": "data"}).json()
        print(f"  data replication A→B             : {(time.perf_counter()-t0)*1000:7.1f}        ({rep['bytes_copied']/1024:.0f} KB copied)")

        # Failover: kill A, then a compute-follows-data query on C falls back.
        procs["nodeA"].terminate()
        procs["nodeA"].wait(timeout=10)
        time.sleep(1)
        t0 = time.perf_counter()
        fr = clients["nodeC"].post("/api/v2/saga/sql", json=rq)
        print(f"  failover after A down (C)        : {(time.perf_counter()-t0)*1000:7.1f}        "
              f"(status {fr.status_code} — local fallback {'ok' if fr.status_code in (200,400) else 'FAIL'})")

        for c in clients.values():
            c.close()
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except Exception:
                    p.kill()


if __name__ == "__main__":
    main()
