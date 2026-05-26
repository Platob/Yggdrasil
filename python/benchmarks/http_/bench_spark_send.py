"""Spark send_many benchmarks.

Measures the Spark fan-out path: scatter requests to workers via
mapInArrow, collect responses, with and without cache.
"""
from __future__ import annotations

import argparse
import http.server
import json
import statistics
import threading
import time

import pytest

try:
    from pyspark.sql import SparkSession
except ImportError:
    print("pyspark not available, skipping")
    raise SystemExit(0)

from yggdrasil.http_.cache_config import CacheConfig
from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.send_config import SendConfig
from yggdrasil.http_.session import HTTPSession
from yggdrasil.io.nested.folder_path import FolderPath
from yggdrasil.io.path.local_path import LocalPath


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _time(label, fn, repeat=5, inner=1):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        elapsed = (time.perf_counter() - t0) / inner
        times.append(elapsed * 1e6)
    best = min(times)
    med = statistics.median(times)
    avg = statistics.mean(times)
    print(f"{label:<70s} best={best:10.2f} us  median={med:10.2f} us  mean={avg:10.2f} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("ygg-bench-spark")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url=base)
    n = args.n
    repeat = args.repeat

    print(f"# repeat={repeat}  n={n}")
    print(f"# {'label':<70s} {'best':>10s}        {'median':>10s}        {'mean':>10s}")

    reqs = [HTTPRequest.prepare(method="GET", url=f"{base}/bench_{i}") for i in range(n)]

    # --- Spark fan-out (no cache) ---
    _time(
        f"send_many({n} reqs, spark, no cache) → list",
        lambda: list(session.send_many(iter(reqs), spark_session=spark)),
        repeat=repeat,
    )

    # --- Spark fan-out (1 req) ---
    one = [HTTPRequest.prepare(method="GET", url=f"{base}/bench_one")]
    _time(
        "send_many(1 req, spark, no cache) → list",
        lambda: list(session.send_many(iter(one), spark_session=spark)),
        repeat=repeat,
    )

    # --- Local send (no spark, no cache) baseline ---
    _time(
        f"send_many({n} reqs, local, no cache) → list",
        lambda: list(session.send_many(iter(reqs))),
        repeat=repeat,
    )

    # --- Spark with local cache (cold then warm) ---
    import tempfile, shutil
    cache_dir = tempfile.mkdtemp()
    cache = CacheConfig(tabular=FolderPath(path=LocalPath.from_(cache_dir)))

    cold_reqs = [HTTPRequest.prepare(method="GET", url=f"{base}/cache_{i}") for i in range(n)]
    for r in cold_reqs:
        r.send_config = SendConfig(local_cache=cache)

    _time(
        f"send_many({n}, spark, local cache COLD) → list",
        lambda: list(session.send_many(iter(cold_reqs), spark_session=spark)),
        repeat=1,
    )

    warm_reqs = [HTTPRequest.prepare(method="GET", url=f"{base}/cache_{i}") for i in range(n)]
    for r in warm_reqs:
        r.send_config = SendConfig(local_cache=cache)

    _time(
        f"send_many({n}, spark, local cache WARM) → list",
        lambda: list(session.send_many(iter(warm_reqs), spark_session=spark)),
        repeat=repeat,
    )

    shutil.rmtree(cache_dir, ignore_errors=True)

    # --- send_many_batches (batch object) ---
    _time(
        f"send_many_batches({n}, spark) → list[batch]",
        lambda: list(session.send_many_batches(iter(reqs), spark_session=spark)),
        repeat=repeat,
    )

    # --- Batch read_arrow_batches ---
    def _batch_to_arrow():
        batches = list(session.send_many_batches(iter(reqs), spark_session=spark))
        for b in batches:
            list(b.read_arrow_batches())

    _time(
        f"send_many_batches({n}, spark) → read_arrow_batches",
        _batch_to_arrow,
        repeat=repeat,
    )

    spark.stop()
    srv.shutdown()
    print("# done")


if __name__ == "__main__":
    main()
