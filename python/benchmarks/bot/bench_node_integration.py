"""Integration benchmarks for node operations.

Measures real API call latency, streaming throughput, and
local vs remote path optimization.

Requires a running node: ygg node serve --no-front
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("YGG_BENCH_URL", "http://127.0.0.1:8100")


def _timed(label: str, fn, n: int = 1):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = statistics.mean(times)
    med = statistics.median(times)
    p99 = sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else times[0]
    print(f"  {label}: avg={avg*1000:.1f}ms  med={med*1000:.1f}ms  p99={p99*1000:.1f}ms  (n={n})")
    return result


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _post(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _delete(path: str) -> None:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=10):
        pass


def bench_discovery():
    print("\n=== Discovery ===")
    _timed("GET /api/hello", lambda: _get("/api/hello"), n=50)
    _timed("GET /api/hello/peers", lambda: _get("/api/hello/peers"), n=50)


def bench_function_lifecycle():
    print("\n=== Function Lifecycle ===")

    func = _timed("POST /api/function (create)", lambda: _post("/api/function", {
        "name": f"bench-func-{time.monotonic()}",
        "code": "print('hello')",
        "language": "python",
    }))
    func_id = func["function"]["id"]

    _timed("GET /api/function/{id}", lambda: _get(f"/api/function/{func_id}"), n=50)

    _timed("GET /api/function (list)", lambda: _get("/api/function"), n=50)

    run = _timed("POST /api/function/{id}/run", lambda: _post(f"/api/function/{func_id}/run", {}))
    run_id = run["run"]["id"]

    time.sleep(2)

    _timed("GET /api/run/{id}", lambda: _get(f"/api/run/{run_id}"), n=20)

    _delete(f"/api/function/{func_id}")


def bench_function_upsert():
    print("\n=== Upsert Performance ===")
    name = f"bench-upsert-{int(time.time())}"
    resp = _timed("POST /api/function (create)", lambda: _post("/api/function", {
        "name": name, "code": "x = 1",
    }))
    func_id = resp["function"]["id"]
    _timed("POST /api/function (update same name)", lambda: _post("/api/function", {
        "name": name, "code": "x = 2",
    }), n=50)
    _delete(f"/api/function/{func_id}")


def bench_monitor():
    print("\n=== Monitor ===")
    _timed("GET /api/monitor", lambda: _get("/api/monitor"), n=50)


def bench_node_path_local():
    print("\n=== NodePath Local ===")
    from yggdrasil.node.path import NodePath

    p = NodePath("bench-test")
    p.mkdir()

    _timed("write_text", lambda: (p / "test.txt").write_text("hello " * 1000), n=100)
    _timed("read_text", lambda: (p / "test.txt").read_text(), n=100)
    _timed("stat", lambda: (p / "test.txt").stat(), n=100)
    _timed("iterdir", lambda: list(p.iterdir()), n=100)

    import shutil
    lp = p._local_path()
    if lp.exists():
        shutil.rmtree(lp)


def bench_streaming():
    print("\n=== Streaming ===")
    from yggdrasil.node.path import NodePath

    p = NodePath("bench-stream")
    p.mkdir()
    data = b"x" * (1024 * 1024)  # 1MB
    (p / "large.bin").write_bytes(data)

    def stream_read():
        chunks = list((p / "large.bin").stream_read(chunk_size=65536))
        return sum(len(c) for c in chunks)

    _timed("stream_read 1MB", stream_read, n=20)

    import shutil
    lp = p._local_path()
    if lp.exists():
        shutil.rmtree(lp)


if __name__ == "__main__":
    print(f"Benchmarking against: {BASE_URL}")

    try:
        _get("/api/hello")
    except Exception:
        print("ERROR: Node not reachable. Start with: ygg node serve --no-front")
        exit(1)

    bench_discovery()
    bench_function_lifecycle()
    bench_function_upsert()
    bench_monitor()
    bench_node_path_local()
    bench_streaming()
    print("\nDone.")
