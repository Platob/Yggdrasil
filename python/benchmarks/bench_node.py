#!/usr/bin/env python3
"""Quick benchmark for the Yggdrasil node API — measures key endpoint latencies.

Usage:
    python benchmarks/bench_node.py [--url http://127.0.0.1:8100] [--runs 10]

Endpoints tested:
  - GET /api/ping          (health)
  - GET /api/card          (node identity)
  - GET /api/v2/backend    (resource snapshot)
  - POST /api/v2/analysis/finance  (finance metrics, if a test file exists)
  - POST /api/v2/trading/indicators (new endpoint)
"""
import argparse
import json
import statistics
import time
import urllib.request
from typing import Any


def _req(url: str, method: str = "GET", body: Any = None, timeout: int = 10) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except Exception as e:
        return 0, str(e)


def bench(label: str, fn, runs: int = 10) -> dict:
    latencies = []
    errors = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        status, _ = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        if status >= 200 and status < 300:
            latencies.append(elapsed)
        else:
            errors += 1
    if not latencies:
        return {"label": label, "error": f"all {runs} calls failed"}
    return {
        "label": label,
        "runs": runs,
        "errors": errors,
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    runs = args.runs

    print(f"\nYggdrasil node benchmark — {base} — {runs} runs each\n{'='*60}")

    cases = [
        ("GET /api/ping", lambda: _req(f"{base}/api/ping")),
        ("GET /api/card", lambda: _req(f"{base}/api/card")),
        ("GET /api/v2/backend", lambda: _req(f"{base}/api/v2/backend")),
        ("GET /api/v2/network/self", lambda: _req(f"{base}/api/v2/network/self")),
    ]

    results = []
    for label, fn in cases:
        r = bench(label, fn, runs)
        results.append(r)
        if "error" in r:
            print(f"  {label:40s}  ERROR: {r['error']}")
        else:
            print(f"  {label:40s}  mean={r['mean_ms']:6.1f}ms  p95={r['p95_ms']:6.1f}ms  errors={r['errors']}/{runs}")

    print(f"\n{'='*60}")
    print("Done. Results (JSON):")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
