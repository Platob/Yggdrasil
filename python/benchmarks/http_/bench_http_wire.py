"""Wire-level benchmarks: real sockets over localhost.

Measures the full send pipeline cost with a real HTTP server on
localhost — connection setup, keep-alive reuse, response parsing,
body read, and connection release.

Usage::

    python benchmarks/http_/bench_http_wire.py
    python benchmarks/http_/bench_http_wire.py --repeat 5
"""
from __future__ import annotations

import argparse
import http.server
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession


# ---------------------------------------------------------------------------
# Local HTTP server
# ---------------------------------------------------------------------------

_SMALL_JSON = json.dumps({"ok": True, "n": 1}).encode()
_MEDIUM_JSON = json.dumps({"data": "x" * 1_000}).encode()
_LARGE_JSON = json.dumps({"data": "x" * 100_000}).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/small":
            self._respond(_SMALL_JSON)
        elif path == "/medium":
            self._respond(_MEDIUM_JSON)
        elif path == "/large":
            self._respond(_LARGE_JSON)
        else:
            self._respond(_SMALL_JSON)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length) if length else b""
        self._respond(_SMALL_JSON)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def _start_server():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}", srv


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _bench(label: str, fn: Callable, *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 3)):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {"label": label, "best": min(samples), "median": statistics.median(samples)}


def _fmt(r: dict) -> str:
    best_us = r["best"] * 1e6
    med_us = r["median"] * 1e6
    best_rps = 1.0 / r["best"] if r["best"] > 0 else 0
    return f"{r['label']:<55s}  best={best_us:9.1f} us  median={med_us:9.1f} us  ({best_rps:,.0f} req/s)"


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run(repeat: int) -> list[dict]:
    base_url, srv = _start_server()
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url=base_url)
    out = []

    # --- Single GET, warm pool ---
    out.append(_bench(
        "GET /small (warm pool, keep-alive)",
        lambda: session.get("/small"),
        repeat=repeat, inner=500,
    ))

    out.append(_bench(
        "GET /medium (~1 KiB body)",
        lambda: session.get("/medium"),
        repeat=repeat, inner=200,
    ))

    out.append(_bench(
        "GET /large (~100 KiB body)",
        lambda: session.get("/large"),
        repeat=repeat, inner=50,
    ))

    # --- POST ---
    body_small = b'{"key": "value"}'
    out.append(_bench(
        "POST /small (small body, warm pool)",
        lambda: session.post("/small", data=body_small),
        repeat=repeat, inner=200,
    ))

    body_64k = b"x" * 64_000
    out.append(_bench(
        "POST /small (64 KiB body)",
        lambda: session.post("/small", data=body_64k),
        repeat=repeat, inner=50,
    ))

    # --- Cold URL (different path each call, identity recomputed) ---
    counter = [0]
    def _cold_get():
        counter[0] += 1
        session.get(f"/small?i={counter[0]}")
    out.append(_bench(
        "GET /small?i=N (cold URL each call)",
        _cold_get,
        repeat=repeat, inner=200,
    ))

    # --- Connection pool: clear + rebuild ---
    def _clear_and_get():
        session.clear_connections()
        session.get("/small")
    out.append(_bench(
        "clear_connections + GET (fresh socket each call)",
        _clear_and_get,
        repeat=repeat, inner=100,
    ))

    # --- send_many: batched parallel ---
    reqs_10 = [HTTPRequest.prepare("GET", f"{base_url}/small?b={i}") for i in range(10)]
    out.append(_bench(
        "send_many(10 reqs) (parallel dispatch)",
        lambda: list(session.send_many(reqs_10, raise_error=False)),
        repeat=repeat, inner=5,
    ))

    reqs_50 = [HTTPRequest.prepare("GET", f"{base_url}/small?b={i}") for i in range(50)]
    out.append(_bench(
        "send_many(50 reqs) (parallel dispatch)",
        lambda: list(session.send_many(reqs_50, raise_error=False)),
        repeat=repeat, inner=3,
    ))

    # --- Session construction (singleton hit) ---
    out.append(_bench(
        "HTTPSession(base_url=...) singleton hit",
        lambda: HTTPSession(base_url=base_url),
        repeat=repeat, inner=10_000,
    ))

    # --- Arrow metadata extraction ---
    out.append(_bench(
        "GET + arrow_values extraction",
        lambda: session.get("/small").arrow_values,
        repeat=repeat, inner=100,
    ))

    # --- Throughput: sequential GETs for 1 second ---
    print("\n  Throughput (sequential, 1s window):")
    for path, label in [("/small", "small"), ("/medium", "1KiB"), ("/large", "100KiB")]:
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            session.get(path)
            count += 1
        elapsed = time.perf_counter() - t0
        rps = count / elapsed
        print(f"    {label:<12s}  {count:>6,d} reqs in {elapsed:.2f}s = {rps:,.0f} req/s")

    srv.shutdown()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    print(f"# Wire benchmarks (repeat={args.repeat})")
    print(f"# {'label':<55s}  {'best':>14s}  {'median':>14s}")
    results = run(args.repeat)
    for r in results:
        print(_fmt(r))


if __name__ == "__main__":
    main()
