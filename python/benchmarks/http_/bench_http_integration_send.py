"""Integration-send benchmarks — performance **and** memory.

Exercises the full :meth:`HTTPSession.send` / :meth:`send_many` pipeline
two ways:

* **Wire-stubbed** — ``_send_once`` is overridden to build an
  :class:`HTTPResponse` without touching a socket, so the numbers
  isolate the *pure Python* per-request CPU cost (prepare → send →
  response build → media normalisation → body wire-up). This is the
  part we can actually optimise; it is invisible under real socket
  noise.

* **Live localhost** — a keep-alive HTTP/1.1 server with ``TCP_NODELAY``
  and a single combined header+body write, so the warm-pool path is
  measured without the delayed-ACK / Nagle stall that makes a naive
  ``BaseHTTPRequestHandler`` report ~40 ms/req. This is the realistic
  end-to-end latency.

Memory is measured with :mod:`tracemalloc`: peak allocation across a
run, and the *retained* growth after the run (a streaming pipeline
should not accumulate — ``send_many`` over N requests must not hold N
responses live).

Usage::

    PYTHONPATH=src python benchmarks/http_/bench_http_integration_send.py
    PYTHONPATH=src python benchmarks/http_/bench_http_integration_send.py --repeat 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import http.server
import json
import socket
import statistics
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Callable

# Ensure project src is importable when run directly.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.memory import Memory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SMALL = json.dumps({"ok": True, "n": 1}).encode()
_MEDIUM = json.dumps({"data": "x" * 1_000}).encode()
_RECEIVED_AT = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


class _StubSession(HTTPSession):
    """HTTPSession whose wire send returns a canned 200 — no socket.

    Lets the benchmark measure the surrounding pipeline (prepare →
    response build → media normalisation) without any network cost,
    so a regression in the Python hot path shows up cleanly.
    """

    _body: bytes = _SMALL

    def _send_once(self, *, request, timeout, preload_content, decode_content, tags=None):
        return HTTPResponse(
            request=request,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(self._body)),
            },
            tags={},
            buffer=Memory(binary=self._body),
            received_at=_RECEIVED_AT,
        )


# ---------------------------------------------------------------------------
# Live localhost server — keep-alive, TCP_NODELAY, single write
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive

    def setup(self):
        super().setup()
        # Without TCP_NODELAY a keep-alive localhost exchange stalls on
        # delayed-ACK for ~40 ms — a server artifact that would swamp the
        # client-side cost we want to measure.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _reply(self, body: bytes):
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(head + body)  # one write — no second-packet stall

    def do_GET(self):
        self._reply(_MEDIUM if self.path.startswith("/medium") else _SMALL)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self._reply(_SMALL)

    def log_message(self, *a):
        pass


class _Server(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _start_server() -> tuple[str, _Server]:
    srv = _Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


# ---------------------------------------------------------------------------
# Timing + memory harness
# ---------------------------------------------------------------------------


def _time(label: str, fn: Callable[[], object], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 5)):
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
    rps = 1.0 / r["best"] if r["best"] > 0 else 0.0
    return (
        f"{r['label']:<52s}  best={best_us:9.2f} us  "
        f"median={med_us:9.2f} us  ({rps:,.0f} ops/s)"
    )


def _mem(label: str, fn: Callable[[], object], *, ops: int) -> dict:
    """Run *fn* *ops* times under tracemalloc; report peak + retained.

    ``peak`` is the high-water allocation during the run (per op).
    ``retained`` is the net heap growth that survived the run (total) —
    a streaming pipeline should keep this near zero regardless of *ops*.
    """
    gc.collect()
    tracemalloc.start()
    base_cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for _ in range(ops):
        fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return {
        "label": label,
        "peak_per_op": (peak - base_cur) / ops,
        "retained_total": cur - base_cur,
        "ops": ops,
    }


def _fmt_mem(r: dict) -> str:
    return (
        f"{r['label']:<52s}  peak={r['peak_per_op']/1024:8.2f} KiB/op  "
        f"retained={r['retained_total']/1024:9.2f} KiB total over {r['ops']:,} ops"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run(repeat: int) -> None:
    HTTPSession._INSTANCES.clear()

    # --- pure-pipeline (wire stubbed) -------------------------------------
    stub = _StubSession(base_url="http://localhost:9999")
    reqs_stub = [HTTPRequest.prepare("GET", f"http://localhost:9999/x?i={i}") for i in range(50)]

    print("# Pure pipeline — wire stubbed (isolates Python CPU cost)")
    print(f"# {'label':<52s}  {'best':>14s}  {'median':>16s}")
    perf = [
        _time("prepare_request only", lambda: stub.prepare_request("GET", "/x"),
              repeat=repeat, inner=20_000),
        _time("send (prepare + wire + response build)", lambda: stub.get("/x"),
              repeat=repeat, inner=20_000),
        _time("send_many(50) sequential fast path",
              lambda: list(stub.send_many(iter(list(reqs_stub)), raise_error=False)),
              repeat=repeat, inner=400),
    ]
    for r in perf:
        print(_fmt(r))

    # --- live localhost (warm keep-alive pool) ----------------------------
    base_url, srv = _start_server()
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url=base_url)
    reqs_live = [HTTPRequest.prepare("GET", f"{base_url}/x?i={i}") for i in range(50)]
    body = b'{"k":"v"}'

    print("\n# Live localhost — keep-alive, TCP_NODELAY (realistic end-to-end)")
    print(f"# {'label':<52s}  {'best':>14s}  {'median':>16s}")
    live = [
        _time("GET /small (warm pool)", lambda: session.get("/small"),
              repeat=repeat, inner=2_000),
        _time("GET /medium (~1 KiB)", lambda: session.get("/medium"),
              repeat=repeat, inner=2_000),
        _time("POST /small (small body)", lambda: session.post("/small", data=body),
              repeat=repeat, inner=2_000),
        _time("send_many(50) sequential",
              lambda: list(session.send_many(iter(list(reqs_live)), raise_error=False)),
              repeat=repeat, inner=40),
    ]
    for r in live:
        print(_fmt(r))

    # --- memory -----------------------------------------------------------
    print("\n# Memory — tracemalloc (peak per op + retained after run)")
    mem = [
        _mem("send (stubbed wire)", lambda: stub.get("/x"), ops=5_000),
        _mem("send (live, body discarded)", lambda: session.get("/small"), ops=2_000),
    ]

    # send_many must stream: draining the iterator without keeping
    # responses should not grow the heap by N responses.
    def _drain_send_many():
        for _ in session.send_many(iter(list(reqs_live)), raise_error=False):
            pass
    mem.append(_mem("send_many(50) drained (streaming)", _drain_send_many, ops=40))

    for r in mem:
        print(_fmt_mem(r))

    srv.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    print(f"# integration-send benchmarks (repeat={args.repeat})\n")
    run(args.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
