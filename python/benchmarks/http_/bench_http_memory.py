"""Deep memory benchmarks for the HTTPSession send / stream pipeline.

Goes beyond the peak/retained summary in
``bench_http_integration_send.py`` to answer four questions that a
simple per-op number can't:

1. **Leak detection** — run a fixed workload N times at two different
   N and confirm retained heap + live object count do *not* scale with
   N. A real leak shows up as retained(N=2k) ≈ 2× retained(N=1k); a
   clean pipeline keeps both flat.

2. **Allocation-source attribution** — :mod:`tracemalloc` snapshot
   diff over a workload, grouped by source line, so a regression points
   at the file:line that allocated instead of just "memory went up".

3. **Streaming vs buffering memory shape** — a large body read three
   ways (``.content`` whole-buffer, ``.stream(64k)`` chunked,
   ``.data``) to show the peak-RSS difference between holding the whole
   payload and sliding a window over it.

4. **Concurrency memory under load** — peak heap while a
   ``send_many`` fan-out is in flight against a latent server, plus a
   check that the connection pool does not accumulate idle sockets
   across repeated batches.

Usage::

    PYTHONPATH=src python benchmarks/http_/bench_http_memory.py
    PYTHONPATH=src python benchmarks/http_/bench_http_memory.py --repeat 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import http.server
import json
import linecache
import os
import socket
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from yggdrasil.http_.request import HTTPRequest
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession
from yggdrasil.path.memory import Memory


_SMALL = json.dumps({"ok": True, "n": 1}).encode()
_LARGE = b"x" * (8 * 1024 * 1024)          # 8 MiB
_LATENCY_S = 0.002
_RECEIVED_AT = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)


class _StubSession(HTTPSession):
    """Wire stubbed — no socket, so memory reflects only the pipeline."""

    _body: bytes = _SMALL

    def _send_once(self, *, request, timeout, preload_content, decode_content, tags=None):
        return HTTPResponse(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json",
                     "Content-Length": str(len(self._body))},
            tags={},
            buffer=Memory(binary=self._body),
            received_at=_RECEIVED_AT,
        )


# ---------------------------------------------------------------------------
# Live localhost server
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _reply(self, body: bytes):
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/octet-stream\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        self.wfile.write(head + body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/large":
            self._reply(_LARGE)
        elif path == "/slow":
            time.sleep(_LATENCY_S)
            self._reply(_SMALL)
        else:
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
# Measurement primitives
# ---------------------------------------------------------------------------


def _run_traced(fn: Callable[[], object], ops: int) -> tuple[int, int, int]:
    """Run *fn* *ops* times. Return (peak_total, retained_total, obj_delta)."""
    gc.collect()
    objs_before = len(gc.get_objects())
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    for _ in range(ops):
        fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    objs_after = len(gc.get_objects())
    return peak - base, cur - base, objs_after - objs_before


def _leak_probe(label: str, fn: Callable[[], object], *, low: int, high: int) -> None:
    """Run *fn* at two op counts; flag retention / objects that scale with N.

    A clean streaming pipeline keeps retained heap and live-object delta
    flat across ``low`` and ``high``. If ``high`` retention ≈ (high/low)×
    ``low`` retention, that is a leak.
    """
    _peak_l, ret_l, obj_l = _run_traced(fn, low)
    _peak_h, ret_h, obj_h = _run_traced(fn, high)
    ratio = high / low
    # Growth factor relative to op-count growth: ~0 = flat (good),
    # ~1.0 = retention scales 1:1 with ops (leak).
    ret_scale = (ret_h / ret_l) / ratio if ret_l > 0 else 0.0
    verdict = "LEAK?" if (ret_h > 64 * 1024 and ret_scale > 0.5) or obj_h > obj_l + ratio * 5 else "ok"
    print(
        f"{label:<40s}  "
        f"ret@{low}={ret_l/1024:7.1f}K  ret@{high}={ret_h/1024:7.1f}K  "
        f"obj@{low}={obj_l:+5d}  obj@{high}={obj_h:+5d}  [{verdict}]"
    )


def _attribute(label: str, fn: Callable[[], object], ops: int, top: int = 6) -> None:
    """Snapshot-diff *fn*'s workload and print the top allocating lines."""
    gc.collect()
    tracemalloc.start()
    for _ in range(min(ops, 10)):
        fn()
    gc.collect()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(ops):
        fn()
    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    stats = snap2.compare_to(snap1, "lineno")
    print(f"\n# Allocation sources — {label} ({ops} ops)")
    shown = 0
    for st in stats:
        if st.size_diff <= 0:
            continue
        frame = st.traceback[0]
        fname = os.sep.join(frame.filename.split(os.sep)[-2:])
        line = linecache.getline(frame.filename, frame.lineno).strip()
        print(f"  {st.size_diff/1024:8.1f} KiB  {st.count_diff:+6d} blk  "
              f"{fname}:{frame.lineno}")
        if line:
            print(f"             {line[:72]}")
        shown += 1
        if shown >= top:
            break


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run(repeat: int) -> None:
    HTTPSession._INSTANCES.clear()
    stub = _StubSession(base_url="http://localhost:9999")
    base_url, srv = _start_server()
    HTTPSession._INSTANCES.clear()
    session = HTTPSession(base_url=base_url)

    # === 1. Leak detection ================================================
    print("# 1. Leak detection — retention + live objects must stay flat as ops grow")
    reqs = [HTTPRequest.prepare("GET", f"{base_url}/x?i={i}") for i in range(20)]

    _leak_probe("send (stubbed)", lambda: stub.get("/x"), low=2_000, high=8_000)
    _leak_probe("send (live, body discarded)", lambda: session.get("/x"),
                low=1_000, high=4_000)
    _leak_probe("send + .content", lambda: session.get("/x").content,
                low=1_000, high=4_000)
    _leak_probe("send + .json()", lambda: stub.get("/x").json(),
                low=1_000, high=4_000)

    def _drain():
        for _ in session.send_many(iter(list(reqs)), raise_error=False):
            pass
    _leak_probe("send_many(20) drained", _drain, low=50, high=200)

    # === 2. Allocation attribution ========================================
    _attribute("send (stubbed)", lambda: stub.get("/x"), 3_000)
    _attribute("send + .json()", lambda: stub.get("/x").json(), 3_000)

    # === 3. Streaming vs buffering memory shape ===========================
    print("\n# 3. Large body (8 MiB) — peak heap by read strategy")

    def _content():
        session.get("/large").content

    def _stream_chunks():
        r = session.get("/large")
        total = 0
        for chunk in r.stream(64 * 1024):
            total += len(chunk)
        return total

    def _data():
        _ = session.get("/large").data

    for label, fn in (
        (".content (whole buffer)", _content),
        (".stream(64k) (chunked drain)", _stream_chunks),
        (".data (whole buffer)", _data),
    ):
        peak, ret, _obj = _run_traced(fn, 20)
        print(f"  {label:<34s}  peak={peak/1024/1024:7.2f} MiB/op  "
              f"retained={ret/1024:8.1f} KiB total")

    # === 4. Concurrency memory under load =================================
    print(f"\n# 4. Concurrency memory — send_many fan-out under {_LATENCY_S*1e3:.0f} ms latency")
    n = 40
    slow = [HTTPRequest.prepare("GET", f"{base_url}/slow?i={i}") for i in range(n)]

    def _send_many_fanout():
        list(session.send_many(iter(list(slow)), raise_error=False))

    peak, ret, obj = _run_traced(_send_many_fanout, 10)
    print(f"  send_many({n}) fan-out          peak={peak/1024:8.1f} KiB  "
          f"retained={ret/1024:7.1f} KiB  obj_delta={obj:+d}")

    # Pool must not accumulate idle sockets across repeated batches.
    before_pool = sum(len(q) for q in session._connections.values())
    for _ in range(20):
        _send_many_fanout()
    after_pool = sum(len(q) for q in session._connections.values())
    print(f"  idle sockets in pool: before={before_pool}  after 20 batches={after_pool}  "
          f"(cap=pool_maxsize={session.pool_maxsize})  "
          f"[{'ok' if after_pool <= session.pool_maxsize else 'GROWTH'}]")

    srv.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", type=int, default=1,
                    help="Accepted for run_all.py compatibility; this bench "
                         "is allocation-deterministic and ignores it.")
    ap.parse_args()
    print("# deep memory benchmarks\n")
    run(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
