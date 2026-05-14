"""End-to-end :class:`HTTPSession.send` benchmarks against a localhost server.

What this covers
----------------

The existing ``bench_http.py`` stubs ``_local_send`` to skip the wire
entirely so its numbers measure yggdrasil's own per-call overhead.
This bench keeps the wire but pins it to ``127.0.0.1`` so the network
stack doesn't dominate, then measures the realistic end-to-end cost
of ``session.send``:

* tiny JSON response (warm pool, identical URL),
* tiny JSON response (cold per call: different URL each request, so
  the urllib3 connection pool keeps reusing one host but the request
  identity has to be re-prepared),
* `~1 KiB` JSON response,
* `~64 KiB` JSON response,
* same workload through :meth:`HTTPSession.send_many` so we see what
  batching saves at the dispatch layer,
* same workload through a :class:`SchemaSession` with the remote
  cache **disabled** but a local on-disk cache enabled — measures
  the on-disk fast-path read/write overhead over a real socket.

The numbers expose three buckets of work:

  network + parse: time on the socket and parsing the response;
  request identity: ``prepare`` + the cached hash surface
                    (recomputed cold each call when URLs vary);
  session dispatch: ``prepare_request_before_send`` +
                    ``_send`` cache pipeline.

Comparing a bare ``HTTPSession`` and a ``SchemaSession`` with
``local_cache=True`` (and remote disabled) isolates the on-disk
cache cost — the optimisation surface that's worth chasing first
because every cache-hit request that comes back from disk dodges
the entire wire.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_http_live.py
    PYTHONPATH=src python benchmarks/io/bench_http_live.py --repeat 3
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from yggdrasil.io.http_.session import HTTPSession
from yggdrasil.io.request import PreparedRequest


# ---------------------------------------------------------------------------
# Localhost test server
# ---------------------------------------------------------------------------


def _build_payload(size_bytes: int) -> bytes:
    """Return a JSON object with a ``data`` string padded to *size_bytes*."""
    # 11 bytes of framing for ``{"data":""}`` plus the body. The exact
    # size doesn't matter to the bench — we just want a fixed,
    # reproducible payload per scenario.
    pad = max(0, size_bytes - 11)
    return ('{"data":"' + ("x" * pad) + '"}').encode("ascii")


PAYLOADS: dict[str, bytes] = {
    # Tiny JSON: ``{"data":"x"}`` plus a few bytes — dominates by the
    # send/receive round trip rather than the body parse.
    "tiny": _build_payload(16),
    # ~1 KiB — typical small API response.
    "kib1": _build_payload(1024),
    # ~64 KiB — exercises buffered IO and the response holder copy.
    "kib64": _build_payload(64 * 1024),
}


class _BenchHandler(BaseHTTPRequestHandler):
    """Serve a fixed-size JSON payload based on the request path.

    Path conventions: ``/tiny`` / ``/kib1`` / ``/kib64`` map to the
    matching :data:`PAYLOADS` entry. Anything else returns 404 so a
    typo in a bench scenario fails loudly instead of yielding silent
    misses.

    HTTP/1.1 with persistent connections — the stdlib default of
    HTTP/1.0 forces one TCP handshake per request, which under
    Linux's delayed-ACK on small payloads inflates loopback latency
    to ~40 ms / request. With keep-alive the urllib3 pool reuses one
    socket across the whole scenario and the bench measures actual
    request-dispatch cost.
    """

    protocol_version = "HTTP/1.1"

    # Silence the noisy default access log — every bench call would
    # otherwise dump a line to stderr.
    def log_message(self, format, *args):  # noqa: A002, D401 - stdlib name
        return

    def _serve(self, path: str) -> None:
        key = path.strip("/").split("/", 1)[0]
        payload = PAYLOADS.get(key)
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - stdlib name
        self._serve(self.path)


def _start_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Bind a server to a random localhost port and run it in a daemon thread.

    Returns the server, the serving thread, and the resolved base URL
    (``http://127.0.0.1:<port>``). The thread is daemonic so the bench
    process exits cleanly even if a scenario short-circuits the
    teardown.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BenchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Timing helpers (mirror bench_http.py / bench_http_cache.py)
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], object], *, repeat: int, inner: int) -> dict:
    # One warmup pass to amortise pool / TLS handshake costs (no TLS here,
    # but keeps the same shape as the cached benches so reading the output
    # against ``bench_http.py`` is direct).
    for _ in range(min(inner, 50)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale, unit = (1e9, "ns") if r["best"] < 1e-6 else (1e6, "us")
    return (
        f"{r['label']:<70s}  "
        f"best={r['best']*scale:9.2f} {unit}  "
        f"median={r['median']*scale:9.2f} {unit}  "
        f"mean={r['mean']*scale:9.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _wire_scenarios(base_url: str, repeat: int) -> list[dict]:
    """Bare ``HTTPSession.send`` over the localhost socket."""
    out: list[dict] = []
    sess = HTTPSession(base_url=base_url, key="bench-live")

    for sz in ("tiny", "kib1", "kib64"):
        path = f"/{sz}"
        # Pre-built request reused per iteration — measures the send
        # pipeline without paying for ``prepare`` each call. ``prepare``
        # is benched separately in ``bench_http.py``; keep the wire
        # bench focused on the dispatch + transport cost.
        req = PreparedRequest.prepare("GET", f"{base_url}{path}")
        _ = req.public_hash  # warm identity surface
        out.append(_time_one(
            f"HTTPSession.send {sz} (reused PreparedRequest)",
            lambda r=req: sess.send(r, raise_error=False),
            repeat=repeat, inner=200,
        ))

        # Cold per-iter: rebuild the request every call so the bench
        # also captures the ``prepare`` cost realistic call sites pay.
        def _send_fresh(_path=path):
            r = PreparedRequest.prepare("GET", f"{base_url}{_path}")
            return sess.send(r, raise_error=False)
        out.append(_time_one(
            f"HTTPSession.send {sz} (cold prepare per call)",
            _send_fresh,
            repeat=repeat, inner=200,
        ))

    return out


def _send_many_scenarios(base_url: str, repeat: int) -> list[dict]:
    out: list[dict] = []
    sess = HTTPSession(base_url=base_url, key="bench-many")
    batch_size = 32

    # ``send_many`` consumes an iterator. Rebuild it on every iter so
    # the bench measures the dispatch path, not a single replay of an
    # already-drained generator.
    def _batch_get_tiny():
        reqs = (
            PreparedRequest.prepare("GET", f"{base_url}/tiny")
            for _ in range(batch_size)
        )
        # Drain the iterator into a list to actually wait for sends.
        return list(sess.send_many(reqs, raise_error=False, ordered=False))

    out.append(_time_one(
        f"HTTPSession.send_many tiny (n={batch_size})",
        _batch_get_tiny,
        repeat=repeat, inner=20,
    ))
    return out


def _local_cache_scenarios(base_url: str, repeat: int) -> list[dict]:
    """Compare bare ``HTTPSession.send`` against a session with the local
    on-disk cache turned on. Cache hits dodge the entire socket round trip,
    so the delta is the visible win the on-disk fast-path buys.
    """
    out: list[dict] = []

    # Bare session for the baseline. Same singleton-cache key it had
    # in ``_wire_scenarios`` — singleton-by-key means the pool stays
    # hot across scenarios.
    sess = HTTPSession(base_url=base_url, key="bench-live")
    tmp = tempfile.mkdtemp(prefix="ygg-bench-live-cache-")
    try:
        from yggdrasil.io.send_config import CacheConfig

        cfg = CacheConfig(path=tmp)
        # First call seeds the cache, subsequent calls hit it.
        cold_req = PreparedRequest.prepare("GET", f"{base_url}/tiny")
        sess.send(cold_req, local_cache=cfg, raise_error=False)

        warm_req = PreparedRequest.prepare("GET", f"{base_url}/tiny")
        _ = warm_req.public_hash
        out.append(_time_one(
            "HTTPSession.send tiny (local-cache HIT, no wire)",
            lambda: sess.send(warm_req, local_cache=cfg, raise_error=False),
            repeat=repeat, inner=200,
        ))

        # Cache MISS: a fresh URL each iteration so we walk the wire +
        # store-to-disk path. Compares directly with the ``cold
        # prepare per call`` baseline above.
        counter = {"n": 0}

        def _send_cache_miss():
            counter["n"] += 1
            r = PreparedRequest.prepare(
                "GET", f"{base_url}/tiny?probe={counter['n']}",
            )
            return sess.send(r, local_cache=cfg, raise_error=False)
        out.append(_time_one(
            "HTTPSession.send tiny (local-cache MISS, store-to-disk)",
            _send_cache_miss,
            repeat=repeat, inner=200,
        ))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return out


# ---------------------------------------------------------------------------
# Aggregator + CLI
# ---------------------------------------------------------------------------


def scenarios(base_url: str, repeat: int) -> list[dict]:
    return [
        *_wire_scenarios(base_url, repeat),
        *_send_many_scenarios(base_url, repeat),
        *_local_cache_scenarios(base_url, repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    server, thread, base_url = _start_server()
    try:
        print(f"# repeat={args.repeat}")
        print(f"# server={base_url}")
        print(f"# {'label':<70s}  {'best':>15s}  {'median':>17s}  {'mean':>15s}")
        for row in scenarios(base_url, args.repeat):
            print(_fmt(row))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
