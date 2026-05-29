"""Benchmark random-seek reads on :class:`VolumePath` over the Files API.

Unlike the other Databricks path benchmarks (which mock the workspace
handle and measure in-process wrapper cost), this one stands up a real
localhost HTTP server that emulates the Databricks Files REST API and
drives a *real* :class:`~yggdrasil.http_.HTTPSession` through
:class:`VolumePath`. It exists to validate the headline optimization of
the HTTP transport: a bounded / offset read issues an HTTP ``Range``
request and transfers only the requested slice, instead of downloading
the whole object and slicing locally (which the SDK's Files client — and
the first cut of this transport — did on every call).

Two modes run against the same payload:

* **range**  — the server honours ``Range`` (real Files-API behaviour);
  each random read pulls only its slice.
* **full**   — the server ignores ``Range`` and returns 200 + the whole
  body; :class:`VolumePath` slices locally. This reproduces the
  "download everything per seek" cost the optimization removes.

The win is dominated by bytes-over-the-wire (and therefore network
time on a real workspace), so the report leads with the transfer
amplification factor and effective wall time.

Usage::

    python benchmarks/databricks/bench_databricks_volume_random_io.py
    python benchmarks/databricks/bench_databricks_volume_random_io.py \\
        --size-mib 64 --block 4096 --reads 400 --repeat 3
"""
from __future__ import annotations

import argparse
import random
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.http_ import HTTPSession
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Fake Files-API server — honours (or ignores) Range, counts bytes served
# ---------------------------------------------------------------------------


class _State:
    """Mutable server state shared with the request handler."""

    def __init__(self) -> None:
        self.blob: bytes = b""
        self.honor_range: bool = True
        self.bytes_served: int = 0
        self.requests: int = 0
        self.calls: int = 0  # all HTTP requests (GET + HEAD + PUT)


def _make_handler(state: _State):
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Disable Nagle so small 206 bodies don't eat a ~40ms delayed-ACK
        # stall — that artifact would otherwise dwarf the request cost on
        # loopback and hide what we're measuring (bytes + per-request work).
        disable_nagle_algorithm = True

        def log_message(self, *a):  # silence per-request logging
            pass

        def _path_ok(self) -> bool:
            return urlsplit(self.path).path.startswith("/api/2.0/fs/files")

        def do_PUT(self):  # noqa: N802
            state.calls += 1
            length = int(self.headers.get("Content-Length", "0"))
            state.blob = self.rfile.read(length) if length else b""
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):  # noqa: N802
            state.calls += 1
            if not self._path_ok():
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(state.blob)))
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if not self._path_ok():
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            state.requests += 1
            state.calls += 1
            blob = state.blob
            rng = self.headers.get("Range")
            if rng and state.honor_range and rng.startswith("bytes="):
                spec = rng[len("bytes="):].split("-", 1)
                start = int(spec[0]) if spec[0].strip().isdigit() else 0
                end = spec[1].strip() if len(spec) > 1 else ""
                stop = int(end) + 1 if end.isdigit() else len(blob)
                body = blob[start:stop]
                state.bytes_served += len(body)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(blob)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            state.bytes_served += len(blob)
            self.send_response(200)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

    return _Handler


# ---------------------------------------------------------------------------
# Minimal client / service shims — VolumePath only touches base_url,
# files_session(), files_authorization(), and service.client.
# ---------------------------------------------------------------------------


class _Client:
    # Plain class (hashable by id) — the path singleton key folds the
    # bound service in, and ``SimpleNamespace`` defines ``__eq__`` so it
    # would be unhashable.
    def __init__(self, host: str) -> None:
        self.base_url = URL.from_(host)
        self._session = HTTPSession(base_url=host, verify=False)

    def files_session(self) -> HTTPSession:
        return self._session

    def files_authorization(self) -> str:
        return "Bearer bench"


class _Service:
    def __init__(self, client: _Client) -> None:
        self.client = client


def _volume(host: str, page_size) -> VolumePath:
    vp = VolumePath(
        "/Volumes/bench/seek/vol/data.bin",
        service=_Service(_Client(host)),
        page_size=page_size,
    )
    vp.invalidate_singleton()
    return vp


# ---------------------------------------------------------------------------
# Read strategies — opened vs non-opened, paged vs unpaged
# ---------------------------------------------------------------------------


def _direct(host, offsets, block, total):
    # Lowest-level: VolumePath._read_mv(n, pos) — one Range per read.
    vp = _volume(host, None)
    for off in offsets:
        n = min(block, total - off)
        assert len(vp._read_mv(n, off)) == n


def _opened(host, offsets, block, total, page_size):
    # Open a byte cursor once, seek + read through it (the way a random
    # reader uses a file handle). Reads route through _read_mv, quantized
    # to ``page_size`` (None = exact slice).
    vp = _volume(host, page_size)
    with vp.open("rb") as fh:
        for off in offsets:
            n = min(block, total - off)
            fh.seek(off)
            assert len(fh.read(n)) == n


def _non_opened_whole(host, offsets, block, total):
    # Non-opened convenience: each random read re-materialises the whole
    # object (fresh path so the buffer cache doesn't hide the transfer).
    for off in offsets:
        n = min(block, total - off)
        vp = _volume(host, None)
        _ = bytes(vp.read_bytes())[off:off + n]


def _run(fn, state: _State, repeat: int):
    import tracemalloc

    times, served, calls, peak = [], 0, 0, 0
    for _ in range(repeat):
        state.bytes_served = 0
        state.calls = 0
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        peak = max(peak, tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        served = state.bytes_served
        calls = state.calls
    return min(times), statistics.median(times), served, calls, peak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-mib", type=int, default=16, help="payload size (MiB)")
    ap.add_argument("--block", type=int, default=4096, help="bytes per random read")
    ap.add_argument("--reads", type=int, default=200, help="random reads per repeat")
    ap.add_argument("--repeat", type=int, default=3, help="repeats (best wins)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    state = _State()
    state.blob = random.Random(args.seed).randbytes(args.size_mib * 1024 * 1024)
    state.honor_range = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        total = len(state.blob)
        rng = random.Random(args.seed + 1)
        offsets = [rng.randint(0, total - args.block) for _ in range(args.reads)]
        b = args.block

        strategies = [
            ("direct _read_mv", lambda: _direct(host, offsets, b, total)),
            ("opened, unpaged", lambda: _opened(host, offsets, b, total, None)),
            ("opened, 64KiB page", lambda: _opened(host, offsets, b, total, 65536)),
            ("opened, 4MiB page", lambda: _opened(host, offsets, b, total, 4 << 20)),
            ("non-opened (whole)", lambda: _non_opened_whole(host, offsets, b, total)),
        ]

        print(
            f"\nVolumePath random-seek reads — payload={args.size_mib} MiB, "
            f"block={args.block} B, reads={args.reads}, repeat={args.repeat}\n"
        )
        header = (
            f"{'strategy':<22} {'best (s)':>9} {'MiB served':>11} {'amp':>6} "
            f"{'calls':>7} {'peak MiB':>9}"
        )
        print(header)
        print("-" * len(header))
        ideal = args.reads * args.block
        for name, fn in strategies:
            best, _med, served, calls, peak = _run(fn, state, args.repeat)
            print(
                f"{name:<22} {best:>9.4f} {served / 1048576:>11.2f} "
                f"{served / ideal:>5.0f}x {calls:>7} {peak / 1048576:>9.2f}"
            )
        print(
            f"\n(reads={args.reads}, block={args.block} B)\n"
            "What matters on a real network is round-trips (calls) and "
            "memory, not loopback time:\n"
            "  - opened+unpaged: fewest bytes & lowest memory, but one call "
            "per read (latency x calls).\n"
            "  - larger page: fewer calls (cache coalescing) at the cost of "
            "bytes + memory per page — better on high-latency links.\n"
            "  - non-opened (whole): one call but the whole object resident "
            "in memory and on the wire.\n"
            "Pick the page grain to trade calls against bytes/memory for "
            "your access pattern."
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
