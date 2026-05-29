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
            length = int(self.headers.get("Content-Length", "0"))
            state.blob = self.rfile.read(length) if length else b""
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):  # noqa: N802
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


def _volume(host: str) -> VolumePath:
    return VolumePath(
        "/Volumes/bench/seek/vol/data.bin", service=_Service(_Client(host)),
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _run_mode(vol: VolumePath, state: _State, offsets, block: int) -> tuple[float, int, int]:
    state.bytes_served = 0
    state.requests = 0
    total = len(state.blob)
    t0 = time.perf_counter()
    for off in offsets:
        n = min(block, total - off)
        mv = vol._read_mv(n, off)
        assert len(mv) == n
    elapsed = time.perf_counter() - t0
    return elapsed, state.bytes_served, state.requests


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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        vol = _volume(host)
        rng = random.Random(args.seed + 1)
        max_off = len(state.blob) - args.block
        offsets = [rng.randint(0, max_off) for _ in range(args.reads)]

        results: dict[str, list[tuple[float, int, int]]] = {"range": [], "full": []}
        for mode in ("range", "full"):
            state.honor_range = mode == "range"
            for _ in range(args.repeat):
                results[mode].append(_run_mode(vol, state, offsets, args.block))

        print(
            f"\nVolumePath random-seek reads — "
            f"payload={args.size_mib} MiB, block={args.block} B, "
            f"reads={args.reads}, repeat={args.repeat}\n"
        )
        header = f"{'mode':<8} {'best (s)':>10} {'median (s)':>12} {'MiB served':>12} {'amplification':>14}"
        print(header)
        print("-" * len(header))
        ideal = args.reads * args.block
        for mode in ("range", "full"):
            times = [r[0] for r in results[mode]]
            served = results[mode][0][1]
            print(
                f"{mode:<8} {min(times):>10.4f} {statistics.median(times):>12.4f} "
                f"{served / 1024 / 1024:>12.2f} {served / ideal:>13.1f}x"
            )

        rbest = min(r[0] for r in results["range"])
        fbest = min(r[0] for r in results["full"])
        rserved = results["range"][0][1]
        fserved = results["full"][0][1]
        print(
            f"\nrange transfers {fserved / max(rserved, 1):.0f}x fewer bytes "
            f"and is {fbest / max(rbest, 1e-9):.1f}x faster on loopback "
            f"(the gap widens on a real network where bytes dominate)."
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
