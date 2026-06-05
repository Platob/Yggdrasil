"""Connection-pool reuse bench — quantifies what keep-alive pooling buys.

What this covers
----------------

:class:`HTTPSession` *is* its own connection pool: a drained response parks
its socket back into a per-host idle deque, so the next request to the same
host skips the TCP (and, for HTTPS, the TLS) handshake. This bench isolates
that win by timing the same serial request loop two ways against the shared
localhost fixture in :mod:`benchmarks.http_._bench_http_server`:

* ``warm``  — the pool is left intact between requests (steady-state reuse);
* ``cold``  — :meth:`HTTPSession.clear_connections` is called before every
  request, forcing a fresh dial each time (the no-pool baseline).

What to read out of it
----------------------

For each mode the bench reports per-request median/best latency. The
``speedup`` line is the headline: ``cold / warm`` per-request time — how many
× the idle-socket cache saves on a connection-reuse-friendly workload. On
loopback (no TLS, sub-ms handshake) the gap is modest; the same bench against
an HTTPS endpoint with real RTT shows the pool earning multiples. If ``warm``
is *not* faster than ``cold`` on loopback, the pool's park/pop path has
regressed into doing real work.

Usage::

    PYTHONPATH=src python benchmarks/http_/bench_http_pool.py
    PYTHONPATH=src python benchmarks/http_/bench_http_pool.py --repeat 7 --requests 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from yggdrasil.http_ import HTTPSession  # noqa: E402
from yggdrasil.http_.request import HTTPRequest  # noqa: E402

from _bench_http_server import start_bench_server  # noqa: E402


DEFAULT_ROUTE = "kib1"


def _serial_loop(session: HTTPSession, url: str, n: int, *, cold: bool) -> int:
    """Fire ``n`` requests serially; clear the pool each iteration when ``cold``."""
    count = 0
    for _ in range(n):
        if cold:
            session.clear_connections()
        resp = session.send(HTTPRequest.prepare("GET", url))
        if resp.status_code == 200:
            count += 1
    return count


def _time_mode(
    session: HTTPSession, url: str, n: int, *, cold: bool, repeat: int,
) -> dict:
    _serial_loop(session, url, min(n, 10), cold=cold)  # warmup
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        got = _serial_loop(session, url, n, cold=cold)
        samples.append(time.perf_counter() - t0)
        assert got == n, f"expected {n} responses, drained {got}"
    best = min(samples)
    median = statistics.median(samples)
    return {
        "mode": "cold" if cold else "warm",
        "best_per_req_us": best * 1e6 / n,
        "median_per_req_us": median * 1e6 / n,
    }


def _format_row(r: dict) -> str:
    return (
        f"mode={r['mode']:<5} "
        f"per_req(best)={r['best_per_req_us']:>9.1f} µs  "
        f"per_req(median)={r['median_per_req_us']:>9.1f} µs"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument(
        "--route", default=DEFAULT_ROUTE,
        choices=("tiny", "kib1", "kib64", "mib2"),
    )
    args = parser.parse_args()

    server, _thread, base_url = start_bench_server()
    url = f"{base_url}/{args.route}"
    print(
        f"# bench_http_pool — route=/{args.route} requests={args.requests} "
        f"repeat={args.repeat} server={base_url}"
    )
    try:
        HTTPSession._INSTANCES.clear()
        session = HTTPSession(base_url=base_url)
        warm = _time_mode(session, url, args.requests, cold=False, repeat=args.repeat)
        cold = _time_mode(session, url, args.requests, cold=True, repeat=args.repeat)
        print(_format_row(warm))
        print(_format_row(cold))
        speedup = cold["best_per_req_us"] / warm["best_per_req_us"]
        print(f"speedup(cold/warm, best per-req) = {speedup:.2f}x")
    finally:
        HTTPSession._INSTANCES.clear()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
