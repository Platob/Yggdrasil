"""Heavy benchmark for :class:`yggdrasil.http_.response_cache.HttpResponseCache`.

Answers one question: is a cache hit at least as fast as a real HTTP call, with
less memory than the old generic (Hive-partitioned Arrow dataset) cache?

Three contenders, same N responses, same payloads:

* **real http** — ``HTTPSession.send`` against a localhost keep-alive server
  (the rawest "real call" baseline — loopback, no network latency to hide
  behind, so the cache has to actually beat request dispatch + parse);
* **new cache** — :class:`HttpResponseCache` (one content-addressed Arrow-IPC
  *body* file per request, schema reattached from the package);
* **old cache** — the generic path: responses written to a :class:`Folder` as a
  partitioned dataset, read back with a predicate scan
  (``CacheConfig`` with an explicit ``tabular=Folder``).

Reports per-op wall time (single + bulk), throughput, the cache↔http speedup,
and tracemalloc peak memory for a bulk read.

Usage::

    PYTHONPATH=src python benchmarks/http_/bench_response_cache.py
    PYTHONPATH=src python benchmarks/http_/bench_response_cache.py --n 4000 --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src"),
)

from yggdrasil.http_ import HTTPSession                       # noqa: E402
from yggdrasil.http_.cache_config import CacheConfig          # noqa: E402
from yggdrasil.http_.request import PreparedRequest           # noqa: E402
from yggdrasil.http_.response import HTTPResponse as Response  # noqa: E402
from yggdrasil.http_.response_cache import HttpResponseCache   # noqa: E402
from yggdrasil.path.folder import Folder                      # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _bench_http_server import start_bench_server             # noqa: E402

SIZES = ("tiny", "kib1", "kib64")


def _median_ms(fn, reps):
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(times)


def _fetch_responses(session, base_url, size, n):
    """Pull N distinct responses off the server (also = the real-http baseline)."""
    reqs = [PreparedRequest.prepare("GET", f"{base_url}/{size}?i={i}") for i in range(n)]
    responses = [session.send(r, raise_error=False) for r in reqs]
    for r in responses:                       # materialise bodies before caching
        _ = r.content
    return reqs, responses


def _bench_size(session, base_url, size, n, reps, tmp):
    reqs, responses = _fetch_responses(session, base_url, size, n)
    payload = len(responses[0].content)

    # --- populate both caches -------------------------------------------
    new_cache = HttpResponseCache(path=str(tmp / f"new_{size}"))
    new_cache.write_arrow(Response.values_to_arrow_batch(responses))

    old_cfg = CacheConfig(tabular=Folder(path=str(tmp / f"old_{size}")))
    old_cfg.write_responses(responses)
    read_cfg = CacheConfig()

    # warm one read each (folder schema cache / dir stat)
    new_cache.read_responses(reqs[:1], config=read_cfg)
    old_cfg.read_responses(reqs[:1])

    # --- single-op latency ----------------------------------------------
    one = reqs[len(reqs) // 2]
    http_one = _median_ms(
        lambda: session.send(
            PreparedRequest.prepare("GET", f"{base_url}/{size}?probe=1"),
            raise_error=False,
        ), reps,
    )
    new_one = _median_ms(lambda: new_cache.read_responses([one], config=read_cfg), reps)
    old_one = _median_ms(lambda: old_cfg.read_responses([one]), reps)

    # --- bulk read (all N) ----------------------------------------------
    new_bulk = _median_ms(lambda: new_cache.read_responses(reqs, config=read_cfg), reps)
    old_bulk = _median_ms(lambda: old_cfg.read_responses(reqs), reps)

    # --- write throughput -----------------------------------------------
    batch = Response.values_to_arrow_batch(responses)
    new_write = _median_ms(lambda: new_cache.write_arrow(batch), max(2, reps // 2))
    old_write = _median_ms(lambda: old_cfg.write_responses(responses), max(2, reps // 2))

    # --- memory (bulk read peak) ----------------------------------------
    gc.collect(); tracemalloc.start()
    new_cache.read_responses(reqs, config=read_cfg)
    new_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    gc.collect(); tracemalloc.start()
    old_cfg.read_responses(reqs)
    old_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # --- the cache's own resident RAM (the hot tier — hard-capped) ------
    from yggdrasil.http_.response_cache import _ram, _RAM_MAX_BYTES
    ram_tier = _ram._bytes

    # --- on-disk footprint ----------------------------------------------
    new_disk = sum(f.stat().st_size for f in (tmp / f"new_{size}").rglob("*") if f.is_file())
    old_disk = sum(f.stat().st_size for f in (tmp / f"old_{size}").rglob("*") if f.is_file())

    print(f"\n── {size}  (payload {payload:,} B · N={n}) "
          f"{'─' * 30}")
    print(f"  single-op   real-http {http_one:8.3f} ms | "
          f"new {new_one:7.3f} ms | old {old_one:7.3f} ms")
    print(f"              new vs http: {http_one / new_one:5.1f}x faster | "
          f"new vs old: {old_one / new_one:5.1f}x faster")
    print(f"  bulk N      new {new_bulk:8.2f} ms ({n / (new_bulk/1000):,.0f}/s) | "
          f"old {old_bulk:8.2f} ms ({n / (old_bulk/1000):,.0f}/s) | "
          f"{old_bulk / new_bulk:4.1f}x")
    print(f"  write N     new {new_write:8.2f} ms | old {old_write:8.2f} ms | "
          f"{old_write / new_write:4.1f}x")
    print(f"  read peak   new {new_peak/1e6:8.2f} MB | old {old_peak/1e6:8.2f} MB | "
          f"{old_peak / max(new_peak,1):4.1f}x lighter "
          f"(incl. the {n}-response result the caller asked for)")
    print(f"  RAM tier    new {ram_tier/1e6:8.2f} MB  (hard cap {_RAM_MAX_BYTES/1e6:.0f} MB — "
          f"the cache's own resident memory, can't balloon)")
    print(f"  on disk     new {new_disk/1e6:8.2f} MB | old {old_disk/1e6:8.2f} MB")
    return http_one, new_one


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="responses per size")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args(argv)

    server, _thread, base_url = start_bench_server()
    session = HTTPSession(base_url=base_url)
    import tempfile
    tmp = Path(args.tmp or tempfile.mkdtemp(prefix="ygg-rcbench-"))

    print(f"HttpResponseCache benchmark — server {base_url}, N={args.n}, "
          f"repeat={args.repeat}, tmp={tmp}")
    try:
        wins = []
        for size in SIZES:
            http_one, new_one = _bench_size(session, base_url, size, args.n, args.repeat, tmp)
            wins.append(http_one / new_one)
        print(f"\n══ summary: cache hit is {statistics.median(wins):.1f}x faster than a "
              f"localhost HTTP call (median across sizes) ══")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
