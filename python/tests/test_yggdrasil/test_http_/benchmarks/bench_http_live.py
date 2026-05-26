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

    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_live.py
    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_live.py --repeat 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

_BENCH_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCH_DIR.parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
os.environ["PYTHONPATH"] = (
    str(_PROJECT_ROOT)
    + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
)

from yggdrasil.http_ import HTTPSession  # noqa: E402
from yggdrasil.io.request import PreparedRequest  # noqa: E402

from _bench_http_server import (  # noqa: E402
    start_bench_server as _start_server,
)


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
    sess = HTTPSession(base_url=base_url)

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
    sess = HTTPSession(base_url=base_url)
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

    # send_many through a populated local cache — exercises the
    # staged pipeline's hot path (the snapshot loop above stage 1,
    # the single-pass classification in _split_remote_cache, the
    # HTTPResponseBatch flatten). Distinct URLs so the cache reads N
    # different rows per call rather than one row N times — that
    # surfaces the per-request resolution cost more honestly.
    cache_n = 128
    tmp_cache_many = tempfile.mkdtemp(prefix="ygg-bench-many-cache-")
    try:
        from yggdrasil.http_.cache_config import CacheConfig

        cfg_many = CacheConfig(tabular=tmp_cache_many)
        cache_reqs = [
            PreparedRequest.prepare("GET", f"{base_url}/tiny?x={i}")
            for i in range(cache_n)
        ]
        # Seed by sending each request once; the session writes them
        # to the cache. The ``time.sleep`` drains the fire-and-forget
        # writer jobs so the warm pass below finds every entry.
        for r in cache_reqs:
            sess.send(r, local_cache=cfg_many, raise_error=False)
        time.sleep(0.5)

        def _send_many_cache_hits():
            return list(sess.send_many(
                iter(cache_reqs),
                local_cache=cfg_many,
                batch_size=cache_n,
                raise_error=False,
            ))

        out.append(_time_one(
            f"HTTPSession.send_many local-cache HITs (n={cache_n})",
            _send_many_cache_hits,
            repeat=repeat, inner=5,
        ))
    finally:
        shutil.rmtree(tmp_cache_many, ignore_errors=True)

    return out


def _local_cache_scenarios(base_url: str, repeat: int) -> list[dict]:
    """Compare bare ``HTTPSession.send`` against a session with the local
    on-disk cache turned on. Cache hits dodge the entire socket round trip,
    so the delta is the visible win the on-disk fast-path buys.

    Includes a 2 MiB JSON scenario to exercise the cache-persist
    auto-compress helper: on the MISS the body is gzipped before the
    Arrow IPC write (much smaller on-disk row); on the HIT the
    compressed buffer is materialised and ``Response.content`` /
    ``.json()`` decompress transparently through the existing
    :class:`Codec` path. The numbers expose the compress cost on
    store, the decompress cost on read, and the dispatch baseline
    for a large compressed payload.
    """
    out: list[dict] = []

    # Bare session for the baseline. Same singleton-cache key it had
    # in ``_wire_scenarios`` — singleton-by-key means the pool stays
    # hot across scenarios.
    sess = HTTPSession(base_url=base_url)
    tmp = tempfile.mkdtemp(prefix="ygg-bench-live-cache-")
    try:
        from yggdrasil.http_.cache_config import CacheConfig

        cfg = CacheConfig(tabular=tmp)
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

        # 2 MiB JSON path — above the auto-compress threshold. Seed
        # one entry then time the HIT (decompress cost) and the MISS
        # (compress + store cost) at smaller inner counts since each
        # call moves ~2 MiB.
        big_seed = PreparedRequest.prepare("GET", f"{base_url}/mib2")
        sess.send(big_seed, local_cache=cfg, raise_error=False)
        big_warm = PreparedRequest.prepare("GET", f"{base_url}/mib2")
        _ = big_warm.public_hash
        out.append(_time_one(
            "HTTPSession.send mib2 (local-cache HIT, auto-gzip body)",
            lambda: sess.send(big_warm, local_cache=cfg, raise_error=False),
            repeat=repeat, inner=50,
        ))

        big_counter = {"n": 0}

        def _send_big_miss():
            big_counter["n"] += 1
            r = PreparedRequest.prepare(
                "GET", f"{base_url}/mib2?probe={big_counter['n']}",
            )
            return sess.send(r, local_cache=cfg, raise_error=False)
        out.append(_time_one(
            "HTTPSession.send mib2 (local-cache MISS, compress + store)",
            _send_big_miss,
            repeat=repeat, inner=50,
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
