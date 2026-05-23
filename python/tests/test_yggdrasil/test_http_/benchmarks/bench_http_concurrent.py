"""HTTPSession.send_many concurrency bench against a localhost server.

What this covers
----------------

Drives :meth:`HTTPSession.send_many` over batches of varying size against
the shared localhost fixture in :mod:`benchmarks.io._bench_http_server`.
The numbers show how a single ``send_many`` call scales as the request
chunk grows:

* batch_size 1 (one-shot — measures pure dispatch overhead, the floor
  the bigger batches have to beat);
* batch_size 10 (small batch — JobPoolExecutor fan-out kicks in);
* batch_size 100 (saturating the default 8-socket pool — workers wait
  on free sockets behind the pool's ``block=True`` semantics);
* batch_size 1000 (firmly past saturation — the pool's fairness +
  throughput per fixed connection count is what's being measured).

What to read out of it
----------------------

For each batch size the bench reports:

* ``total_s`` — wall-clock to drain the whole batch;
* ``per_req_us`` — total time divided by batch size (the per-request
  amortised cost, the headline number);
* ``rps`` — total requests / total seconds (the throughput number).

If ``per_req_us`` at batch_size 1 ≈ per_req_us at batch_size 1000, the
job pool is paying for itself but not winning — usually a sign the
underlying transport (pool size, TLS, response holder copy) is the
bottleneck rather than dispatch. If ``rps`` plateaus well before the
pool saturates, the localhost server is the cap — bench against a real
service to learn what the pool can actually do.

Usage::

    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_concurrent.py
    PYTHONPATH=src python tests/test_yggdrasil/test_http_/benchmarks/bench_http_concurrent.py --repeat 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

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
from yggdrasil.io.send_config import SendManyConfig  # noqa: E402

from _bench_http_server import start_bench_server  # noqa: E402


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

#: Payload route → human label. Pinned to /kib1 — small enough that the
#: server isn't the bottleneck, big enough that the response body is a
#: real (non-empty) hop through the holder. Use /tiny for dispatch-cost
#: stress; use /kib64 to surface buffer-copy regressions.
DEFAULT_ROUTE = "kib1"

#: Batch sizes to sweep. The progression spans "one-shot dispatch" to
#: "well past the default 8-socket pool" so the per-request amortised
#: cost surfaces every regime: pure overhead, fan-out win, pool-block
#: floor.
BATCH_SIZES: tuple[int, ...] = (1, 10, 100, 1000)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _build_requests(base_url: str, route: str, n: int) -> list[PreparedRequest]:
    """Return ``n`` PreparedRequests pointing at ``<base_url>/<route>``.

    Same URL on every request — keeps the pool warm so we measure
    ``send_many`` dispatch rather than DNS / TCP handshake. Cache-aware
    paths are intentionally bypassed: the bench wires ``raise_error=False``
    and no ``CacheConfig`` so every miss falls straight to the wire.
    """
    url = f"{base_url}/{route}"
    return [
        PreparedRequest.prepare(method="GET", url=url, normalize=False)
        for _ in range(n)
    ]


def _drain_send_many(session: HTTPSession, requests: list[PreparedRequest]) -> int:
    """Send the batch and count drained responses.

    Iterates the ``send_many`` generator to completion so the worker
    pool actually fans out — leaving it un-iterated would only build
    the pipeline plan and exit. Returns the response count so the
    caller can sanity-check the run.
    """
    cfg = SendManyConfig.check_arg(None, raise_error=False)
    count = 0
    for _ in session.send_many(iter(requests), cfg):
        count += 1
    return count


def _time_batch(
    session: HTTPSession,
    base_url: str,
    route: str,
    batch_size: int,
    *,
    repeat: int,
) -> dict:
    # One warmup pass to let the pool's connection slots fill and any
    # JIT-shaped allocation churn settle. Without it the first sample
    # consistently runs 5–10× the steady state.
    _drain_send_many(session, _build_requests(base_url, route, batch_size))

    samples: list[float] = []
    drained: list[int] = []
    for _ in range(repeat):
        reqs = _build_requests(base_url, route, batch_size)
        t0 = time.perf_counter()
        n = _drain_send_many(session, reqs)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        drained.append(n)

    best = min(samples)
    median = statistics.median(samples)
    return {
        "batch_size": batch_size,
        "best_total_s": best,
        "median_total_s": median,
        "best_per_req_us": best * 1e6 / max(batch_size, 1),
        "median_per_req_us": median * 1e6 / max(batch_size, 1),
        "best_rps": batch_size / best if best > 0 else float("inf"),
        "median_rps": batch_size / median if median > 0 else float("inf"),
        "drained_min": min(drained),
        "drained_max": max(drained),
    }


def _format_row(result: dict) -> str:
    return (
        f"batch={result['batch_size']:<5} "
        f"best={result['best_total_s']*1e3:>9.2f} ms  "
        f"median={result['median_total_s']*1e3:>9.2f} ms  "
        f"per_req(best)={result['best_per_req_us']:>9.1f} µs  "
        f"rps(best)={result['best_rps']:>10.0f}  "
        f"drained={result['drained_min']}"
        + (f"-{result['drained_max']}" if result['drained_min'] != result['drained_max'] else "")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Outer repeat count per batch size (median across).",
    )
    parser.add_argument(
        "--route",
        default=DEFAULT_ROUTE,
        choices=("tiny", "kib1", "kib64", "mib2"),
        help="Payload route — picks the response size each request returns.",
    )
    parser.add_argument(
        "--pool-maxsize",
        type=int,
        default=None,
        help=(
            "HTTPSession pool_maxsize. Default lets HTTPSession clamp "
            "to its built-in 8-socket ceiling — the realistic shape."
        ),
    )
    args = parser.parse_args()

    server, _thread, base_url = start_bench_server()
    print(
        f"# bench_http_concurrent — route=/{args.route} "
        f"repeat={args.repeat} server={base_url}"
    )
    try:
        kwargs: dict = {"base_url": base_url}
        if args.pool_maxsize is not None:
            kwargs["pool_maxsize"] = args.pool_maxsize
        session = HTTPSession(**kwargs)
        for batch_size in BATCH_SIZES:
            result = _time_batch(
                session, base_url, args.route, batch_size, repeat=args.repeat,
            )
            print(_format_row(result))
    finally:
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
