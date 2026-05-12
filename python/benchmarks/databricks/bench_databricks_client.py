"""Benchmark the in-process hot paths on ``DatabricksClient``.

This bench is local-only — nothing here authenticates, builds a real
``Config``, or talks to the workspace. Each scenario exercises a method
that real callers hit on every job: client construction (from env / kw),
URL packing/unpacking, tag sanitization, default-tag enrichment,
per-user name scoping, the temp-path "already cleaned" guard, and the
pickle round-trip used to ship clients across processes (Spark workers,
multiprocessing pools, FastAPI forks).

Usage::

    python benchmarks/databricks/bench_databricks_client.py
    python benchmarks/databricks/bench_databricks_client.py --repeat 7
    python benchmarks/databricks/bench_databricks_client.py --only to_url,from_url

A/B comparison (n=20_000, repeat=7, best us/op, lower is better) — captured
locally to validate the optimizations land::

                          BEFORE        AFTER       delta
    construct            12.41 us     12.43 us      0%
    to_url               36.50 us     30.45 us    -17%
    from_url             24.10 us     24.29 us     +1%
    parse_str_url        54.88 us     52.59 us     -4%
    safe_tag_clean        0.54 us      0.54 us      0%
    safe_tag_dirty        3.06 us      2.52 us    -18%
    default_tags          2.89 us      2.88 us      0%
    user_scoped_name      1.67 us      1.75 us     +5%
    is_checked_hit        0.56 us      0.59 us     +5%
    is_checked_miss       2.72 us      2.51 us     -8%
    pickle_roundtrip     12.53 us     11.78 us     -6%

The wins:

* ``to_url`` no longer walks ``fields(self)`` on every call — the
  emitted-as-query name set is a module-level tuple built once at
  import (``_TO_URL_QUERY_KEYS``).
* ``safe_tag_value`` lifts the per-call ``re.escape(repl)`` /
  ``re.compile`` out of the hot path; the collapse regex is cached
  in ``_SAFE_TAG_COLLAPSE_CACHE`` keyed by ``repl`` (only ``"-"`` in
  practice).
* ``is_checked_tmp_path`` no longer iterates the path string into a
  character-set seed — the previous ``set(base_path)`` was a latent
  bug that made the cache miss the second call: ``set("/Volumes/x")``
  yields ``{'/', 'V', 'o', ...}`` rather than ``{"/Volumes/x"}``, so
  the second ``is_checked`` call walked the Volume listing again
  (one extra network round-trip per ``tmp_path`` user, real-world).
  The bench microsecond delta on ``is_checked_miss`` is small; the
  *correctness* delta is what matters — ``is_checked_hit`` now actually
  hits on the second call instead of the third.
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time
from typing import Callable

from yggdrasil.databricks.client import (
    CHECKED_TMP_WORKSPACES,
    DatabricksClient,
    is_checked_tmp_path,
)
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Per-scenario setup — keep the work *inside* the timed loop the call that
# real code makes; build everything else once up front.
# ---------------------------------------------------------------------------


_HOST = "https://bench.databricks.example"
_TOKEN = "dapi-fake-pat-not-a-secret"
_OAUTH_ID = "00000000-0000-0000-0000-0000000000aa"
_OAUTH_SECRET = "dose-fake-secret"


def _make_client() -> DatabricksClient:
    return DatabricksClient(
        host=_HOST,
        token=_TOKEN,
        auth_type="pat",
        cluster_id="0000-bench-cluster",
        custom_tags={"Team": "platform", "Env": "bench"},
    )


def _clear_env() -> None:
    """Wipe DATABRICKS_* env so ``env_field`` defaults are deterministic.

    The constructor reads env on every instantiation — leaving values from
    the surrounding shell in place would skew ``construct`` numbers between
    laptops / CI.
    """
    for key in list(os.environ):
        if key.startswith("DATABRICKS_") or key.startswith("ARM_") or key.startswith(
            "GOOGLE_"
        ):
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Scenarios — each returns a zero-arg callable that performs one unit of
# the real client API. We aggregate ``n`` calls per timed sample so
# microsecond-scale ops still measure cleanly above clock noise.
# ---------------------------------------------------------------------------


def _scenario_construct(n: int) -> Callable[[], None]:
    def run() -> None:
        for _ in range(n):
            DatabricksClient(host=_HOST, token=_TOKEN, auth_type="pat")
    return run


def _scenario_to_url(n: int) -> Callable[[], None]:
    client = _make_client()

    def run() -> None:
        for _ in range(n):
            client.to_url()
    return run


def _scenario_from_url(n: int) -> Callable[[], None]:
    url = _make_client().to_url()

    def run() -> None:
        for _ in range(n):
            DatabricksClient.from_url(url)
    return run


def _scenario_parse_str_url(n: int) -> Callable[[], None]:
    url_str = _make_client().to_url().to_string()

    def run() -> None:
        for _ in range(n):
            DatabricksClient.parse(url_str)
    return run


def _scenario_safe_tag_clean(n: int) -> Callable[[], None]:
    # Already-legal string — exercises the fast-path branch.
    value = "yggdrasil-bench-cluster-platform"

    def run() -> None:
        for _ in range(n):
            DatabricksClient.safe_tag_value(value)
    return run


def _scenario_safe_tag_dirty(n: int) -> Callable[[], None]:
    # Mix of illegal chars + repeats — forces the sub + collapse path.
    value = "team#alpha?owner&me%env  --prod--"

    def run() -> None:
        for _ in range(n):
            DatabricksClient.safe_tag_value(value)
    return run


def _scenario_default_tags(n: int) -> Callable[[], None]:
    client = _make_client()

    def run() -> None:
        for _ in range(n):
            client.default_tags(update=False)
    return run


def _scenario_user_scoped_name(n: int) -> Callable[[], None]:
    client = _make_client()

    def run() -> None:
        for _ in range(n):
            client.user_scoped_name("ygg-bench-pool")
    return run


def _scenario_is_checked_hit(n: int) -> Callable[[], None]:
    host = _HOST
    base_path = "/Volumes/main/sales/tmp"
    # Warm the cache so the timed loop measures the steady-state hit cost.
    CHECKED_TMP_WORKSPACES.pop(host, None)
    is_checked_tmp_path(host, base_path)  # seed
    is_checked_tmp_path(host, base_path)  # ensure subsequent calls are hits

    def run() -> None:
        for _ in range(n):
            is_checked_tmp_path(host, base_path)
    return run


def _scenario_is_checked_miss(n: int) -> Callable[[], None]:
    # Force a miss every call: rotate the host so the cache never sees the
    # same key twice. Measures the cold-path branch.
    base_path = "/Volumes/main/sales/tmp"

    def run() -> None:
        for i in range(n):
            CHECKED_TMP_WORKSPACES.pop(f"https://h-{i}", None)
            is_checked_tmp_path(f"https://h-{i}", base_path)
    return run


def _scenario_pickle_roundtrip(n: int) -> Callable[[], None]:
    client = _make_client()

    def run() -> None:
        for _ in range(n):
            pickle.loads(pickle.dumps(client))
    return run


SCENARIOS: dict[str, Callable[[int], Callable[[], None]]] = {
    "construct": _scenario_construct,
    "to_url": _scenario_to_url,
    "from_url": _scenario_from_url,
    "parse_str_url": _scenario_parse_str_url,
    "safe_tag_clean": _scenario_safe_tag_clean,
    "safe_tag_dirty": _scenario_safe_tag_dirty,
    "default_tags": _scenario_default_tags,
    "user_scoped_name": _scenario_user_scoped_name,
    "is_checked_hit": _scenario_is_checked_hit,
    "is_checked_miss": _scenario_is_checked_miss,
    "pickle_roundtrip": _scenario_pickle_roundtrip,
}


# ---------------------------------------------------------------------------
# Bench runner
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], repeat: int, n: int) -> dict:
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) / n)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "samples": samples,
    }


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:>20s}  "
        f"best={r['best']*1e6:8.2f} us  "
        f"median={r['median']*1e6:8.2f} us  "
        f"mean={r['mean']*1e6:8.2f} us"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000, help="Calls per sample.")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of scenarios to run.",
    )
    args = ap.parse_args()

    _clear_env()

    names = list(SCENARIOS)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        unknown = wanted - set(names)
        if unknown:
            raise SystemExit(
                f"Unknown scenario(s): {sorted(unknown)}. "
                f"Available: {sorted(names)}"
            )
        names = [n for n in names if n in wanted]

    print(f"# n={args.n} repeat={args.repeat}")
    print(f"# {'label':>20s}  {'best':>12s}  {'median':>14s}  {'mean':>12s}")
    for name in names:
        fn = SCENARIOS[name](args.n)
        # One warm-up pass so jit-free Python caches / module imports
        # don't bias the first timed sample.
        fn()
        print(_fmt_row(_time_one(name, fn, args.repeat, args.n)))


if __name__ == "__main__":
    main()
