"""Benchmark the in-process hot paths on ``DatabricksClient``.

This bench is local-only — nothing here authenticates, builds a real
``Config``, or talks to the workspace. Each scenario exercises a method
that real callers hit on every job: client construction (from env / kw),
the singleton hit / miss path that drives every cached lookup, sub-service
``lazy_property`` access, URL packing/unpacking, tag sanitization,
default-tag enrichment, per-user name scoping, the temp-path "already
cleaned" guard, and the pickle round-trip used to ship clients across
processes (Spark workers, multiprocessing pools, FastAPI forks).

Usage::

    python benchmarks/databricks/bench_databricks_client.py
    python benchmarks/databricks/bench_databricks_client.py --repeat 7
    python benchmarks/databricks/bench_databricks_client.py --only singleton_hit

A/B comparison (n=5_000, repeat=5, best us/op, lower is better) — captured
locally to validate the singleton optimizations land::

                            BEFORE        AFTER       delta
    construct              26.21 us      7.18 us    -73%
    singleton_hit          26.02 us      7.41 us    -72%
    singleton_miss         27.42 us     18.29 us    -33%
    singleton_key          24.44 us      4.70 us    -81%
    to_url                 32.63 us     25.35 us    -22%
    from_url               46.81 us     16.07 us    -66%
    parse_str_url          86.85 us     38.31 us    -56%
    lazy_property_hit       1.55 us      0.12 us    -92%
    lazy_property_chain    (n/a)         0.45 us     —
    pickle_roundtrip       35.06 us     27.25 us    -22%
    singleton_pickle_hit   (n/a)        17.85 us     —

The wins:

* Env defaults are snapshotted once into a module-level dict
  (``_env_defaults_snapshot``) instead of paying ~14 us of
  ``os.getenv`` per constructor call. Every singleton-hit /
  miss / ``from_url`` build benefits.
* ``_singleton_key`` no longer calls ``sorted()`` on the resolved
  kwargs — the dict already lands in the fixed insertion order of
  ``_ENV_DEFAULTS`` + ``_STATIC_DEFAULTS``, and ``custom_tags`` (the
  only field that was ever a dict) is gone, so the items tuple is
  hashable as-is.
* The ``custom_tags`` field is dropped: callers who want resource
  tags use :meth:`DatabricksClient.default_tags` (which already
  pulls owner / product info), and the SDK's per-resource
  ``custom_tags`` (cluster, warehouse, instance pool) keeps working
  unchanged.
* Sub-service properties (``client.sql``, ``client.tables``,
  ``client.warehouses``, …) inline the cache lookup through
  ``self.__dict__`` — one dict get + one dict set, no
  ``getattr`` descriptor walk, no lambda allocation per call.

Earlier (pre-singleton) wins kept:

* ``to_url`` no longer walks ``fields(self)`` on every call.
* ``safe_tag_value`` caches the collapse regex per ``repl``.
* ``is_checked_tmp_path`` seeds the cache with ``{base_path}``,
  not ``set(base_path)`` (which iterated the string into chars).
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


def _scenario_singleton_hit(n: int) -> Callable[[], None]:
    """Same kwargs → cached singleton. Drives the env-snapshot fast path."""
    DatabricksClient._INSTANCES.clear()
    DatabricksClient(
        host=_HOST, token=_TOKEN, auth_type="pat",
        cluster_id="0000-bench-cluster", singleton_ttl=None,
    )

    def run() -> None:
        for _ in range(n):
            DatabricksClient(
                host=_HOST, token=_TOKEN, auth_type="pat",
                cluster_id="0000-bench-cluster", singleton_ttl=None,
            )
    return run


def _scenario_singleton_miss(n: int) -> Callable[[], None]:
    """Rotate ``host`` to force a fresh singleton entry on every build."""
    def run() -> None:
        for i in range(n):
            DatabricksClient._INSTANCES.clear()
            DatabricksClient(
                host=f"https://h-{i}.example", token=_TOKEN, auth_type="pat",
                singleton_ttl=None,
            )
    return run


def _scenario_singleton_pickle_hit(n: int) -> Callable[[], None]:
    """Pickle round-trip that collapses to the live in-process singleton.

    The cross-process restore path normally rebuilds; with the singleton
    cache populated, ``__new__`` finds the live entry and ``__setstate__``
    short-circuits — the receiver pays only the unpickle decode cost.
    """
    DatabricksClient._INSTANCES.clear()
    client = DatabricksClient(
        host=_HOST, token=_TOKEN, auth_type="pat",
        cluster_id="0000-bench-cluster", singleton_ttl=None,
    )
    blob = pickle.dumps(client)

    def run() -> None:
        for _ in range(n):
            pickle.loads(blob)
    return run


def _scenario_lazy_property_hit(n: int) -> Callable[[], None]:
    """Cached ``client.sql`` access — measures the inlined fast path."""
    client = _make_client()
    _ = client.sql  # warm

    def run() -> None:
        for _ in range(n):
            client.sql
    return run


def _scenario_lazy_property_chain(n: int) -> Callable[[], None]:
    """Mixed sub-service cache hits — the realistic per-call shape."""
    client = _make_client()
    _ = client.sql; _ = client.tables; _ = client.warehouses; _ = client.catalogs

    def run() -> None:
        for _ in range(n):
            client.sql
            client.tables
            client.warehouses
            client.catalogs
    return run


def _scenario_singleton_key(n: int) -> Callable[[], None]:
    """Just the key-build cost — no cache lookup, no instance creation."""
    def run() -> None:
        for _ in range(n):
            DatabricksClient._singleton_key(
                host=_HOST, token=_TOKEN, auth_type="pat",
                cluster_id="0000-bench-cluster",
            )
    return run


SCENARIOS: dict[str, Callable[[int], Callable[[], None]]] = {
    "construct": _scenario_construct,
    "singleton_hit": _scenario_singleton_hit,
    "singleton_miss": _scenario_singleton_miss,
    "singleton_key": _scenario_singleton_key,
    "to_url": _scenario_to_url,
    "from_url": _scenario_from_url,
    "parse_str_url": _scenario_parse_str_url,
    "safe_tag_clean": _scenario_safe_tag_clean,
    "safe_tag_dirty": _scenario_safe_tag_dirty,
    "default_tags": _scenario_default_tags,
    "user_scoped_name": _scenario_user_scoped_name,
    "is_checked_hit": _scenario_is_checked_hit,
    "is_checked_miss": _scenario_is_checked_miss,
    "lazy_property_hit": _scenario_lazy_property_hit,
    "lazy_property_chain": _scenario_lazy_property_chain,
    "pickle_roundtrip": _scenario_pickle_roundtrip,
    "singleton_pickle_hit": _scenario_singleton_pickle_hit,
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
