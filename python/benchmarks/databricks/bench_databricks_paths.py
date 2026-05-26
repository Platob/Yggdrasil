"""Benchmark the in-process hot paths on Databricks filesystem paths.

Covers :class:`DBFSPath`, :class:`VolumePath`, :class:`WorkspacePath`,
and the dispatch / coercion / pickle / stat-cache machinery they share
through :class:`DatabricksPath` and :class:`RemotePath`.

None of the scenarios issue SDK calls — the workspace handle is a
``MagicMock`` and stats are pre-seeded into the path's local cache.
This bench measures the per-call wrapper cost real callers pay
*before* (or *instead of*) the network round trip: constructing a
path from a POSIX string in tight loops, rendering ``full_path`` /
``api_path`` for log lines, walking ``parent`` for ancestry checks,
``ls``-time child construction, ``is_file`` / ``exists`` / ``size``
when a listing entry has already seeded the stat cache, pickle
round-trips for Spark / multiprocessing shipping, and the singleton
hit / miss machinery.

Usage::

    python benchmarks/databricks/bench_databricks_paths.py
    python benchmarks/databricks/bench_databricks_paths.py --repeat 7
    python benchmarks/databricks/bench_databricks_paths.py \\
        --only volume_construct_posix,full_path_volume

A/B comparison (n=3_000, repeat=3, best us/op — lower is better)
captured locally to validate the URL-stash optimization
(:class:`DatabricksPath.__new__` hands the normalized URL to
:meth:`__init__` so the second ``URL.from_`` + ``_strip_dbfs_family_prefix``
pass goes away)::

                            BEFORE         AFTER       delta
    dbfs_construct_posix     33.25 us      26.98 us    -19%
    volume_construct_posix   35.43 us      28.91 us    -18%
    workspace_construct_posix 34.01 us     26.87 us    -21%
    dispatch_new             40.18 us      33.45 us    -17%
    singleton_miss_volume    51.64 us      43.68 us    -15%
    volume_construct_url      9.66 us       9.45 us     -2%
    stat_cached_fresh         0.16 us       0.16 us      0% (no regression)
    exists_cached             0.28 us       0.27 us      0% (no regression)

Where the wins come from
------------------------
* ``__new__`` already normalizes a POSIX seed
  (``/Volumes/cat/sch/vol/x`` → ``dbfs+volume:///cat/sch/vol/x``)
  through :func:`_resolve_databricks_subclass`. The result used to
  be discarded — :meth:`__init__` would re-run
  ``_coerce_to_url_str`` + ``URL.from_`` + ``_strip_dbfs_family_prefix``
  on the same string and produce the same URL. The optimized path
  stashes the URL in a process-wide id-keyed dict so the upcoming
  ``__init__`` reads it directly. One ``URL.from_`` per construction
  instead of two.
* The :class:`DatabricksPath` dispatcher (``DatabricksPath(...)`` on
  the abstract base) ALSO normalizes and now stashes the result on
  the concrete target before forwarding — :class:`dispatch_new`
  drops from one-parse-per-layer to one parse total.
* The stash deliberately lives outside ``self.__dict__``: writing the
  URL into the instance dict and then popping it leaves a CPython
  "dummy" slot in the internal hash table that every subsequent
  attribute lookup probes past — measured at +30% on
  ``_stat_cached_fresh`` reads before moving to the module-level
  dict.
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time
from typing import Callable
from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.dbfs_path import DBFSPath
from yggdrasil.databricks.fs.service import DBFSService
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.databricks.path import (
    DatabricksPath,
    _coerce_to_url_str,
    _resolve_databricks_subclass,
    _strip_dbfs_family_prefix,
)
from yggdrasil.databricks.volume.volumes import Volumes
from yggdrasil.databricks.workspaces.service import Workspaces
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Fixtures — keep authentication & SDK access out of the timed loops.
# ---------------------------------------------------------------------------


_HOST = "https://bench.databricks.example"
_TOKEN = "fake-pat-not-a-secret"


def _clear_env() -> None:
    for key in list(os.environ):
        if key.startswith(("DATABRICKS_", "ARM_", "GOOGLE_")):
            os.environ.pop(key, None)


def _make_client(*, with_mock_sdk: bool = True) -> DatabricksClient:
    """Build a :class:`DatabricksClient` whose SDK handle is a mock.

    Path construction touches ``self.service.client``, but nothing in
    the timed scenarios actually issues SDK calls — we stub the
    workspace client so any accidental invocation surfaces loudly.
    Pickle scenarios pass ``with_mock_sdk=False`` because
    ``MagicMock`` is not picklable; the host is rotated so the
    no-mock client doesn't collide with the mock-bearing singleton
    in ``DatabricksClient._INSTANCES``.
    """
    host = _HOST if with_mock_sdk else _HOST.replace("bench", "bench-pickle")
    client = DatabricksClient(
        host=host, token=_TOKEN, auth_type="pat",
        singleton_ttl=None,
    )
    if with_mock_sdk:
        object.__setattr__(client, "_workspace_client", MagicMock())
        object.__setattr__(client, "_workspace_config", MagicMock())
    return client


def _services(*, with_mock_sdk: bool = True):
    client = _make_client(with_mock_sdk=with_mock_sdk)
    return (
        client,
        DBFSService(client=client),
        Volumes(client=client),
        Workspaces(client=client),
    )


def _seed_stat(path, size: int = 1024) -> None:
    """Seed a fresh :class:`IOStats` so cache-hit scenarios stay local."""
    path._persist_stat_cache(IOStats(
        kind=IOKind.FILE, size=size, mtime=time.time(),
    ))


# ---------------------------------------------------------------------------
# Sample data — keep strings constant so allocation noise stays uniform.
# ---------------------------------------------------------------------------


_DBFS_POSIX = "/dbfs/tmp/yggdrasil/bench/file.parquet"
_DBFS_URL_STR = "dbfs+dbfs:///tmp/yggdrasil/bench/file.parquet"
_VOLUME_POSIX = "/Volumes/main/sales/staging/year=2024/month=05/part-00000.parquet"
_VOLUME_URL_STR = (
    "dbfs+volume:///main/sales/staging/year=2024/month=05/part-00000.parquet"
)
_WORKSPACE_POSIX = "/Workspace/Users/bench@example.com/notebooks/run.py"
_WORKSPACE_URL_STR = "dbfs+workspace:///Users/bench@example.com/notebooks/run.py"


# ---------------------------------------------------------------------------
# Construction scenarios — POSIX coerce, URL parse, dispatch.
# ---------------------------------------------------------------------------


def _scenario_dbfs_construct_posix(n: int) -> Callable[[], None]:
    _, dbfs_service, _, _ = _services()

    def run() -> None:
        for _ in range(n):
            DBFSPath(_DBFS_POSIX, service=dbfs_service, singleton_ttl=False)
    return run


def _scenario_volume_construct_posix(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()

    def run() -> None:
        for _ in range(n):
            VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    return run


def _scenario_workspace_construct_posix(n: int) -> Callable[[], None]:
    _, _, _, workspaces = _services()

    def run() -> None:
        for _ in range(n):
            WorkspacePath(_WORKSPACE_POSIX, service=workspaces, singleton_ttl=False)
    return run


def _scenario_volume_construct_url(n: int) -> Callable[[], None]:
    """URL-shaped construction skips the POSIX coercion branch."""
    _, _, volumes, _ = _services()
    url = URL.from_(_VOLUME_URL_STR)

    def run() -> None:
        for _ in range(n):
            VolumePath(url=url, service=volumes, singleton_ttl=False)
    return run


def _scenario_dispatch_from_posix(n: int) -> Callable[[], None]:
    """:meth:`DatabricksPath.from_` on the abstract base — full dispatch."""
    _, _, volumes, _ = _services()

    def run() -> None:
        for _ in range(n):
            DatabricksPath.from_(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    return run


def _scenario_dispatch_from_url(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    url = URL.from_(_VOLUME_URL_STR)

    def run() -> None:
        for _ in range(n):
            DatabricksPath.from_url(url, service=volumes, singleton_ttl=False)
    return run


def _scenario_dispatch_new(n: int) -> Callable[[], None]:
    """``DatabricksPath(...)`` on the abstract base allocates the right
    concrete subclass — exercises ``_resolve_databricks_subclass``."""
    _, _, volumes, _ = _services()

    def run() -> None:
        for _ in range(n):
            DatabricksPath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    return run


def _scenario_coerce_to_url_str_volume(n: int) -> Callable[[], None]:
    def run() -> None:
        for _ in range(n):
            _coerce_to_url_str(_VOLUME_POSIX)
    return run


def _scenario_coerce_to_url_str_passthrough(n: int) -> Callable[[], None]:
    """Already-URL inputs short-circuit — measures the rejection cost."""
    s = _VOLUME_URL_STR

    def run() -> None:
        for _ in range(n):
            _coerce_to_url_str(s)
    return run


def _scenario_strip_dbfs_family_prefix_passthrough(n: int) -> Callable[[], None]:
    """Already-qualified URLs hit the fast-return path."""
    url = URL.from_(_VOLUME_URL_STR)

    def run() -> None:
        for _ in range(n):
            _strip_dbfs_family_prefix(url)
    return run


def _scenario_strip_dbfs_family_prefix_legacy(n: int) -> Callable[[], None]:
    """Un-qualified ``dbfs://`` URLs need the namespace-flip walk."""
    url = URL.from_("dbfs:///Volumes/main/sales/staging/part-0.parquet")

    def run() -> None:
        for _ in range(n):
            _strip_dbfs_family_prefix(url)
    return run


def _scenario_resolve_subclass_str(n: int) -> Callable[[], None]:
    def run() -> None:
        for _ in range(n):
            _resolve_databricks_subclass(data=_VOLUME_POSIX)
    return run


def _scenario_resolve_subclass_url(n: int) -> Callable[[], None]:
    url = URL.from_(_VOLUME_URL_STR)

    def run() -> None:
        for _ in range(n):
            _resolve_databricks_subclass(url=url)
    return run


def _scenario_singleton_hit_volume(n: int) -> Callable[[], None]:
    """Same URL + service → cached singleton; measures the lookup cost."""
    _, _, volumes, _ = _services()
    # Warm the cache.
    DatabricksPath._INSTANCES.clear()
    VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=None)

    def run() -> None:
        for _ in range(n):
            VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=None)
    return run


def _scenario_singleton_miss_volume(n: int) -> Callable[[], None]:
    """Rotate the path to force a fresh entry every call."""
    _, _, volumes, _ = _services()

    def run() -> None:
        for i in range(n):
            DatabricksPath._INSTANCES.clear()
            VolumePath(
                f"/Volumes/main/sales/staging/file-{i}.parquet",
                service=volumes, singleton_ttl=None,
            )
    return run


def _scenario_singleton_key(n: int) -> Callable[[], None]:
    """Just the ``_singleton_key`` build — no allocation, no cache walk."""
    _, _, volumes, _ = _services()

    def run() -> None:
        for _ in range(n):
            VolumePath._singleton_key(_VOLUME_POSIX, service=volumes)
    return run


# ---------------------------------------------------------------------------
# Path-rendering / pure-path scenarios — every log line walks these.
# ---------------------------------------------------------------------------


def _scenario_full_path_volume(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.full_path()
    return run


def _scenario_full_path_workspace(n: int) -> Callable[[], None]:
    _, _, _, workspaces = _services()
    p = WorkspacePath(_WORKSPACE_POSIX, service=workspaces, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.full_path()
    return run


def _scenario_full_path_dbfs(n: int) -> Callable[[], None]:
    _, dbfs_service, _, _ = _services()
    p = DBFSPath(_DBFS_POSIX, service=dbfs_service, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.full_path()
    return run


def _scenario_api_path_volume(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.api_path
    return run


def _scenario_split_volume(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p._split_volume()
    return run


def _scenario_volume_catalog_name(n: int) -> Callable[[], None]:
    """``catalog_name`` runs ``_split_volume`` under the hood — every
    UC routing decision pays this cost."""
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.catalog_name
    return run


def _scenario_parent_url(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            p.url.parent
    return run


def _scenario_from_url_sibling(n: int) -> Callable[[], None]:
    """``_from_url`` is the sibling-path constructor used in walks."""
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    sibling_url = URL.from_(_VOLUME_URL_STR.replace("part-00000", "part-00001"))

    def run() -> None:
        for _ in range(n):
            p._from_url(sibling_url)
    return run


# ---------------------------------------------------------------------------
# Stat-cache scenarios — listing entries seed the cache, follow-ups are local.
# ---------------------------------------------------------------------------


def _scenario_exists_cached(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            p.exists()
    return run


def _scenario_is_file_cached(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            p.is_file()
    return run


def _scenario_size_cached(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            p.size
    return run


def _scenario_stat_cached_fresh(n: int) -> Callable[[], None]:
    """Inner helper — every cached predicate calls this once."""
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            p._stat_cached_fresh()
    return run


def _scenario_size_known(n: int) -> Callable[[], None]:
    """Format readers consult this before risking an empty-file probe."""
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            p.size_known
    return run


# ---------------------------------------------------------------------------
# Pickle scenarios — Spark / multiprocessing fan-out hits these.
# ---------------------------------------------------------------------------


def _scenario_pickle_roundtrip_volume(n: int) -> Callable[[], None]:
    """Pickle dump + load — the cross-process shape."""
    _, _, volumes, _ = _services(with_mock_sdk=False)
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            pickle.loads(pickle.dumps(p))
    return run


def _scenario_pickle_dumps_volume(n: int) -> Callable[[], None]:
    """Pure serialisation cost — measures ``__getstate__`` + payload."""
    _, _, volumes, _ = _services(with_mock_sdk=False)
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)

    def run() -> None:
        for _ in range(n):
            pickle.dumps(p)
    return run


def _scenario_singleton_pickle_hit(n: int) -> Callable[[], None]:
    """In-process pickle: ``__new__`` finds the live singleton."""
    _, _, volumes, _ = _services(with_mock_sdk=False)
    DatabricksPath._INSTANCES.clear()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=None)
    blob = pickle.dumps(p)

    def run() -> None:
        for _ in range(n):
            pickle.loads(blob)
    return run


def _scenario_repr_volume(n: int) -> Callable[[], None]:
    _, _, volumes, _ = _services()
    p = VolumePath(_VOLUME_POSIX, service=volumes, singleton_ttl=False)
    _seed_stat(p)

    def run() -> None:
        for _ in range(n):
            repr(p)
    return run


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SCENARIOS: dict[str, Callable[[int], Callable[[], None]]] = {
    # Construction
    "dbfs_construct_posix": _scenario_dbfs_construct_posix,
    "volume_construct_posix": _scenario_volume_construct_posix,
    "workspace_construct_posix": _scenario_workspace_construct_posix,
    "volume_construct_url": _scenario_volume_construct_url,
    "dispatch_from_posix": _scenario_dispatch_from_posix,
    "dispatch_from_url": _scenario_dispatch_from_url,
    "dispatch_new": _scenario_dispatch_new,
    # Coercion helpers
    "coerce_to_url_str_volume": _scenario_coerce_to_url_str_volume,
    "coerce_to_url_str_passthrough": _scenario_coerce_to_url_str_passthrough,
    "strip_family_prefix_passthrough": _scenario_strip_dbfs_family_prefix_passthrough,
    "strip_family_prefix_legacy": _scenario_strip_dbfs_family_prefix_legacy,
    "resolve_subclass_str": _scenario_resolve_subclass_str,
    "resolve_subclass_url": _scenario_resolve_subclass_url,
    # Singleton
    "singleton_hit_volume": _scenario_singleton_hit_volume,
    "singleton_miss_volume": _scenario_singleton_miss_volume,
    "singleton_key": _scenario_singleton_key,
    # Path rendering
    "full_path_volume": _scenario_full_path_volume,
    "full_path_workspace": _scenario_full_path_workspace,
    "full_path_dbfs": _scenario_full_path_dbfs,
    "api_path_volume": _scenario_api_path_volume,
    "split_volume": _scenario_split_volume,
    "volume_catalog_name": _scenario_volume_catalog_name,
    "parent_url": _scenario_parent_url,
    "from_url_sibling": _scenario_from_url_sibling,
    # Stat cache
    "exists_cached": _scenario_exists_cached,
    "is_file_cached": _scenario_is_file_cached,
    "size_cached": _scenario_size_cached,
    "stat_cached_fresh": _scenario_stat_cached_fresh,
    "size_known": _scenario_size_known,
    # Pickle / repr
    "pickle_roundtrip_volume": _scenario_pickle_roundtrip_volume,
    "pickle_dumps_volume": _scenario_pickle_dumps_volume,
    "singleton_pickle_hit": _scenario_singleton_pickle_hit,
    "repr_volume": _scenario_repr_volume,
}


# ---------------------------------------------------------------------------
# Runner
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
        f"{r['label']:>34s}  "
        f"best={r['best']*1e6:8.2f} us  "
        f"median={r['median']*1e6:8.2f} us  "
        f"mean={r['mean']*1e6:8.2f} us"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5_000, help="Calls per sample.")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--only", default=None,
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
    print(f"# {'label':>34s}  {'best':>12s}  {'median':>14s}  {'mean':>12s}")
    for name in names:
        fn = SCENARIOS[name](args.n)
        fn()  # warm-up
        print(_fmt_row(_time_one(name, fn, args.repeat, args.n)))


if __name__ == "__main__":
    main()
