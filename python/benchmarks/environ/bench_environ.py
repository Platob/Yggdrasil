"""Benchmark the :mod:`yggdrasil.environ` hot paths.

Why this exists
---------------

``yggdrasil.environ`` sits on a few real hot paths:

* ``safe_pip_name`` and ``module_name_to_project_name`` run once per
  package on every install / jobs introspection pass.
* ``cached_from_import`` is hit by ``yggdrasil.data.types.nested`` for
  every Arrow/Polars map/struct round-trip — it's effectively free
  thanks to ``lru_cache``, but the bench keeps that promise honest.
* ``PyEnv.current()`` is a process singleton used everywhere; the
  fast path is one ``is not None`` check and should stay there.
* The ``in_databricks`` / ``in_aws*`` / ``can_access_databricks``
  family runs at module import in ``pickle.ser.callables`` and in
  ``UserInfo`` resolution. They should not pay attribute lookups
  or copy ``os.environ`` views.
* ``UserInfo.current()`` is called on every request / response
  sanitization path — hostname resolution should be a single
  Singleton cache hit.

Usage::

    PYTHONPATH=src python benchmarks/environ/bench_environ.py
    PYTHONPATH=src python benchmarks/environ/bench_environ.py --repeat 7
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Callable

from yggdrasil.environ import PyEnv, UserInfo, cached_from_import, runtime_import_module
from yggdrasil.environ.environment import PIP_MODULE_NAME_MAPPINGS, safe_pip_name
from yggdrasil.environ.modules import (
    module_name_to_project_name,
    packages_distributions_cached,
)
from yggdrasil.environ.userinfo import USERINFO_STRUCT


# ---------------------------------------------------------------------------
# Timing helpers — mirror benchmarks/data/bench_field.py.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 1000)):
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
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale = 1e9
        unit = "ns"
    return (
        f"{r['label']:<60s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


PKG_LIST = ["yaml", "jwt", "pyarrow", "requests", ("dotenv", "1.0.0"), "yggdrasil"]


def scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []

    # ── safe_pip_name ────────────────────────────────────────────────────
    results.append(_time_one(
        "safe_pip_name: 'yaml' (known mapping)",
        lambda: safe_pip_name("yaml"),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "safe_pip_name: 'requests' (passthrough)",
        lambda: safe_pip_name("requests"),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "safe_pip_name: ('yaml', '6.0.2') (tuple)",
        lambda: safe_pip_name(("yaml", "6.0.2")),
        repeat=repeat, inner=200_000,
    ))
    results.append(_time_one(
        "safe_pip_name: list[6] (mixed)",
        lambda: safe_pip_name(PKG_LIST),
        repeat=repeat, inner=50_000,
    ))

    # ── modules.* lookups ────────────────────────────────────────────────
    results.append(_time_one(
        "module_name_to_project_name: 'yggdrasil'",
        lambda: module_name_to_project_name("yggdrasil"),
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "module_name_to_project_name: 'requests' (passthrough)",
        lambda: module_name_to_project_name("requests"),
        repeat=repeat, inner=1_000_000,
    ))
    # Warm once before timing — first call walks site-packages.
    packages_distributions_cached()
    results.append(_time_one(
        "packages_distributions_cached: warm (cached)",
        lambda: packages_distributions_cached(),
        repeat=repeat, inner=1_000_000,
    ))

    # ── cached_from_import ───────────────────────────────────────────────
    # Warm the lru_cache once so we measure the cached lookup, which
    # is the hot-path shape in data/types/nested/{map,struct}.py.
    cached_from_import("yggdrasil.data.data_field", "Field")
    cached_from_import("yggdrasil.data.types.nested", "StructType")
    results.append(_time_one(
        "cached_from_import: ('data_field', 'Field') warm",
        lambda: cached_from_import("yggdrasil.data.data_field", "Field"),
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "cached_from_import: ('nested', 'StructType') warm",
        lambda: cached_from_import("yggdrasil.data.types.nested", "StructType"),
        repeat=repeat, inner=1_000_000,
    ))

    # ── PyEnv singleton / detection ──────────────────────────────────────
    # Warm so we time the cached fast path (one ``is not None`` check).
    PyEnv.current()
    results.append(_time_one(
        "PyEnv.current() warm (singleton hit)",
        lambda: PyEnv.current(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "PyEnv.in_databricks()",
        lambda: PyEnv.in_databricks(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "PyEnv.in_aws_lambda()",
        lambda: PyEnv.in_aws_lambda(),
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "PyEnv.in_aws_batch()",
        lambda: PyEnv.in_aws_batch(),
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "PyEnv.in_aws() (compound)",
        lambda: PyEnv.in_aws(),
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "PyEnv.should_use_databricks_connect()",
        lambda: PyEnv.should_use_databricks_connect(),
        repeat=repeat, inner=500_000,
    ))

    # ── PyEnv properties ─────────────────────────────────────────────────
    env = PyEnv.current()
    results.append(_time_one(
        "env.is_current",
        lambda: env.is_current,
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "env.is_windows",
        lambda: env.is_windows,
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "env.bin_path",
        lambda: env.bin_path,
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "env.root_path",
        lambda: env.root_path,
        repeat=repeat, inner=1_000_000,
    ))

    # ── runtime_import_module fast path ──────────────────────────────────
    # Warm once so the lookup is in the env's ``_checked_modules`` set
    # (use_cache=True path).
    runtime_import_module("json", use_cache=True)
    results.append(_time_one(
        "runtime_import_module('json') warm",
        lambda: runtime_import_module("json", use_cache=True),
        repeat=repeat, inner=20_000,
    ))

    # ── UserInfo ─────────────────────────────────────────────────────────
    # Warm once so we time the Singleton cache hit, not the first
    # hostname lookup.
    UserInfo.current()
    results.append(_time_one(
        "UserInfo.current() warm (singleton hit)",
        lambda: UserInfo.current(),
        repeat=repeat, inner=200_000,
    ))
    info = UserInfo.current()
    # Force lazy slots so the property access we time is the cache hit.
    _ = info.url
    _ = info.git_url
    _ = info.key
    results.append(_time_one(
        "UserInfo.hostname",
        lambda: info.hostname,
        repeat=repeat, inner=1_000_000,
    ))
    results.append(_time_one(
        "UserInfo.url warm (cached)",
        lambda: info.url,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "UserInfo.git_url warm (cached)",
        lambda: info.git_url,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "UserInfo.key warm (cached)",
        lambda: info.key,
        repeat=repeat, inner=500_000,
    ))
    results.append(_time_one(
        "UserInfo.to_struct_dict() warm",
        lambda: info.to_struct_dict(),
        repeat=repeat, inner=50_000,
    ))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# mappings={len(PIP_MODULE_NAME_MAPPINGS)} userinfo_fields={len(USERINFO_STRUCT)}")
    print(f"# os.environ size={len(os.environ)}")
    print(f"# {'label':<60s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
