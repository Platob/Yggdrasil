"""Benchmark the :mod:`yggdrasil.dataclasses` helpers.

Why this exists
---------------

Two primitives in this module are reached for everywhere:

* :class:`ExpiringDict` — the cache primitive behind Databricks SDK
  caches (catalogs / schemas / tables / warehouses), MSAL auth
  singletons, and any other expiring registry.
* :class:`WaitingConfig` — the retry / backoff config dataclass
  threaded through every long-running operation.

Plus the dataclass↔Arrow bridge (``dataclass_to_arrow_field`` and
the ``serialize_dataclass_state`` / ``restore_dataclass_state``
helpers used by pickle round-trip support across the codebase).

The helpers run on the hot path: ``ExpiringDict.get`` / ``set`` /
``get_or_set`` run per cache hit; ``WaitingConfig`` constructs from
the public ``from_(...)`` factory per retry attempt. Per-call
overhead matters.

Usage::

    PYTHONPATH=src python benchmarks/bench_dataclasses.py
    PYTHONPATH=src python benchmarks/bench_dataclasses.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle
import statistics
import time
from dataclasses import dataclass
from typing import Callable

from yggdrasil.dataclasses import (
    ExpiringDict,
    WaitingConfig,
)
from yggdrasil.dataclasses.dataclass import (
    dataclass_to_arrow_field,
    serialize_dataclass_state,
    restore_dataclass_state,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ArrowShape:
    id: int
    name: str
    amount: float
    paid: bool = False
    placed_on: dt.date | None = None


SHAPE = _ArrowShape(id=1, name="x", amount=1.5, paid=True, placed_on=dt.date(2024, 1, 1))


# ---------------------------------------------------------------------------
# Timing helpers.
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
        f"{r['label']:<58s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------


def _dataclass_helpers_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # Pickle / state round-trip — drives ``serialize_dataclass_state`` /
    # ``restore_dataclass_state``, the picklable-by-config pattern used
    # by every Session / Client / Path class.
    payload = pickle.dumps(SHAPE)
    out.append(_time_one(
        "dataclass: pickle.dumps(SHAPE)",
        lambda: pickle.dumps(SHAPE),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "dataclass: pickle.loads(SHAPE)",
        lambda: pickle.loads(payload),
        repeat=repeat, inner=20_000,
    ))
    out.append(_time_one(
        "dataclass: serialize_dataclass_state(SHAPE)",
        lambda: serialize_dataclass_state(SHAPE),
        repeat=repeat, inner=50_000,
    ))
    state = serialize_dataclass_state(SHAPE)
    out.append(_time_one(
        "dataclass: restore_dataclass_state(_, state)",
        lambda: restore_dataclass_state(
            _ArrowShape(id=0, name="", amount=0.0), state,
        ),
        repeat=repeat, inner=20_000,
    ))
    # Arrow-field cache hit — should be ~100ns dict lookup.
    out.append(_time_one(
        "dataclass_to_arrow_field: cached",
        lambda: dataclass_to_arrow_field(_ArrowShape),
        repeat=repeat, inner=500_000,
    ))

    return out


def _expiring_dict_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    # No-TTL cache (singleton-style) — the Databricks SDK / MSAL pattern.
    d_inf = ExpiringDict[str, int](default_ttl=None)
    d_inf["a"] = 1
    d_inf["b"] = 2

    out.append(_time_one(
        "ExpiringDict: get(hit) no-TTL",
        lambda: d_inf.get("a"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "ExpiringDict: get(miss) no-TTL",
        lambda: d_inf.get("missing"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "ExpiringDict: __getitem__(hit)",
        lambda: d_inf["a"],
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "ExpiringDict: __contains__(hit)",
        lambda: "a" in d_inf,
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "ExpiringDict: set('k', 1)",
        lambda: d_inf.set("k", 1),
        repeat=repeat, inner=100_000,
    ))

    # Bounded + TTL — the typical "warehouse cache, 5 minutes, 64 entries"
    # shape.
    d_ttl = ExpiringDict[str, int](default_ttl=300.0, max_size=64)
    for i in range(32):
        d_ttl[f"k{i}"] = i

    out.append(_time_one(
        "ExpiringDict: get(hit) 5min-TTL bounded",
        lambda: d_ttl.get("k0"),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "ExpiringDict: set hit-overwrite 5min-TTL",
        lambda: d_ttl.set("k0", 0),
        repeat=repeat, inner=100_000,
    ))

    # get_or_set — common idempotent build-once-and-cache pattern.
    d_cached = ExpiringDict[str, int](default_ttl=None)

    def loader() -> int:
        return 42
    out.append(_time_one(
        "ExpiringDict: get_or_set(hit)",
        lambda: d_cached.get_or_set("k", loader),
        repeat=repeat, inner=100_000,
    ))

    return out


def _waiting_config_scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    out.append(_time_one(
        "WaitingConfig: default()",
        lambda: WaitingConfig.default(),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "WaitingConfig: WaitingConfig() construct",
        lambda: WaitingConfig(),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "WaitingConfig: from_(None)",
        lambda: WaitingConfig.from_(None),
        repeat=repeat, inner=200_000,
    ))
    out.append(_time_one(
        "WaitingConfig: from_(30.0) seconds",
        lambda: WaitingConfig.from_(30.0),
        repeat=repeat, inner=100_000,
    ))
    out.append(_time_one(
        "WaitingConfig: from_({'timeout': 30})",
        lambda: WaitingConfig.from_({"timeout": 30}),
        repeat=repeat, inner=100_000,
    ))
    cfg = WaitingConfig()
    out.append(_time_one(
        "WaitingConfig: hash(cfg)",
        lambda: hash(cfg),
        repeat=repeat, inner=500_000,
    ))
    out.append(_time_one(
        "WaitingConfig: cfg == cfg",
        lambda: cfg == cfg,
        repeat=repeat, inner=500_000,
    ))

    return out


def scenarios(repeat: int) -> list[dict]:
    return [
        *_dataclass_helpers_scenarios(repeat),
        *_expiring_dict_scenarios(repeat),
        *_waiting_config_scenarios(repeat),
    ]


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<58s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
