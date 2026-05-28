"""Concurrency benchmarks for :class:`ExpiringDict` and :class:`Singleton`.

Why this exists
---------------

The single-threaded ``bench_dataclasses.py`` numbers don't surface
the cost that matters most in production: lock contention on the
process-wide singleton caches (``VolumePath._INSTANCES`` /
``UCTable._INSTANCES`` / MSAL token store / Databricks SDK client
registries) when many threads construct, inspect, or invalidate
instances at the same time.

This bench fires N threads at the same cache and measures the
amortized per-op latency under contention, plus an isolated
:class:`Singleton` workload that mirrors a ``VolumePath`` /
``UCTable`` constructor walk without dragging in the Databricks
SDK or URL parser. Useful when tuning locks inside
:class:`ExpiringDict` or :class:`Singleton`.

Usage::

    PYTHONPATH=src python benchmarks/dataclasses/bench_expiring_contention.py
    PYTHONPATH=src python benchmarks/dataclasses/bench_expiring_contention.py --threads 16
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable, ClassVar

from yggdrasil.dataclasses import ExpiringDict
from yggdrasil.dataclasses.singleton import Singleton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bench_threaded(
    label: str,
    fn: Callable[[int], None],
    *,
    threads: int,
    n_per_thread: int,
    repeat: int,
) -> dict:
    """Run ``fn(tid)`` on N threads, ``n_per_thread`` calls each.

    Returns the same shape as ``_time_one`` in ``bench_dataclasses.py``
    so the formatter prints uniformly.
    """
    samples: list[float] = []
    for _ in range(repeat):
        barrier = threading.Barrier(threads)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(n_per_thread):
                fn(tid * n_per_thread + i)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(worker, t) for t in range(threads)]
            wait(futs)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed / (threads * n_per_thread))
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale, unit = (1e9, "ns") if r["best"] < 1e-6 else (1e6, "us")
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}"
    )


# ---------------------------------------------------------------------------
# ExpiringDict contention scenarios — exercise the read-side lock removal.
# ---------------------------------------------------------------------------


def _expiring_contention_scenarios(threads: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    # Pre-populated cache shared across threads.
    d = ExpiringDict[int, int](default_ttl=300.0)
    for i in range(1000):
        d[i] = i

    out.append(_bench_threaded(
        f"ExpiringDict[contended]: get(hit) TTL=300s",
        lambda i: d.get(i % 1000),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: __contains__(hit)",
        lambda i: (i % 1000) in d,
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: ttl(live)",
        lambda i: d.ttl(i % 1000),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: ttl_ns(live)",
        lambda i: d.ttl_ns(i % 1000),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: len(d)",
        lambda i: len(d),
        threads=threads, n_per_thread=100_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: get_or_set(hit)",
        lambda i: d.get_or_set(i % 1000, 0),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))

    # Mixed 90% read / 10% write — the realistic SDK shape.
    d2 = ExpiringDict[int, int](default_ttl=300.0)
    for i in range(1000):
        d2[i] = i

    def mixed(i: int) -> None:
        k = i % 1000
        if i % 10 == 0:
            d2[k] = i
        else:
            d2.get(k)

    out.append(_bench_threaded(
        f"ExpiringDict[contended]: mixed 90R/10W",
        mixed,
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))

    # Snapshot reads — keys() / items() / snapshot() materialize a
    # filtered copy of the dict. Was lock-held, now atomic via
    # ``list(self._store.items())``.
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: keys()",
        lambda i: d.keys(),
        threads=threads, n_per_thread=2_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: items()",
        lambda i: d.items(),
        threads=threads, n_per_thread=2_000, repeat=repeat,
    ))

    # Writer-heavy: __setitem__ overwrites and pop cycle.
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: __setitem__ (overwrite)",
        lambda i: d.__setitem__(i % 1000, i),
        threads=threads, n_per_thread=20_000, repeat=repeat,
    ))
    out.append(_bench_threaded(
        f"ExpiringDict[contended]: pop+set cycle",
        lambda i: (d.pop(i % 1000, None), d.__setitem__(i % 1000, i)),
        threads=threads, n_per_thread=10_000, repeat=repeat,
    ))

    return out


# ---------------------------------------------------------------------------
# Singleton scenarios — mirror VolumePath / UCTable construction patterns.
# ---------------------------------------------------------------------------


# A minimal Singleton subclass that mirrors the VolumePath /
# UCTable hot path WITHOUT pulling in the Databricks SDK. The key
# is a small tuple, the constructor is idempotent, and the cache
# uses a 300s TTL — same shape as the real classes.

class _PathSingleton(Singleton):
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=300.0)
    _INSTANCES_LOCK: ClassVar[threading.RLock] = threading.RLock()
    _SINGLETON_TTL: ClassVar[Any] = 300.0
    __slots__ = ("_initialized", "_singleton_key_")

    @classmethod
    def _singleton_key(cls, *args: Any, **kwargs: Any) -> Any:
        return (cls, args, tuple(sorted(kwargs.items())))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_initialized", False):
            return
        object.__setattr__(self, "_initialized", True)


KEYS = [(f"k{i}",) for i in range(500)]


def _singleton_scenarios(threads: int, repeat: int) -> list[dict]:
    out: list[dict] = []

    # Warm the cache.
    for k in KEYS:
        _PathSingleton(*k)

    # Steady-state hit path — every constructor call lands in the
    # lock-free ``_INSTANCES.get`` read.
    out.append(_bench_threaded(
        f"Singleton[contended]: __new__ hit (TTL=300s)",
        lambda i: _PathSingleton(*KEYS[i % len(KEYS)]),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))

    # singleton_ttl=None → cached forever, exercises the no-TTL branch
    # in ExpiringDict.get.
    class _Forever(Singleton):
        _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=None)
        _INSTANCES_LOCK: ClassVar[threading.RLock] = threading.RLock()
        _SINGLETON_TTL: ClassVar[Any] = None
        __slots__ = ("_initialized", "_singleton_key_")

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if getattr(self, "_initialized", False):
                return
            object.__setattr__(self, "_initialized", True)

    for k in KEYS:
        _Forever(*k)

    out.append(_bench_threaded(
        f"Singleton[contended]: __new__ hit (TTL=None)",
        lambda i: _Forever(*KEYS[i % len(KEYS)]),
        threads=threads, n_per_thread=50_000, repeat=repeat,
    ))

    # Invalidate + reconstruct — 1-in-8 ops force a writer round-trip.
    # Exercises ``_INSTANCES.pop`` (now lock-free) and ``_INSTANCES.set``.
    def invalidate_cycle(i: int) -> None:
        inst = _PathSingleton(*KEYS[i % len(KEYS)])
        if i % 8 == 0:
            inst.invalidate_singleton()

    out.append(_bench_threaded(
        f"Singleton[contended]: invalidate + reconstruct (1/8)",
        invalidate_cycle,
        threads=threads, n_per_thread=20_000, repeat=repeat,
    ))

    # to_singleton churn — the iterdir/_ls/glob pattern. Build many
    # short-lived children with ``singleton_ttl=False`` and promote
    # a fraction. Exercises both the bypass path and the writer.
    class _Churn(Singleton):
        _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(default_ttl=300.0)
        _INSTANCES_LOCK: ClassVar[threading.RLock] = threading.RLock()
        _SINGLETON_TTL: ClassVar[Any] = 300.0
        __slots__ = ("_initialized", "_singleton_key_")

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if getattr(self, "_initialized", False):
                return
            object.__setattr__(self, "_initialized", True)

    def churn(i: int) -> None:
        inst = _Churn(*KEYS[i % len(KEYS)], singleton_ttl=False)
        if i % 4 == 0:
            inst.to_singleton(ttl=300.0)

    out.append(_bench_threaded(
        f"Singleton[contended]: iterdir-style churn (ttl=False + promote)",
        churn,
        threads=threads, n_per_thread=20_000, repeat=repeat,
    ))

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def scenarios(threads: int, repeat: int) -> list[dict]:
    return [
        *_expiring_contention_scenarios(threads, repeat),
        *_singleton_scenarios(threads, repeat),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8,
                    help="Worker threads per scenario.")
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# threads={args.threads}  repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.threads, args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
