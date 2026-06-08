"""Benchmark :mod:`yggdrasil.loki.resources` — the workstation/accelerator probe.

Why this exists
---------------

Loki sizes a local model to the box and routes it onto the best accelerator
(NVIDIA ``cuda``, **Intel GPU** ``xpu``, Apple ``mps``, else CPU). Two probes
sit on the interactive hot path:

* :func:`resources.accelerator` — imports ``torch`` once (cached in
  ``sys.modules`` after) and queries each backend. ``Loki.select`` calls
  ``can_run_local`` on **every turn**, so this must stay cheap once warm.
* :func:`resources.snapshot` — bundles cpu/ram/accelerator/npu; the
  ``has_npu`` half optionally touches OpenVINO, so the snapshot's cost depends
  on whether that's installed.

The scenarios below isolate the warm steady-state cost (what a running REPL
pays per turn) from the device resolution the transformers engine does on a
pipeline build. Heavy real backends are stubbed so the benchmark measures the
probe's own dispatch overhead, not GPU driver init.

Usage::

    PYTHONPATH=src python benchmarks/loki/bench_resources.py
    PYTHONPATH=src python benchmarks/loki/bench_resources.py --repeat 7
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import types
from typing import Callable
from unittest.mock import patch

from yggdrasil.loki import resources
from yggdrasil.loki.engines import TransformersEngine


def _time_one(label: str, fn: Callable[[], object], *, repeat: int, inner: int) -> dict:
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


def _fake_torch(*, cuda=False, xpu=False, mps=False) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    if xpu:
        t.xpu = types.SimpleNamespace(is_available=lambda: True)
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    return t


def scenarios(repeat: int) -> list[dict]:
    results: list[dict] = []

    # Warm accelerator probe: torch present (the realistic steady state — the
    # module stays in sys.modules, so each call is just the backend queries).
    with patch.dict(sys.modules, {"torch": _fake_torch(xpu=True)}):
        results.append(_time_one("accelerator() · Intel xpu present",
                                 resources.accelerator, repeat=repeat, inner=20_000))
    with patch.dict(sys.modules, {"torch": _fake_torch(cuda=True)}):
        results.append(_time_one("accelerator() · cuda present",
                                 resources.accelerator, repeat=repeat, inner=20_000))
    with patch.dict(sys.modules, {"torch": _fake_torch()}):
        results.append(_time_one("accelerator() · cpu-only (all backends checked)",
                                 resources.accelerator, repeat=repeat, inner=20_000))

    # Full snapshot — what size_tier / can_run_local resolve through. has_npu is
    # stubbed to False (OpenVINO absent) so we measure the dict assembly + probe
    # dispatch, not an OpenVINO Core() init.
    with patch.dict(sys.modules, {"torch": _fake_torch(xpu=True)}), \
            patch.object(resources, "has_npu", return_value=False):
        results.append(_time_one("snapshot() · accelerator + cpu/ram",
                                 resources.snapshot, repeat=repeat, inner=20_000))
        results.append(_time_one("can_run_local() · per-turn gate",
                                 resources.can_run_local, repeat=repeat, inner=20_000))

    # Device resolution on a pipeline build: pin short-circuits; auto falls to
    # the accelerator probe.
    pinned = TransformersEngine(device="xpu")
    results.append(_time_one("resolve_device() · explicit pin (no probe)",
                             pinned.resolve_device, repeat=repeat, inner=50_000))
    auto = TransformersEngine()
    with patch.dict(sys.modules, {"torch": _fake_torch(xpu=True)}):
        results.append(_time_one("resolve_device() · auto-detect xpu",
                                 auto.resolve_device, repeat=repeat, inner=20_000))

    return results


def _fmt(r: dict) -> str:
    scale, unit = (1e9, "ns") if r["best"] < 1e-6 else (1e6, "us")
    return (f"# {r['label']:<52s}  {r['best'] * scale:>10.2f} {unit}"
            f"  {r['median'] * scale:>10.2f} {unit}"
            f"  {r['mean'] * scale:>10.2f} {unit}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5,
                    help="Outer repeat count per scenario (median across).")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<52s}  {'best':>13s}  {'median':>13s}  {'mean':>13s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
