"""Benchmark the ygg CLI style rendering hot paths (:mod:`yggdrasil.cli.style`).

Every ``ygg`` CLI paints its output through ``style`` — the semantic colour
helpers (``dim``/``brand``/``good``/…) are called several times per rendered
line (171 call sites in the Loki CLI alone), and the Loki REPL streams a reply
through ``style.out`` token by token. This bench measures the per-call cost of
those hot paths so a regression (or an optimization) is visible.

Scenarios
---------
- colour helpers: a single ``dim``/``brand`` call (ANSI path, colour forced on)
- ``status_line``: a realistic line built from ~5 helper calls + an f-string
- ``usage_row``: a usage-table row (helpers + ljust/rjust)
- ``logo``: rendering the gradient wordmark
- ``out`` throughput: N writes one-per-call (each flushes) vs one batched write,
  to a real fd — shows the syscall cost of per-token flushing

Usage::

    PYTHONPATH=src python benchmarks/cli/bench_style.py
    PYTHONPATH=src python benchmarks/cli/bench_style.py --repeat 7
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any, Callable

from yggdrasil.cli import style

# Measure the real ANSI rendering path, not the no-colour short-circuit a
# non-TTY bench would otherwise take.
style.force_color(True)

INNER = 2000  # inner iterations per outer sample


def _time(fn: Callable[[], Any], *, repeat: int, inner: int = INNER) -> list[float]:
    for _ in range(min(inner, 100)):  # warm-up
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return samples


_HDR = f"{'scenario':<34}  {'best':>10}  {'median':>10}  unit"
_SEP = "-" * (len(_HDR) + 4)


def _fmt(label: str, samples: list[float], *, unit: str = "µs", scale: float = 1e6) -> str:
    best = min(samples) * scale
    med = statistics.median(samples) * scale
    return f"{label:<34}  {best:>10.4f}  {med:>10.4f}  {unit}"


def _bench_helpers(repeat: int) -> None:
    print(_fmt("dim(short)", _time(lambda: style.dim("hello"), repeat=repeat)))
    print(_fmt("brand(short)", _time(lambda: style.brand("hello"), repeat=repeat)))

    def status_line() -> str:
        return (f"  {style.cyan('agent')}   {style.bold('loki')}  "
                f"{style.dim('#8338906790559775204')}  {style.good('●')} "
                f"{style.brand('databricks')}{style.dim(' (default)')}")

    print(_fmt("status_line (6 helpers)", _time(status_line, repeat=repeat)))

    def usage_row() -> str:
        name = "claude · claude-opus-4-8"
        return (f"  {style.brand(name.ljust(32))}{str(12).rjust(6)}"
                f"{'1,234'.rjust(9)}{'5,678'.rjust(9)}"
                f"{'6,912'.rjust(10)}{style.good('$0.0042'.rjust(11))}")

    print(_fmt("usage_row (2 helpers)", _time(usage_row, repeat=repeat)))


def _bench_logo(repeat: int) -> None:
    print(_fmt("logo(YGGLOKI)", _time(lambda: style.logo("YGGLOKI"), repeat=repeat, inner=400)))


def _bench_out(repeat: int) -> None:
    # Write to a real fd (/dev/null) so flush() costs a genuine write syscall,
    # the way it does to a terminal — StringIO would hide it.
    chunks = [f"tok{i} " for i in range(80)]          # ~a short streamed reply
    line = "".join(chunks)
    devnull = open(os.devnull, "w")
    saved = sys.stdout
    sys.stdout = devnull
    try:
        per_call = _time(lambda: [style.out(c) for c in chunks],
                         repeat=repeat, inner=200)
        batched = _time(lambda: style.out(line), repeat=repeat, inner=200)
    finally:
        sys.stdout = saved
        devnull.close()
    print(_fmt(f"out x{len(chunks)} (per-call flush)", per_call))
    print(_fmt("out x1 (batched)", batched))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark yggdrasil.cli.style rendering.")
    ap.add_argument("--repeat", type=int, default=5, help="outer samples per scenario")
    args = ap.parse_args(argv)

    print(_HDR)
    print(_SEP)
    _bench_helpers(args.repeat)
    _bench_logo(args.repeat)
    _bench_out(args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
