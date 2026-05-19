# Skill: write and run benchmarks during development

## When to use

The user says "this should be faster", "is this slow?", "I think we
can win on X", "let's measure before merging", "add a bench for this
path", or pastes a function and asks for a perf comparison. Also as
a *gate* on any change that touches a hot path (cast registry, IO
buffers, HTTP send, table insert, schema reconciliation) — see
[`AGENTS.md` → "Benchmark-driven optimization"](https://github.com/Platob/Yggdrasil/blob/main/AGENTS.md).
**Performance changes that don't quote a before / after don't ship.**

## Where benches live

```
python/benchmarks/
├── arrow/         bench_cast.py, bench_struct.py, …
├── concurrent/    bench_parallelize.py
├── data/          bench_cast.py, bench_field.py, bench_registry.py, …
├── databricks/    bench_volume_io.py, bench_insert_staging.py
├── dataclasses/   bench_expiring_dict.py
├── environ/       bench_userinfo.py
├── io/            bench_http_send.py, primitive/, path/, …
├── polars/        bench_cast.py
└── run_all.py
```

Layout mirrors `python/src/yggdrasil/`. **Add a bench in the
matching folder for any hot path you touch** — when the matching
folder doesn't exist, create it.

## The bench shape

Every bench is a runnable script. The minimal template:

```python
"""Benchmark <one-line description>.

Why this exists
---------------
<which call shape lives here, and what regression the bench catches>.

Usage::

    PYTHONPATH=src python benchmarks/<area>/bench_<name>.py
    PYTHONPATH=src python benchmarks/<area>/bench_<name>.py --rows 100000 --repeat 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable


def make_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--repeat", type=int, default=5)
    return p.parse_args()


def time_call(fn: Callable[[], object], *, repeat: int) -> dict:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return {
        "best_us": min(samples) / 1_000,
        "median_us": statistics.median(samples) / 1_000,
        "worst_us": max(samples) / 1_000,
    }


def main() -> None:
    args = make_args()

    # --- fixtures ---
    sample = ...

    # --- scenarios ---
    scenarios: dict[str, Callable[[], object]] = {
        "baseline":     lambda: do_old(sample),
        "candidate":    lambda: do_new(sample),
    }

    print(f"rows={args.rows}  repeat={args.repeat}")
    for name, fn in scenarios.items():
        result = time_call(fn, repeat=args.repeat)
        print(f"  {name:20s} best={result['best_us']:>10.2f} us  "
              f"median={result['median_us']:>10.2f} us")


if __name__ == "__main__":
    main()
```

`bench_cast.py`, `bench_field.py`, `bench_registry.py` in
`benchmarks/data/` are good full-size references.

## The before / after workflow

The canonical loop (from `AGENTS.md`):

1. **Find or add the bench first.** If the hot path you're touching
   doesn't have one, *adding the bench counts as part of the change*.
2. **Capture the baseline** on the pre-change tree:

   ```bash
   git stash               # or git checkout main
   PYTHONPATH=src python benchmarks/data/bench_cast.py --repeat 5
   git stash pop
   ```

3. **Apply one conceptual change at a time.**
4. **Re-run the same bench.** Quote `best` + `median` before / after.
5. **Run the full suite** to confirm nothing else regressed:

   ```bash
   PYTHONPATH=src python benchmarks/run_all.py --repeat 3
   ```

6. **Put the numbers in the commit body** so the next reader knows
   which numbers are load-bearing.

## Example commit body shape

```
perf(data/cast): short-circuit identity casts in convert()

before (bench_cast.py --rows 100_000 --repeat 5):
  identity_int64       best= 142.30 us  median= 145.10 us
  identity_string      best= 138.70 us  median= 141.90 us

after:
  identity_int64       best=   8.20 us  median=   8.40 us  (-94 %)
  identity_string      best=   8.10 us  median=   8.30 us  (-94 %)

The MRO walk landed on the same converter every time; bypass it
when from_type is to_type. Other scenarios unchanged within noise.
```

## When the workload needs a live system

Benchmarks that touch Databricks / Postgres / Mongo live in their
respective subfolder (`databricks/bench_volume_io.py`, …) and are
**skipped by default** in `run_all.py`. Opt in explicitly:

```bash
PYTHONPATH=src python benchmarks/run_all.py \
    --include bench_databricks_insert_staging
```

Pattern:

```python
import os
import unittest

if not os.environ.get("DATABRICKS_HOST"):
    raise SystemExit("DATABRICKS_HOST not set — skipping live bench.")
```

Inside the bench, fresh resources (Volume, Table, Schema) must be
created and cleaned up per run — the bench should be reentrant and
not leave workspace state behind.

## Pick the right metric

- **CPU-bound vectorised work** → `best_us` is the signal (system
  noise dominates median otherwise). Use `time.perf_counter_ns()`.
- **IO / network work** → `median_us` over a larger `--repeat` (15+);
  drop outliers tighter than ±2 σ before quoting.
- **Memory regressions** → `tracemalloc` + `gc.collect()` between
  iterations. Add a `--memory` flag and report peak bytes.
- **Throughput** → quote rows/sec (`rows / (best_us / 1e6)`) so a
  reader can sanity-check against the engine docs.

## Don'ts

- Don't quote a single `time.time()` measurement — `time.perf_counter`
  or `perf_counter_ns`, multiple samples, report `best` + `median`.
- Don't bench through `pytest` — the test runner adds ~tens of ms of
  fixture overhead per call. Plain scripts under `benchmarks/`.
- Don't bench in a debug Python — `-O` off is fine, but check
  `sys.gettrace() is None` so no coverage / debugger is attached.
- Don't include a bench that requires live credentials in
  `run_all.py`'s default set — gate via `--include`.
- Don't claim a regression / win without running `run_all.py` to
  confirm the rest of the suite didn't move. Perf changes commonly
  trade off; the surrounding numbers are the proof you didn't.
- Don't measure once, fix the "feels faster" loop, and ship.
  *Numbers ship, feelings don't.*
