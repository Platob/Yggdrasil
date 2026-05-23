"""Run every ``bench_*`` benchmark in this directory.

Usage::

    PYTHONPATH=src python benchmarks/run_all.py
    PYTHONPATH=src python benchmarks/run_all.py --repeat 7
    PYTHONPATH=src python benchmarks/run_all.py --skip bench_databricks_insert_staging

Per-benchmark output is wrapped in a banner so the combined log
remains scannable. Benchmarks that require live external systems
(Databricks, Postgres, …) are skipped by default — pass
``--include bench_databricks_insert_staging`` to opt them in.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent

# HTTP benchmarks live alongside their tests for better grouping.
_TEST_BENCH_DIR = BENCH_DIR.parent / "tests" / "test_yggdrasil" / "test_http_" / "benchmarks"

# Benchmarks that require live external systems — skipped by default.
# Opt in with ``--include <name>``.
_REQUIRES_LIVE: frozenset[str] = frozenset({
    "bench_databricks_insert_staging",
    "bench_io_remote",
})


def _discover() -> list[Path]:
    """Find every ``bench_*.py`` under ``BENCH_DIR`` and the test HTTP
    benchmarks directory (recursive).

    Benches are organized into module-mirrored subfolders
    (``benchmarks/data/``, ``benchmarks/io/primitive/``, …) and the
    HTTP benchmarks live in ``tests/test_yggdrasil/test_http_/benchmarks/``
    — we walk both trees so all benches get picked up.
    """
    found = [
        p for p in BENCH_DIR.rglob("bench_*.py")
        if p.name != Path(__file__).name
    ]
    if _TEST_BENCH_DIR.is_dir():
        found.extend(_TEST_BENCH_DIR.rglob("bench_*.py"))
    return sorted(found)


def _run_one(path: Path, *, repeat: int) -> int:
    """Run a single benchmark via the current Python interpreter.

    Returns the child process exit code so the caller can decide
    whether to fail the whole sweep.
    """
    name = path.stem
    print()
    print("=" * 78)
    print(f"=== {name}  (repeat={repeat})")
    print("=" * 78)
    sys.stdout.flush()

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path), "--repeat", str(repeat)],
        cwd=BENCH_DIR.parent,
    )
    elapsed = time.perf_counter() - t0
    print(f"--- {name} finished in {elapsed:.2f}s (rc={proc.returncode})")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Outer repeat count handed to each benchmark.",
    )
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Benchmark name (without .py) to skip. May be passed multiple times.",
    )
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Include a benchmark normally skipped by default "
            "(currently: live-system benchmarks like "
            "``bench_databricks_insert_staging``)."
        ),
    )
    args = ap.parse_args()

    skip = set(args.skip)
    include = set(args.include)

    benches = _discover()
    if not benches:
        print(f"# no benchmarks found in {BENCH_DIR}")
        return 0

    selected: list[Path] = []
    for b in benches:
        name = b.stem
        if name in skip:
            continue
        if name in _REQUIRES_LIVE and name not in include:
            print(
                f"# skipping {name} — requires a live external system; "
                "opt in with --include"
            )
            continue
        selected.append(b)

    if not selected:
        print("# no benchmarks selected after filtering")
        return 0

    print(f"# running {len(selected)} benchmarks, repeat={args.repeat}")

    failures: list[str] = []
    overall_t0 = time.perf_counter()
    for b in selected:
        rc = _run_one(b, repeat=args.repeat)
        if rc != 0:
            failures.append(b.stem)

    print()
    print("=" * 78)
    print(f"=== summary: {len(selected) - len(failures)}/{len(selected)} passed"
          f" in {time.perf_counter() - overall_t0:.2f}s")
    if failures:
        print(f"=== failures: {', '.join(failures)}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
