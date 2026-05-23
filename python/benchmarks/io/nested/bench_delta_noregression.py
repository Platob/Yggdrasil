"""Local-dev no-regression bench — yggdrasil :class:`DeltaFolder` vs ``deltalake``.

**This is not run in CI** and is not part of the canonical
``benchmarks/io/nested/bench_delta.py`` flow. It exists so a
developer poking at the Delta read/write paths on a laptop can
sanity-check that a change didn't blow yggdrasil's numbers out vs
the rust-backed reference implementation in the
:pypi:`deltalake` package.

Why a separate file
-------------------

- ``deltalake`` is an optional dependency (extras ``[delta]``); the
  primary bench at ``bench_delta.py`` deliberately doesn't import
  it. We want the canonical bench to surface yggdrasil's *own*
  streaming-memory behaviour, not a head-to-head with the rust
  crate.
- This script is opt-in: pass ``--allow-deltalake`` (or set
  ``YGG_DELTA_NOREGRESSION=1``) so the import doesn't fire by
  accident on a stripped-down dev install.
- Comparisons here are intentionally **collect-all** on both sides
  — that's what ``deltalake`` ergonomically does — so the wall
  clock is comparable. The streaming-vs-collect property is the
  main bench's job.

Usage::

    # Install the extra once:
    uv pip install deltalake>=1.0

    # Run the comparison:
    PYTHONPATH=src python benchmarks/io/nested/bench_delta_noregression.py \\
        --rows 50000 --partitions 16 --repeat 5 --allow-deltalake

The script exits with status 2 (and prints a hint) when the gate
isn't enabled, so a stray invocation doesn't silently produce
misleading numbers from one engine when the other isn't installed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema
from yggdrasil.data.types.primitive import Int64Type, StringType
from yggdrasil.io.nested.delta import DeltaFolder, DeltaOptions
from yggdrasil.execution.expr import col as expr_col


PARTITION_KEYS = [f"p{i:02d}" for i in range(64)]


def _partition_schema() -> Schema:
    s = Schema()
    s.with_field(Field(name="id", dtype=Int64Type()))
    s.with_field(
        Field(name="region", dtype=StringType()).with_partition_by(True)
    )
    s.with_field(Field(name="val", dtype=StringType()))
    return s


def _arrow_table(rows: int, *, partitions: int) -> pa.Table:
    keys = PARTITION_KEYS[:partitions]
    return pa.table({
        "id": pa.array(range(rows), type=pa.int64()),
        "region": pa.array([keys[i % partitions] for i in range(rows)]),
        "val": pa.array([f"row-{i}" for i in range(rows)]),
    })


def _time_one(label: str, fn: Callable[[], object], *, repeat: int, inner: int = 1) -> dict:
    fn()  # warmup
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


def _fmt_time(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:7.1f} us"
    if secs < 1:
        return f"{secs * 1e3:7.1f} ms"
    return f"{secs:7.3f}  s"


def _fmt(r: dict) -> str:
    return (
        f"{r['label']:<60s}  "
        f"best={_fmt_time(r['best'])}  "
        f"median={_fmt_time(r['median'])}  "
        f"mean={_fmt_time(r['mean'])}"
    )


# ---------------------------------------------------------------------------
# Gate — the script refuses to import deltalake unless explicitly
# allowed. Keeps a stray ``python bench_delta_noregression.py`` from
# silently producing yggdrasil-only or deltalake-only numbers.
# ---------------------------------------------------------------------------


def _check_gate(args) -> None:
    if not args.allow_deltalake and not os.environ.get("YGG_DELTA_NOREGRESSION"):
        print(
            "bench_delta_noregression: gated bench — not run by default. "
            "Pass --allow-deltalake or set YGG_DELTA_NOREGRESSION=1 to opt in.",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Scenarios — head-to-head reads and writes against a partitioned table.
# ---------------------------------------------------------------------------


def _write_scenarios(
    table: pa.Table, rows: int, partitions: int, repeat: int,
) -> list[dict]:
    from deltalake import write_deltalake

    out: list[dict] = []

    def _yggdrasil_write():
        tmp = tempfile.mkdtemp(prefix="ygg-nr-w-")
        try:
            DeltaFolder(path=tmp + "/t").write_arrow_table(
                table, options=DeltaOptions(target=_partition_schema()),
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _deltalake_write():
        tmp = tempfile.mkdtemp(prefix="ygg-nr-w-")
        try:
            write_deltalake(tmp + "/t", table, partition_by=["region"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    out.append(_time_one(
        f"write: yggdrasil rows={rows} parts={partitions}",
        _yggdrasil_write, repeat=repeat,
    ))
    out.append(_time_one(
        f"write: deltalake rows={rows} parts={partitions}",
        _deltalake_write, repeat=repeat,
    ))
    return out


def _read_scenarios(
    table_path: Path, rows: int, partitions: int, repeat: int,
) -> list[dict]:
    from deltalake import DeltaTable

    out: list[dict] = []
    target_key = PARTITION_KEYS[0]

    ygg = DeltaFolder(path=str(table_path))
    ygg.snapshot(fresh=True)  # warm
    dt = DeltaTable(str(table_path))

    # Full scan, collect-all on both sides.
    out.append(_time_one(
        f"read: yggdrasil table full-scan rows={rows}",
        lambda: ygg.read_arrow_table(),
        repeat=repeat, inner=2,
    ))
    out.append(_time_one(
        f"read: deltalake table full-scan rows={rows}",
        lambda: dt.to_pyarrow_table(),
        repeat=repeat, inner=2,
    ))

    # Partition prune to one value.
    out.append(_time_one(
        f"read: yggdrasil predicate region == {target_key}",
        lambda: ygg.read_arrow_table(
            options=DeltaOptions(predicate=(expr_col("region") == target_key)),
        ),
        repeat=repeat, inner=10,
    ))
    out.append(_time_one(
        f"read: deltalake partitions=region={target_key}",
        lambda: dt.to_pyarrow_table(
            partitions=[("region", "=", target_key)],
        ),
        repeat=repeat, inner=10,
    ))

    # Partition + row filter.
    out.append(_time_one(
        f"read: yggdrasil predicate region == X AND id > N/2",
        lambda: ygg.read_arrow_table(
            options=DeltaOptions(
                predicate=(expr_col("region") == target_key) & (expr_col("id") > rows // 2),
            ),
        ),
        repeat=repeat, inner=10,
    ))
    out.append(_time_one(
        f"read: deltalake partitions+filters region+id",
        lambda: dt.to_pyarrow_table(
            partitions=[("region", "=", target_key)],
            filters=[("id", ">", rows // 2)],
        ),
        repeat=repeat, inner=10,
    ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--partitions", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--allow-deltalake",
        action="store_true",
        help="Opt in to importing the deltalake dependency. "
             "Defaults to the YGG_DELTA_NOREGRESSION env var.",
    )
    args = ap.parse_args()

    _check_gate(args)

    # Import after the gate so a missing deltalake doesn't cost
    # anything on the default skip path.
    try:
        import deltalake  # noqa: F401
    except ImportError:
        print(
            "bench_delta_noregression: deltalake not installed; "
            "uv pip install 'ygg[delta]' (or 'deltalake>=1.0') and retry.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.partitions > len(PARTITION_KEYS):
        ap.error(f"--partitions must be ≤ {len(PARTITION_KEYS)}")

    table = _arrow_table(args.rows, partitions=args.partitions)
    tmp_root = Path(tempfile.mkdtemp(prefix="ygg-nr-bench-"))
    try:
        read_path = tmp_root / "read_t"
        DeltaFolder(path=str(read_path)).write_arrow_table(
            table, options=DeltaOptions(target=_partition_schema()),
        )

        print(
            f"# rows={args.rows}  partitions={args.partitions}  "
            f"repeat={args.repeat}"
        )
        print(f"# {'label':<60s}  {'best':>13s}  {'median':>18s}  {'mean':>13s}")
        for row in _write_scenarios(
            table, args.rows, args.partitions, args.repeat,
        ):
            print(_fmt(row))
        for row in _read_scenarios(
            read_path, args.rows, args.partitions, args.repeat,
        ):
            print(_fmt(row))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
