"""Benchmark :class:`AsyncInsert` (de)serialization, merge, and SQL rendering.

The async-insert path has two distinct hot spots:

1. **Per-stage** — :func:`stage_async_insert` builds one record per
   ``Table.insert(async_write=True)`` call and writes it as JSON next
   to a Parquet file. The Parquet write is the dominant cost (see
   :mod:`bench_databricks_insert_staging`); this bench focuses on the
   record-building + JSON-serialization side, which fires once per
   staged op.
2. **Per-apply** — the schema-level applier (``AsyncInsert.merge``,
   ``AsyncInsert.merge_with``, ``AsyncInsert.to_sql``) reads N JSON
   records, folds them per target, and renders SQL. With a busy
   schema this can iterate hundreds of records per apply window.

Usage::

    python benchmarks/databricks/bench_async_insert.py
    python benchmarks/databricks/bench_async_insert.py --records 500 --repeat 5

Numbers are wall time per call (best / median of N repeats, lower is
better). The harness does no I/O — every operation runs against
in-memory dataclass instances and bytes buffers — so results are
stable enough to A/B 1-2% changes.
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

from yggdrasil.databricks.table.async_write import AsyncInsert
from yggdrasil.pickle import json as ygg_json


# ---------------------------------------------------------------------------
# Record factories — shapes that exercise the field set the applier sees.
# ---------------------------------------------------------------------------


def _make_record(
    *,
    target: str = "main.sales.orders",
    parquets: int = 1,
    mode: str = "append",
    op_id: str = "async-1",
    created_at: str = "2026-05-15T10:00:00+00:00",
) -> AsyncInsert:
    """Realistic record: a few staged parquets, a couple of options set."""
    return AsyncInsert(
        target_full_name=target,
        parquet_paths=tuple(
            f"/Volumes/main/sales/orders/.sql/async/insert/{op_id}-{i}.parquet"
            for i in range(parquets)
        ),
        metadata_paths=tuple(
            f"/Volumes/main/sales/orders/.sql/async/insert/{op_id}-{i}.json"
            for i in range(parquets)
        ),
        operation_ids=tuple(f"{op_id}-{i}" for i in range(parquets)),
        created_at=created_at,
        target_catalog_name="main",
        target_schema_name="sales",
        target_table_name="orders",
        target_field_names=("id", "label", "amount", "ts"),
        mode=mode,
        match_by=("id",) if mode == "append" else None,
        zorder_by=("ts",),
        where="active = true",
        prune_by=("ts",),
        prune_values={"ts": ("2026-05-15",)},
        safe_merge=False,
    )


def _make_record_batch(
    n: int,
    *,
    targets: int = 5,
    start_epoch: int = 1715760000,
) -> list[AsyncInsert]:
    """Build *n* records spread across *targets* tables in one schema.

    Mode rotation: every 10th record is an overwrite so the merge
    classmethod exercises the latest-overwrite-wipes path. ``created_at``
    is monotonic per-target so sort ordering inside :meth:`merge` is
    deterministic.
    """
    out: list[AsyncInsert] = []
    for i in range(n):
        target_idx = i % targets
        mode = "overwrite" if i % 10 == 9 else "append"
        out.append(
            _make_record(
                target=f"main.sales.t{target_idx}",
                op_id=f"async-{i:05d}",
                created_at=f"2026-05-15T10:00:{i % 60:02d}+00:00",
                mode=mode,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], repeat: int) -> dict:
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "samples": samples,
    }


def run_bench(records: int, repeat: int) -> list[dict]:
    """Drive the AsyncInsert hot paths with *records* records per call."""
    one = _make_record()
    one_with_5_parquets = _make_record(parquets=5)
    batch = _make_record_batch(records)

    # Serialization round-trips ----------------------------------------
    one_dict = one.to_dict()
    one_json = one.to_json_bytes()

    def to_dict_one() -> None:
        one.to_dict()

    def from_dict_one() -> None:
        AsyncInsert.from_dict(one_dict)

    def to_json_one() -> None:
        one.to_json_bytes()

    def from_json_one() -> None:
        AsyncInsert.from_json_bytes(one_json)

    # SQL rendering ----------------------------------------------------
    def to_sql_one_parquet() -> None:
        one.to_sql()

    def to_sql_five_parquets() -> None:
        one_with_5_parquets.to_sql()

    # Pairwise merge ---------------------------------------------------
    a = _make_record(op_id="async-a", created_at="2026-05-15T10:00:00+00:00")
    b = _make_record(op_id="async-b", created_at="2026-05-15T11:00:00+00:00")
    b_overwrite = _make_record(
        op_id="async-b", created_at="2026-05-15T11:00:00+00:00", mode="overwrite",
    )

    def merge_with_appends() -> None:
        a.merge_with(b)

    def merge_with_overwrite_wins() -> None:
        a.merge_with(b_overwrite)

    # N-record merge ---------------------------------------------------
    def merge_n_records() -> None:
        AsyncInsert.merge(batch)

    return [
        _time_one("to_dict (1 record)", to_dict_one, repeat),
        _time_one("from_dict (1 record)", from_dict_one, repeat),
        _time_one("to_json_bytes (1 record)", to_json_one, repeat),
        _time_one("from_json_bytes (1 record)", from_json_one, repeat),
        _time_one("to_sql (1 parquet)", to_sql_one_parquet, repeat),
        _time_one("to_sql (5 parquets)", to_sql_five_parquets, repeat),
        _time_one("merge_with (append+append)", merge_with_appends, repeat),
        _time_one("merge_with (append+overwrite)", merge_with_overwrite_wins, repeat),
        _time_one(f"merge ({records} records, 5 targets)", merge_n_records, repeat),
    ]


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:>32s}  "
        f"best={r['best']*1e6:9.2f} us  "
        f"median={r['median']*1e6:9.2f} us  "
        f"mean={r['mean']*1e6:9.2f} us"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    print(f"# records={args.records} repeat={args.repeat}")
    print(
        f"# {'label':>32s}  {'best':>14s}  {'median':>14s}  {'mean':>14s}"
    )
    for row in run_bench(args.records, args.repeat):
        print(_fmt_row(row))


if __name__ == "__main__":
    main()
