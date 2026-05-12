"""Benchmark engine-type-equality bypass on the arrow / polars / spark cast paths.

What this measures
------------------

Every cast site already has a Field-level ``need_cast`` short-circuit.
The bypass added on top compares the *engine-native* type — the
``pa.DataType`` / ``pl.DataType`` / Spark ``StructType`` already attached
to the source — against the target. The Field abstraction can mark a
cast as needed because of metadata, semantic subclass, or extension
detail that the underlying engine type does not carry; when the engine
types already match, the cast is value-free and the bypass returns the
source unchanged.

A/B comparison without runtime toggles
--------------------------------------

The benchmark constructs two source frames per scenario:

* ``match``  — source engine type identical to the target. The bypass
  fires; the cost we measure is the comparison + identity return.
* ``cast``   — source engine type differs by one width step
  (``int32`` → ``int64`` for arrow / polars, equivalent for Spark).
  The bypass does not fire and the engine actually casts; this
  approximates the work the bypass saves on the ``match`` path.

Numbers are wall-clock per call across ``--repeat`` iterations. The
``match`` row should land at single-digit microseconds; the ``cast``
row scales with row count and engine.

Usage::

    PYTHONPATH=src python benchmarks/bench_engine_type_bypass.py
    PYTHONPATH=src python benchmarks/bench_engine_type_bypass.py --rows 200000 --repeat 5
    PYTHONPATH=src python benchmarks/bench_engine_type_bypass.py --engines arrow,polars
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema


# ---------------------------------------------------------------------------
# Target schema — a representative mix of width / string / timestamp columns,
# matching the kind of frame an integration would hand to ``ParquetIO`` /
# ``Tabular`` with a bound target schema.
# ---------------------------------------------------------------------------


def _build_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("amount", pa.float64()),
            pa.field("qty", pa.int32()),
            pa.field("name", pa.string()),
            pa.field("ts", pa.timestamp("us")),
            pa.field("active", pa.bool_()),
        ]
    )


def _build_arrow_match(rows: int) -> pa.Table:
    rng = _rng()
    ts_arr = _ts(rng, rows, "us")
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int64()),
            "amount": pa.array(rng.normal(0.0, 1.0, size=rows), type=pa.float64()),
            "qty": pa.array(rng.integers(0, 1000, size=rows).astype("int32")),
            "name": pa.array([f"row-{i}" for i in range(rows)], type=pa.string()),
            "ts": pa.array(ts_arr, type=pa.timestamp("us")),
            "active": pa.array((rng.integers(0, 2, size=rows) == 1).tolist(), type=pa.bool_()),
        }
    )


def _build_arrow_cast(rows: int) -> pa.Table:
    # One-step-narrower / timezone-shifted source. ``id`` becomes
    # int32 instead of int64 so the engine bypass can NOT fire and
    # pyarrow performs a real widening cast.
    rng = _rng()
    ts_arr = _ts(rng, rows, "ms")
    return pa.table(
        {
            "id": pa.array(range(rows), type=pa.int32()),
            "amount": pa.array(rng.normal(0.0, 1.0, size=rows), type=pa.float32()),
            "qty": pa.array(rng.integers(0, 1000, size=rows).astype("int16")),
            "name": pa.array([f"row-{i}" for i in range(rows)], type=pa.string()),
            "ts": pa.array(ts_arr, type=pa.timestamp("ms")),
            "active": pa.array((rng.integers(0, 2, size=rows) == 1).tolist(), type=pa.bool_()),
        }
    )


def _rng():
    import numpy as np
    return np.random.default_rng(7)


def _ts(rng, rows: int, unit: str):
    import numpy as np
    base = np.datetime64("2024-01-01")
    deltas = rng.integers(0, 86_400 * 365, size=rows).astype("timedelta64[s]")
    return (base + deltas).astype(f"datetime64[{unit}]")


# ---------------------------------------------------------------------------
# Runner — one ``cast_*_tabular`` call per timing sample.
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], repeat: int) -> dict:
    # Warm-up — first call pays JIT / lazy-import costs.
    fn()
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
    }


def _fmt(r: dict) -> str:
    return (
        f"{r['label']:>34s}  "
        f"best={r['best']*1e6:9.1f} us  "
        f"median={r['median']*1e6:9.1f} us  "
        f"mean={r['mean']*1e6:9.1f} us"
    )


# ---------------------------------------------------------------------------
# Arrow
# ---------------------------------------------------------------------------


def _semantic_source_field(target_field: Field) -> Field:
    """Build a source Field that lowers to the same arrow schema but
    differs at the yggdrasil DataType level.

    Swaps every ``StringType`` child for ``SJsonType`` (string-backed
    JSON). Both lower to ``pa.string()`` so the engine schema stays
    identical, but ``Field.equals`` returns ``False`` — the case the
    new engine-level bypass exists to catch (Field is "too precise"
    relative to the real engine type).
    """
    from yggdrasil.data.types.primitive.json import SJsonType
    from yggdrasil.data.types.primitive import StringType

    new_children = []
    for ch in target_field.children_fields:
        dtype = SJsonType() if isinstance(ch.dtype, StringType) else ch.dtype
        new_children.append(
            Field(name=ch.name, dtype=dtype, nullable=ch.nullable)
        )
    return Field(name=target_field.name, dtype=Schema(inner_fields=new_children).dtype, nullable=target_field.nullable)


def bench_arrow(rows: int, repeat: int) -> list[dict]:
    target_schema = _build_arrow_schema()
    target_field = Schema.from_arrow(target_schema).to_field()
    # Pre-bind source — production paths (``any_to_arrow_table``,
    # ``cast_arrow_record_batch_reader``, …) all bind source via
    # ``_bind_source`` before dispatching. Without it the cast routes
    # through the ``src is None`` early-return, which would mask both
    # the engine-level bypass and the existing children-equality bypass.
    opts_match = CastOptions(target_field=target_field).check_source(
        obj=_build_arrow_match(rows),
        copy=True,
    )
    opts_cast = CastOptions(target_field=target_field).check_source(
        obj=_build_arrow_cast(rows),
        copy=True,
    )
    # Semantic-only divergence: Field carries ``SJsonType`` while the
    # source pa.Schema has ``pa.string()`` — Field equality says
    # "differ", engine equality says "same". Exercises the new bypass
    # without the existing children-Field shortcut firing.
    opts_semantic = CastOptions(
        target_field=target_field,
        source_field=_semantic_source_field(target_field),
    )

    table_match = _build_arrow_match(rows)
    table_cast = _build_arrow_cast(rows)

    def call_match() -> None:
        opts_match.cast_arrow_tabular(table_match)

    def call_cast() -> None:
        opts_cast.cast_arrow_tabular(table_cast)

    def call_semantic() -> None:
        opts_semantic.cast_arrow_tabular(table_match)

    return [
        _time_one("arrow/match    (bypass fires)", call_match, repeat),
        _time_one("arrow/semantic (engine bypass)", call_semantic, repeat),
        _time_one("arrow/cast     (engine works)", call_cast, repeat),
    ]


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------


def bench_polars(rows: int, repeat: int) -> list[dict]:
    import polars as pl

    target_field = Schema.from_arrow(_build_arrow_schema()).to_field()

    df_match = pl.from_arrow(_build_arrow_match(rows))
    df_cast = pl.from_arrow(_build_arrow_cast(rows))

    opts_match = CastOptions(target_field=target_field).check_source(
        obj=df_match, copy=True,
    )
    opts_cast = CastOptions(target_field=target_field).check_source(
        obj=df_cast, copy=True,
    )

    def call_match() -> None:
        opts_match.cast_polars_tabular(df_match)

    def call_cast() -> None:
        opts_cast.cast_polars_tabular(df_cast)

    return [
        _time_one("polars/match (bypass fires)", call_match, repeat),
        _time_one("polars/cast  (engine works)", call_cast, repeat),
    ]


# ---------------------------------------------------------------------------
# Spark — gated; spinning up a local SparkSession costs ~5s and is only
# useful when the user actually wants the Spark numbers.
# ---------------------------------------------------------------------------


def bench_spark(rows: int, repeat: int) -> list[dict]:
    try:
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True, import_error=True, install_spark=False)
    except Exception as exc:
        return [{"label": f"spark/skipped ({exc.__class__.__name__})", "best": 0.0, "median": 0.0, "mean": 0.0}]

    target_field = Schema.from_arrow(_build_arrow_schema()).to_field()
    options = CastOptions(target_field=target_field)

    df_match = spark.createDataFrame(_build_arrow_match(rows).to_pandas())
    df_cast = spark.createDataFrame(_build_arrow_cast(rows).to_pandas())

    def call_match() -> None:
        options.cast_spark_tabular(df_match)

    def call_cast() -> None:
        options.cast_spark_tabular(df_cast)

    # Spark cast is plan-only (lazy) — we measure the planning + select
    # rebuild cost, which is what the bypass actually shortens. Triggering
    # ``count`` here would mix in JVM execution time and drown out the
    # comparison we care about.
    return [
        _time_one("spark/match (bypass fires)", call_match, repeat),
        _time_one("spark/cast  (engine works)", call_cast, repeat),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


ENGINE_DISPATCH: dict[str, Callable[[int, int], list[dict]]] = {
    "arrow": bench_arrow,
    "polars": bench_polars,
    "spark": bench_spark,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument(
        "--engines",
        default="arrow,polars",
        help="Comma-separated subset of {arrow, polars, spark}. Spark is opt-in.",
    )
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINE_DISPATCH]
    if unknown:
        raise SystemExit(f"Unknown engine(s): {unknown}. Pick from {list(ENGINE_DISPATCH)}.")

    print(f"# rows={args.rows} repeat={args.repeat}")
    print(f"# {'label':>34s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for engine in engines:
        for row in ENGINE_DISPATCH[engine](args.rows, args.repeat):
            print(_fmt(row))


if __name__ == "__main__":
    main()
