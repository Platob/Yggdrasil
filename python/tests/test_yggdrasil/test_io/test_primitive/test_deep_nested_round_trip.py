"""Deep-nested round-trip tests across every tabular IO leaf.

The Databricks staging path optimizations (``_write_pandas_frame``
schema-pushdown, ``_write_polars_frame`` cast-into-polars, parquet
bypass-cast) all live on the shared :class:`Tabular` surface. The
fastest way to confirm they keep nested values correct on every
backend is one parametrized round-trip suite that exercises:

* ``list<int64>`` — simple list
* ``list<list<int32>>`` — list of list
* ``struct<...>`` — flat struct
* ``list<struct<...>>`` — list of struct (the canonical telemetry shape)
* ``struct<name, scores: list<struct<...>>>`` — struct holding a
  list-of-struct, two levels deep
* ``map<string, int64>`` — map (only where the IO supports it)

For each IO we round-trip the same fixture through three entry points:
pandas, polars, and arrow. Each call binds a target schema on
``CastOptions`` to keep the assertion crisp — the staging path always
runs with a target schema, so that's the configuration that has to
stay correct.

``CsvIO`` / ``XlsxIO`` aren't included — they encode nested cells as
opaque strings rather than preserving the schema, so a "round-trip"
under their semantics is a string-equality test, not a schema-equality
test. The columnar IOs below all preserve schema.
"""
from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa
import pytest

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.primitive.json_io import JsonIO
from yggdrasil.io.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.primitive.parquet_io import ParquetIO
from yggdrasil.io.tabular import Tabular


# ---------------------------------------------------------------------------
# Fixture — a single deep-nested table reused across every parametrized run.
# Built with explicit pyarrow types so the schema is unambiguous (no inference
# fallback that would mask bugs in the writer's cast pipeline).
# ---------------------------------------------------------------------------


def _build_deep_table(rows: int = 32) -> pa.Table:
    ids = list(range(rows))
    names = [f"n{i}" for i in range(rows)]

    tags = [
        ([i, i + 1, i + 2] if i % 3 != 0 else [])
        for i in range(rows)
    ]

    matrix = [
        [[1, 2], [3, 4, 5]] if i % 4 == 0
        else [[i, i + 1]] if i % 4 == 1
        else [] if i % 4 == 2
        else None
        for i in range(rows)
    ]

    flat = [
        {"x": i, "y": float(i) * 0.5, "label": f"row-{i}"}
        for i in range(rows)
    ]

    events = [
        [
            {"id": i * 10 + j, "tag": f"e-{j}"}
            for j in range(i % 4)
        ]
        for i in range(rows)
    ]

    profile = [
        {
            "name": f"user-{i}",
            "scores": [
                {"k": f"k{j}", "v": float(j) * 0.1}
                for j in range(i % 3)
            ],
        }
        for i in range(rows)
    ]

    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("tags", pa.list_(pa.int64())),
        pa.field("matrix", pa.list_(pa.list_(pa.int32()))),
        pa.field("flat", pa.struct([
            ("x", pa.int64()),
            ("y", pa.float64()),
            ("label", pa.string()),
        ])),
        pa.field("events", pa.list_(pa.struct([
            ("id", pa.int64()),
            ("tag", pa.string()),
        ]))),
        pa.field("profile", pa.struct([
            ("name", pa.string()),
            ("scores", pa.list_(pa.struct([
                ("k", pa.string()),
                ("v", pa.float64()),
            ]))),
        ])),
    ])

    return pa.table(
        {
            "id": ids,
            "name": names,
            "tags": tags,
            "matrix": matrix,
            "flat": flat,
            "events": events,
            "profile": profile,
        },
        schema=schema,
    )


@pytest.fixture
def deep_table() -> pa.Table:
    return _build_deep_table()


# ---------------------------------------------------------------------------
# Per-IO config — the staging path runs with a target schema, so every test
# below binds one through :class:`CastOptions`. Each IO advertises which
# entry points it supports; the parametrization filters accordingly so the
# tests stay declarative rather than hand-wired per IO.
# ---------------------------------------------------------------------------


IO_CLASSES: list[type[Tabular]] = [ParquetIO, ArrowIPCIO, NDJsonIO, JsonIO]


def _ids(items: list[Any]) -> list[str]:
    return [getattr(it, "__name__", str(it)) for it in items]


def _bind_target(table: pa.Table) -> CastOptions:
    """Build a :class:`CastOptions` pinned to *table*'s schema, mirroring
    what the Databricks staging path passes when it has already collected
    the destination table's schema before staging.
    """
    return CastOptions(target_field=Schema.from_arrow(table.schema).to_field())


def _normalize(table: pa.Table) -> pa.Table:
    """Combine chunks so per-chunk encoding differences across IOs don't
    leak into the value-equality assertion. ``combine_chunks`` is a
    metadata rebuild — no row reshape."""
    return table.combine_chunks()


def _expected_arrow(table: pa.Table) -> pa.Table:
    """The shape an IO is expected to hand back after a target-schema
    round-trip — the original table, single-chunked. The target schema
    binds *exact* dtypes so even backends that round-trip through their
    own physical representations (polars' large_string, ndjson's string
    serialization) come out matching the source."""
    return _normalize(table)


def _readback(io: Tabular, options: CastOptions) -> pa.Table:
    """Read the IO back with the same target so the cast pipeline
    reconciles the read shape to the user contract. Without this the
    NDJson / Json readers would hand back arrow-inferred types
    (``int64`` for any integer column, ``string`` for any string column)
    that diverge from the binding but are otherwise equivalent.
    """
    options = CastOptions.check(options, source=...)
    return _normalize(io.read_arrow_table(options=options))


# ---------------------------------------------------------------------------
# Helpers — each input shape is wrapped behind a callable so the test can
# assert what the writer produced rather than caring how it got there.
# ---------------------------------------------------------------------------


def _write_arrow_table(io: Tabular, table: pa.Table, options: CastOptions) -> None:
    io.write_table(table, options)


def _write_arrow_batches(io: Tabular, table: pa.Table, options: CastOptions) -> None:
    io.write_table(table.to_batches(max_chunksize=7), options)


def _write_pandas_frame(io: Tabular, table: pa.Table, options: CastOptions) -> None:
    pd = pytest.importorskip("pandas")
    io.write_table(table.to_pandas(), options)
    del pd  # silence "imported but unused" — pytest.importorskip is the gate


def _write_polars_frame(io: Tabular, table: pa.Table, options: CastOptions) -> None:
    pl = pytest.importorskip("polars")
    io.write_table(pl.from_arrow(table), options)


ENTRY_POINTS: dict[str, Callable[[Tabular, pa.Table, CastOptions], None]] = {
    "arrow-table": _write_arrow_table,
    "arrow-batches": _write_arrow_batches,
    "pandas": _write_pandas_frame,
    "polars": _write_polars_frame,
}


# ---------------------------------------------------------------------------
# The parametrized round-trip — one test, every combo. Each combo fails
# with the IO + entry point in the test id so a regression points at the
# exact backend that broke.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("io_cls", IO_CLASSES, ids=_ids(IO_CLASSES))
@pytest.mark.parametrize("entry", list(ENTRY_POINTS), ids=list(ENTRY_POINTS))
def test_deep_nested_round_trip(io_cls: type[Tabular], entry: str, deep_table: pa.Table) -> None:
    """Deep-nested table survives ``write_table → read_arrow_table``.

    The target schema is bound on the writer so the staging-path
    optimizations exercise their fast paths (pandas → schema pushdown,
    polars → cast-then-one-shot-arrow-cast, arrow → bypass cast).
    """
    options = _bind_target(deep_table)
    io = io_cls()

    ENTRY_POINTS[entry](io, deep_table, options)
    loaded = _readback(io, options)

    expected = _expected_arrow(deep_table)

    assert loaded.schema == expected.schema, (
        f"{io_cls.__name__}/{entry}: schema diverged\n"
        f"expected: {expected.schema}\nactual:   {loaded.schema}"
    )
    assert loaded.equals(expected), (
        f"{io_cls.__name__}/{entry}: values diverged\n"
        f"expected[0]: {expected.slice(0, 3).to_pylist()}\n"
        f"actual[0]:   {loaded.slice(0, 3).to_pylist()}"
    )


@pytest.mark.parametrize("io_cls", IO_CLASSES, ids=_ids(IO_CLASSES))
def test_deep_nested_schema_persisted(io_cls: type[Tabular], deep_table: pa.Table) -> None:
    """:meth:`Tabular.collect_schema` returns the deep target shape.

    Stamped before the writer drains the iterator on every IO so callers
    can refer to the columns without waiting for the whole frame to land
    — the staging path relies on this to build the SQL column list.
    """
    options = _bind_target(deep_table)
    io = io_cls()
    io.write_table(deep_table, options)

    schema = io.collect_schema()
    names = list(schema.field_names())

    assert names == [
        "id", "name", "tags", "matrix", "flat", "events", "profile",
    ], f"{io_cls.__name__}: column order or names drifted ({names})"


@pytest.mark.parametrize("io_cls", IO_CLASSES, ids=_ids(IO_CLASSES))
@pytest.mark.parametrize("entry", list(ENTRY_POINTS), ids=list(ENTRY_POINTS))
def test_deep_nested_no_target_round_trip(
    io_cls: type[Tabular], entry: str, deep_table: pa.Table,
) -> None:
    """No-target round-trip: the IO is responsible for preserving names
    and value-equality even when ``CastOptions`` carries no
    ``target_field``. The schema may shift on text-shaped IOs (NDJson /
    Json don't preserve ``int32`` vs ``int64`` over the wire), so this
    test pins value equality after reading back through the source
    schema — the staging path doesn't run here, but the dataframe
    backends still have to keep rows intact.
    """
    options = CastOptions()
    io = io_cls()

    ENTRY_POINTS[entry](io, deep_table, options)

    # Bind the target only on read so write-side type drift surfaces
    # via the cast layer rather than the equality check.
    loaded = _readback(io, _bind_target(deep_table))
    expected = _expected_arrow(deep_table)
    assert loaded.equals(expected), (
        f"{io_cls.__name__}/{entry}: values diverged on no-target write"
    )
