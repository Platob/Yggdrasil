"""End-to-end integration tests for :class:`yggdrasil.spark.frame.Dataset`.

Exercises the gnarly parts of the API against a real local
``SparkSession`` (skipped cleanly when PySpark isn't installed):

* Heterogeneous Python objects flowing through pickle-backed dynamic
  frames — datetimes, decimals, UUIDs, ``pathlib.Path``, dataclasses,
  nested dicts, sets, ``None``-valued fields.
* Schema inference (full-scan and ``limit=``) merging partition
  schemas in ``APPEND`` mode so fields that only appear in late rows
  still land in the merged schema with widened nullability.
* Dynamic → typed casts that preserve nested struct / list / map
  layouts through ``mapInArrow`` and survive a round trip back to
  dynamic via :meth:`Dataset.to_dynamic`.
* :meth:`Dataset.explode` over heterogeneous iterables, including
  empty iterables (which must drop without erroring on empty batches).
* :meth:`Dataset.apply` returning every tabular shape the cast
  registry knows — dict, dataclass, ``pyarrow.RecordBatch``,
  ``polars.DataFrame`` — fanned out in a single transform.
* :meth:`Dataset.filter` that drops every row (empty output
  partitions must not crash the typed-cast path) and predicate
  carriers that preserve ``installed_modules`` across a chain of
  transforms.
* :meth:`Dataset.parallelize` distributing a closure that
  captures live state by reference (caught via ``yggdrasil.pickle``
  serialization).

Per-test SIGALRM budget mirrors ``test_external_statement_data.py``
so a hung worker fails fast on POSIX runners.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import decimal
import os
import pathlib
import signal
import unittest
import uuid

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data import field, schema
from yggdrasil.data.types.primitive import (
    DateType,
    DecimalType,
    Float64Type,
    Int32Type,
    Int64Type,
    StringType,
)
from yggdrasil.spark.frame import DYNAMIC_SCHEMA, Dataset, is_dynamic_schema
from yggdrasil.spark.tests import SparkTestCase


_TEST_TIMEOUT_SECONDS = int(os.environ.get("YGG_SPARK_TEST_TIMEOUT", "120"))


def _has_alarm() -> bool:
    return hasattr(signal, "SIGALRM")


class _AlarmTimeout(Exception):
    pass


@dataclasses.dataclass
class _Row:
    """Top-level dataclass referenced by tests that exercise apply()
    returning dataclass instances. Must live at module scope so the
    pickle round trip used by ``yggdrasil.pickle`` resolves it back to
    the same class across the in-process Spark worker boundary.
    """

    id: int
    label: str
    score: float


class _DatasetTestBase(SparkTestCase, ArrowTestCase):
    """Shared SIGALRM-guarded setUp/tearDown for the integration tests.

    Also stubs out :func:`yggdrasil.spark.frame._install_modules_on_executors`
    so the per-transform ``_ensure_installed`` scan doesn't build /
    ship real ``yggdrasil`` / ``pyarrow`` / ``polars`` archives for
    every test that calls ``.map`` / ``.filter`` / ``.apply``. The
    scan itself still runs and populates ``installed_modules`` — we
    just don't pay the multi-hundred-MB archive-build cost on the
    local Spark session.
    """

    def setUp(self) -> None:
        super().setUp()
        if _has_alarm():
            self._prev_handler = signal.signal(
                signal.SIGALRM,
                lambda *_: (_ for _ in ()).throw(
                    _AlarmTimeout(
                        f"Spark test exceeded {_TEST_TIMEOUT_SECONDS}s budget"
                    )
                ),
            )
            signal.alarm(_TEST_TIMEOUT_SECONDS)

        # Stub: pretend every requested module installed cleanly. The shim
        # lives at module scope in ``yggdrasil.spark.frame`` and is
        # imported lazily inside ``Dataset._ensure_installed_on_session`` —
        # patching the module attribute is enough for the `from … import`
        # lookup to pick up the stub.
        from yggdrasil.spark import frame as _spark_frame

        self._spark_frame = _spark_frame
        self._orig_install = _spark_frame._install_modules_on_executors

        def _stub(session, modules):
            return set(modules)

        _spark_frame._install_modules_on_executors = _stub  # type: ignore[assignment]

    def tearDown(self) -> None:
        self._spark_frame._install_modules_on_executors = self._orig_install  # type: ignore[assignment]
        if _has_alarm():
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._prev_handler)
        super().tearDown()


# ---------------------------------------------------------------------------
# Dynamic mode — heterogeneous Python objects round-tripping via pickle
# ---------------------------------------------------------------------------


class TestDynamicHeterogeneous(_DatasetTestBase):

    def test_mixed_python_types_round_trip(self) -> None:
        """Mixed shapes (scalars, dataclasses, dicts, nested) survive
        the createDataFrame → collect round trip unchanged.

        The dynamic schema is just a binary column — anything yggdrasil's
        pickle can encode must come back ``==`` to the input.
        """
        items = [
            42,
            "héllo wörld — emoji 🚀",
            decimal.Decimal("3.1415926535897932384626"),
            dt.datetime(2024, 6, 1, 12, 30, 45, 123456, tzinfo=dt.timezone.utc),
            dt.date(1999, 12, 31),
            uuid.UUID("12345678-1234-5678-1234-567812345678"),
            pathlib.PurePosixPath("/tmp/file.parquet"),
            {"nested": {"deep": [1, 2, 3], "set": frozenset({"a", "b"})}},
            _Row(id=7, label="seven", score=7.0),
            None,
            (1, 2, 3),
            b"\x00\x01\x02\xff",
        ]
        frame = Dataset.from_iterable(items, spark_session=self.spark)

        self.assertTrue(frame.is_dynamic)
        self.assertTrue(is_dynamic_schema(frame.df.schema))
        self.assertEqual(frame.count(), len(items))

        # ``collect`` order isn't guaranteed across partitions; compare
        # multisets via repr (set comparison breaks on unhashable items
        # like dicts).
        got = sorted(repr(x) for x in frame.collect())
        want = sorted(repr(x) for x in items)
        self.assertEqual(got, want)

    def test_to_local_iterator_streams_in_order_within_partition(self) -> None:
        """``toLocalIterator`` yields rows partition-by-partition; with a
        single partition the order must equal the input order.
        """
        items = list(range(100))
        frame = Dataset.from_iterable(items, spark_session=self.spark)
        frame = Dataset(frame.df.coalesce(1))
        self.assertEqual(list(frame.to_local_iterator()), items)

    def test_filter_drops_all_rows_without_crashing(self) -> None:
        """An empty-output partition would feed an empty batch into the
        downstream pipeline — must not raise on ``RecordBatch.from_pylist
        ([])`` or Arrow-typing emptiness checks.
        """
        frame = Dataset.from_iterable(range(50), spark_session=self.spark)
        empty = frame.filter(lambda x: x > 1_000_000)
        self.assertEqual(empty.collect(), [])
        self.assertEqual(empty.count(), 0)


# ---------------------------------------------------------------------------
# Schema inference — partition merge in APPEND mode
# ---------------------------------------------------------------------------


class TestInferSchema(_DatasetTestBase):

    def test_full_scan_unions_sparse_fields(self) -> None:
        """Rows have *different* key sets — the merged schema must be
        the union of all keys, with nullability widened wherever a key
        is absent in at least one row.
        """
        items = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b", "score": 0.5},
            {"id": 3, "score": 0.75, "tags": ["x", "y"]},
        ]
        frame = Dataset.from_iterable(items, spark_session=self.spark)
        inferred = frame.infer_schema()

        names = {f.name for f in inferred.fields}
        self.assertEqual(names, {"id", "name", "score", "tags"})
        # ``name`` / ``score`` / ``tags`` each missing in at least one
        # row → nullable in the merged schema.
        for name in ("name", "score", "tags"):
            self.assertTrue(
                inferred.field(name=name).nullable,
                f"expected {name!r} to be nullable after APPEND merge",
            )

    def test_limit_path_short_circuits(self) -> None:
        """``limit=`` walks only the first N rows on the driver — the
        merged schema must reflect only the keys observed in that window.
        """
        items = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
            # Field that only appears in a row past the limit — must be
            # absent from the inferred schema.
            {"id": 3, "name": "third", "late": 1.0},
        ]
        frame = Dataset.from_iterable(items, spark_session=self.spark)
        frame = Dataset(frame.df.coalesce(1))
        partial = frame.infer_schema(limit=2, inplace=False)
        self.assertEqual({f.name for f in partial.fields}, {"id", "name"})

    def test_empty_frame_raises(self) -> None:
        frame = Dataset.from_iterable([], spark_session=self.spark)
        with self.assertRaises(ValueError):
            frame.infer_schema()


# ---------------------------------------------------------------------------
# Typed mode — nested cast, explode, to_dynamic round trip
# ---------------------------------------------------------------------------


class TestTypedCast(_DatasetTestBase):

    def _nested_schema(self):
        return schema([
            field("id", Int64Type, nullable=False),
            field("when", pa.timestamp("us", tz="UTC")),
            field("amount", DecimalType(precision=18, scale=4)),
            field("tags", pa.list_(pa.string())),
            field(
                "child",
                pa.struct([
                    pa.field("k", pa.string()),
                    pa.field("v", pa.float64()),
                ]),
            ),
        ])

    def test_dynamic_to_typed_preserves_nested_values(self) -> None:
        sch = self._nested_schema()
        items = [
            {
                "id": 1,
                "when": dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
                "amount": decimal.Decimal("1234.5678"),
                "tags": ["red", "green"],
                "child": {"k": "alpha", "v": 1.5},
            },
            {
                "id": 2,
                "when": dt.datetime(2024, 1, 3, 0, 0, 0, tzinfo=dt.timezone.utc),
                "amount": decimal.Decimal("0.0001"),
                "tags": [],
                "child": {"k": "beta", "v": -2.25},
            },
        ]
        dyn = Dataset.from_iterable(items, spark_session=self.spark)
        typed = dyn.cast(sch)

        self.assertFalse(typed.is_dynamic)
        rows = sorted(typed.collect(), key=lambda r: r["id"])
        self.assertEqual(rows[0]["id"], 1)
        self.assertEqual(rows[0]["child"], {"k": "alpha", "v": 1.5})
        self.assertEqual(rows[0]["tags"], ["red", "green"])
        self.assertEqual(rows[0]["amount"], decimal.Decimal("1234.5678"))
        self.assertEqual(rows[1]["tags"], [])

    def test_typed_to_dynamic_round_trip(self) -> None:
        """typed → to_dynamic() → cast(same schema) round trips losslessly."""
        sch = self._nested_schema()
        items = [
            {
                "id": i,
                "when": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                "amount": decimal.Decimal(f"{i}.0000"),
                "tags": [f"t{i}"],
                "child": {"k": f"k{i}", "v": float(i)},
            }
            for i in range(5)
        ]
        typed = Dataset.from_iterable(items, sch, spark_session=self.spark)
        dyn = typed.to_dynamic()
        self.assertTrue(dyn.is_dynamic)

        rebuilt = dyn.cast(sch)
        rows = sorted(rebuilt.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], list(range(5)))
        self.assertEqual(
            [r["amount"] for r in rows],
            [decimal.Decimal(f"{i}.0000") for i in range(5)],
        )

    def test_explode_with_typed_schema(self) -> None:
        """explode() on a dynamic frame of iterables — typed output
        keeps the cast schema across heterogeneous element sources.
        """
        sch = schema([field("id", Int32Type), field("name", StringType)])
        items = [
            [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            [{"id": 3, "name": "c"}],
            # Empty iterable — explode must drop it without
            # synthesising an empty batch.
            [],
            [{"id": 4, "name": "d"}, {"id": 5, "name": "e"}],
        ]
        dyn = Dataset.from_iterable(items, spark_session=self.spark)
        exploded = dyn.explode(sch)
        rows = sorted(exploded.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual([r["name"] for r in rows], ["a", "b", "c", "d", "e"])

    def test_explode_typed_mode_rejected(self) -> None:
        """explode is dynamic-only — typed rows are dicts, not iterables."""
        sch = schema([field("id", Int32Type)])
        typed = Dataset.from_iterable(
            [{"id": i} for i in range(3)], sch, spark_session=self.spark,
        )
        with self.assertRaises(TypeError):
            typed.explode()


# ---------------------------------------------------------------------------
# apply() — fan out tabular shapes via the cast registry
# ---------------------------------------------------------------------------


class TestApplyTabularShapes(_DatasetTestBase):

    def test_apply_returning_dataclass(self) -> None:
        sch = schema([
            field("id", Int64Type, nullable=False),
            field("label", StringType),
            field("score", Float64Type),
        ])

        def make_row(i: int) -> _Row:
            return _Row(id=i, label=f"row-{i}", score=float(i) * 1.5)

        out = Dataset.parallelize(
            make_row, range(8), schema=sch, spark_session=self.spark,
        )
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["id"] for r in rows], list(range(8)))
        self.assertEqual(rows[3]["label"], "row-3")
        self.assertEqual(rows[3]["score"], 4.5)

    def test_apply_returning_arrow_record_batch(self) -> None:
        """apply() may return a pyarrow.RecordBatch — the cast pipeline
        absorbs it via the registered any→arrow converters.
        """
        sch = schema([
            field("id", Int64Type, nullable=False),
            field("doubled", Int64Type),
        ])

        def expand(seed: int) -> pa.RecordBatch:
            return pa.RecordBatch.from_pydict(
                {"id": [seed], "doubled": [seed * 2]},
                schema=sch.to_arrow_schema(),
            )

        dyn = Dataset.from_iterable(range(5), spark_session=self.spark)
        out = dyn.apply(expand, sch)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["doubled"] for r in rows], [0, 2, 4, 6, 8])

    def test_apply_returning_polars_frame(self) -> None:
        try:
            import polars as pl  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")

        sch = schema([
            field("id", Int64Type, nullable=False),
            field("squared", Int64Type),
        ])

        def expand(seed: int):
            import polars as pl
            return pl.DataFrame({"id": [seed], "squared": [seed * seed]})

        dyn = Dataset.from_iterable(range(4), spark_session=self.spark)
        out = dyn.apply(expand, sch)
        rows = sorted(out.collect(), key=lambda r: r["id"])
        self.assertEqual([r["squared"] for r in rows], [0, 1, 4, 9])


# ---------------------------------------------------------------------------
# Chained transforms — installed_modules + lineage preservation
# ---------------------------------------------------------------------------


class TestTransformLineage(_DatasetTestBase):

    def test_installed_modules_propagate_through_chain(self) -> None:
        """map → filter → cast keeps the discovered module set on every
        intermediate frame so the executor doesn't re-ship the same
        wheel for each transform.
        """
        seeded = {"ygg-test-module"}
        frame = Dataset.from_iterable(
            range(10), spark_session=self.spark,
        )
        # Seed directly so we exercise the propagation without
        # triggering a real archive build.
        frame.installed_modules.update(seeded)

        mapped = frame.map(lambda x: x + 1)
        filtered = mapped.filter(lambda x: x % 2 == 0)
        out_schema = schema([field("v", Int64Type)])
        cast = filtered.apply(lambda x: {"v": x}, out_schema)

        for f in (mapped, filtered, cast):
            self.assertTrue(seeded.issubset(f.installed_modules))

    def test_filter_then_map_preserves_count(self) -> None:
        frame = Dataset.from_iterable(range(100), spark_session=self.spark)
        out = frame.filter(lambda x: x % 3 == 0).map(lambda x: x * 10)
        rows = sorted(out.collect())
        self.assertEqual(rows, [i * 10 for i in range(100) if i % 3 == 0])

    def test_parallelize_with_closure_capture(self) -> None:
        """``parallelize`` must pickle the closure cells so executors
        see the same captured value the driver did.
        """
        multiplier = 7
        offset = 100

        def fn(x: int) -> int:
            return x * multiplier + offset

        sch = schema([field("y", Int64Type)])
        out = Dataset.parallelize(
            lambda x: {"y": fn(x)}, range(5),
            schema=sch, spark_session=self.spark,
        )
        rows = sorted(r["y"] for r in out.collect())
        self.assertEqual(rows, [fn(i) for i in range(5)])


# ---------------------------------------------------------------------------
# from_iterable + spark schema export
# ---------------------------------------------------------------------------


class TestTypedConstruction(_DatasetTestBase):

    def test_from_iterable_typed_matches_spark_schema(self) -> None:
        sch = schema([
            field("id", Int64Type, nullable=False),
            field("at", DateType),
            field("name", StringType),
        ])
        items = [
            {"id": 1, "at": dt.date(2024, 1, 1), "name": "a"},
            {"id": 2, "at": dt.date(2024, 6, 1), "name": "b"},
        ]
        frame = Dataset.from_iterable(items, sch, spark_session=self.spark)
        self.assertFalse(frame.is_dynamic)

        spark_names = [f.name for f in frame.df.schema.fields]
        self.assertEqual(spark_names, ["id", "at", "name"])

        # Round-trip back through Arrow preserves field order + types.
        tbl = frame.toArrow()
        self.assertEqual(tbl.column_names, ["id", "at", "name"])
        self.assertEqual(tbl.num_rows, 2)

    def test_typed_to_polars(self) -> None:
        try:
            import polars as pl
        except ImportError:
            self.skipTest("polars not installed")

        sch = schema([
            field("id", Int64Type, nullable=False),
            field("v", Float64Type),
        ])
        items = [{"id": i, "v": float(i) / 2.0} for i in range(6)]
        frame = Dataset.from_iterable(items, sch, spark_session=self.spark)
        pdf = frame.toPolars()
        self.assertIsInstance(pdf, pl.DataFrame)
        self.assertEqual(pdf.columns, ["id", "v"])
        self.assertEqual(pdf.sort("id")["v"].to_list(), [i / 2.0 for i in range(6)])


# ---------------------------------------------------------------------------
# Dynamic schema sanity
# ---------------------------------------------------------------------------


class TestDynamicSchemaShape(_DatasetTestBase):

    def test_is_dynamic_schema_detects_pickle_column(self) -> None:
        self.assertTrue(is_dynamic_schema(DYNAMIC_SCHEMA))
        self.assertTrue(is_dynamic_schema(DYNAMIC_SCHEMA.to_arrow_schema()))

        # Anything else must not match.
        wrong = schema([field("not_pickle", pa.binary())])
        self.assertFalse(is_dynamic_schema(wrong))
        wrong = schema([field("_pickle", pa.string())])
        self.assertFalse(is_dynamic_schema(wrong))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
