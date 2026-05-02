"""Tests for ``yggdrasil.spark.frame`` — the pickle-shipping ``DynamicFrame``.

The contract under test is:

* the dynamic schema is a single non-nullable binary column named
  ``_pickle`` and ``is_dynamic_schema`` recognizes it (and rejects
  anything else)
* ``inputs_map_partition`` / ``outputs_map_partition`` round-trip pickled
  payloads through Arrow batches with the documented ``byte_size``
  flushing behavior
* ``DynamicFrame.parallelize`` / ``map`` / ``explode`` / ``cast`` /
  ``toArrow`` / ``collect`` / ``to_local_iterator`` all preserve order
  and values across the Spark boundary

Tests that don't need a SparkSession live in ``TestPickleHelpers`` and
``TestDynamicSchemaHelpers`` so they still run on a base install.
"""
from __future__ import annotations

import unittest
from typing import Any, Iterator

import pyarrow as pa

from yggdrasil.data import schema as schema_builder, field as field_builder
from yggdrasil.pickle.ser.serde import dumps, loads
from yggdrasil.spark.frame import (
    DYNAMIC_SCHEMA,
    PICKLE_COLUMN_NAME,
    DynamicFrame,
    _ARROW_DYNAMIC_SCHEMA,
    _spark_dynamic_schema,
    inputs_map_partition,
    is_dynamic_schema,
    outputs_map_partition,
)
from yggdrasil.spark.tests import SparkTestCase


def _spark_schema_is_dynamic(spark_schema: Any) -> bool:
    """Direct shape check on a pyspark ``StructType``.

    ``is_dynamic_schema`` is keyed on the yggdrasil ``Schema`` form, and
    pyspark ``StructType`` doesn't round-trip through ``Schema.from_any``
    in a way that exposes the inner binary leaf — so we inspect the
    pyspark side directly when validating a Spark DataFrame's schema.
    """
    from pyspark.sql.types import BinaryType

    if len(spark_schema.fields) != 1:
        return False
    f = spark_schema.fields[0]
    return f.name == PICKLE_COLUMN_NAME and isinstance(f.dataType, BinaryType)


# ---------------------------------------------------------------------------
# Pure-Python helpers — no SparkSession required
# ---------------------------------------------------------------------------
class TestDynamicSchemaHelpers(unittest.TestCase):
    """Module constants and ``is_dynamic_schema`` recognition."""

    def test_pickle_column_name_is_underscore_pickle(self) -> None:
        self.assertEqual(PICKLE_COLUMN_NAME, "_pickle")

    def test_dynamic_schema_has_single_binary_field(self) -> None:
        self.assertEqual(len(DYNAMIC_SCHEMA), 1)
        f = DYNAMIC_SCHEMA.get(0)
        self.assertEqual(f.name, PICKLE_COLUMN_NAME)
        self.assertTrue(pa.types.is_binary(f.arrow_type))
        self.assertFalse(f.nullable)

    def test_arrow_dynamic_schema_matches(self) -> None:
        self.assertEqual(len(_ARROW_DYNAMIC_SCHEMA), 1)
        af = _ARROW_DYNAMIC_SCHEMA.field(0)
        self.assertEqual(af.name, PICKLE_COLUMN_NAME)
        self.assertTrue(pa.types.is_binary(af.type))
        self.assertFalse(af.nullable)

    def test_spark_dynamic_schema_matches(self) -> None:
        from pyspark.sql.types import BinaryType

        spark_schema = _spark_dynamic_schema()
        self.assertEqual(len(spark_schema.fields), 1)
        sf = spark_schema.fields[0]
        self.assertEqual(sf.name, PICKLE_COLUMN_NAME)
        self.assertIsInstance(sf.dataType, BinaryType)
        self.assertFalse(sf.nullable)

    def test_is_dynamic_schema_accepts_canonical_schema(self) -> None:
        # The recognizer is keyed on the yggdrasil Schema shape.
        self.assertTrue(is_dynamic_schema(DYNAMIC_SCHEMA))

    def test_is_dynamic_schema_rejects_wrong_column_name(self) -> None:
        bad = schema_builder([
            field_builder("payload", arrow_type=pa.binary(), nullable=False),
        ])
        self.assertFalse(is_dynamic_schema(bad))

    def test_is_dynamic_schema_rejects_wrong_arrow_type(self) -> None:
        bad = schema_builder([
            field_builder(PICKLE_COLUMN_NAME, arrow_type=pa.string(), nullable=False),
        ])
        self.assertFalse(is_dynamic_schema(bad))

    def test_is_dynamic_schema_rejects_multi_column(self) -> None:
        bad = schema_builder([
            field_builder(PICKLE_COLUMN_NAME, arrow_type=pa.binary(), nullable=False),
            field_builder("extra", arrow_type=pa.int64()),
        ])
        self.assertFalse(is_dynamic_schema(bad))


# ---------------------------------------------------------------------------
# Partition iterators — exercise the per-partition logic without Spark
# ---------------------------------------------------------------------------
class TestPickleHelpers(unittest.TestCase):
    """``inputs_map_partition`` / ``outputs_map_partition`` direct invocation.

    Spark calls these with an iterator of ``pa.RecordBatch`` and expects
    a generator of ``pa.RecordBatch`` back. We feed them by hand here so
    we can pin down ordering, byte_size flushing, and empty-input paths.
    """

    @staticmethod
    def _wrap(values: list[Any]) -> pa.RecordBatch:
        """Build a one-batch pickle column from a list of Python values."""
        rows = [{PICKLE_COLUMN_NAME: dumps(v)} for v in values]
        return pa.RecordBatch.from_pylist(rows, schema=_ARROW_DYNAMIC_SCHEMA)

    @staticmethod
    def _unwrap(batches: Iterator[pa.RecordBatch]) -> list[Any]:
        """Materialise the pickled column from a stream of batches."""
        out: list[Any] = []
        for batch in batches:
            col = batch.column(0)
            for i in range(batch.num_rows):
                out.append(loads(col[i].as_py()))
        return out

    # --- inputs_map_partition ------------------------------------------
    def test_inputs_map_partition_applies_function_in_order(self) -> None:
        func_pickle = dumps(lambda x: x * 2)
        batch = self._wrap([1, 2, 3, 4])

        result = list(inputs_map_partition(func_pickle, iter([batch])))
        values = self._unwrap(iter(result))

        self.assertEqual(values, [2, 4, 6, 8])

    def test_inputs_map_partition_handles_multiple_batches(self) -> None:
        func_pickle = dumps(lambda x: f"v{x}")
        batches = [self._wrap([1, 2]), self._wrap([3]), self._wrap([4, 5])]

        result = list(inputs_map_partition(func_pickle, iter(batches)))

        self.assertEqual(self._unwrap(iter(result)), ["v1", "v2", "v3", "v4", "v5"])

    def test_inputs_map_partition_empty_input_yields_nothing(self) -> None:
        func_pickle = dumps(lambda x: x)
        result = list(inputs_map_partition(func_pickle, iter([])))
        self.assertEqual(result, [])

    def test_inputs_map_partition_flushes_on_byte_size(self) -> None:
        # byte_size=1 forces a flush before every row except the first, so
        # five inputs become five output batches. Anything larger and the
        # codec used by `dumps` could compress payloads enough to defeat
        # the threshold — keep it deterministic.
        func_pickle = dumps(lambda x: x)
        batch = self._wrap([1, 2, 3, 4, 5])

        result = list(
            inputs_map_partition(func_pickle, iter([batch]), byte_size=1)
        )

        self.assertEqual(len(result), 5)
        self.assertEqual(self._unwrap(iter(result)), [1, 2, 3, 4, 5])

    def test_inputs_map_partition_preserves_complex_values(self) -> None:
        func_pickle = dumps(lambda d: {"k": d["k"], "doubled": d["v"] * 2})
        batch = self._wrap([{"k": "a", "v": 1}, {"k": "b", "v": 5}])

        result = list(inputs_map_partition(func_pickle, iter([batch])))

        self.assertEqual(
            self._unwrap(iter(result)),
            [{"k": "a", "doubled": 2}, {"k": "b", "doubled": 10}],
        )

    # --- outputs_map_partition -----------------------------------------
    def test_outputs_map_partition_casts_pickled_dicts(self) -> None:
        sch = schema_builder([
            field_builder("id", arrow_type=pa.int64(), nullable=False),
            field_builder("name", arrow_type=pa.string()),
        ])
        # Each pickled row is a list of dicts — convert() takes that to a RecordBatch.
        batch = self._wrap([
            [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            [{"id": 3, "name": "c"}],
        ])

        out = list(outputs_map_partition(iter([batch]), schema=sch))
        merged = pa.Table.from_batches(out)

        self.assertEqual(merged.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(merged.column("name").to_pylist(), ["a", "b", "c"])

    def test_outputs_map_partition_skips_empty_results(self) -> None:
        sch = schema_builder([field_builder("id", arrow_type=pa.int64())])
        batch = self._wrap([[], [{"id": 7}], []])

        out = list(outputs_map_partition(iter([batch]), schema=sch))
        merged = pa.Table.from_batches(out)

        self.assertEqual(merged.column("id").to_pylist(), [7])

    def test_outputs_map_partition_empty_iterator(self) -> None:
        sch = schema_builder([field_builder("id", arrow_type=pa.int64())])
        out = list(outputs_map_partition(iter([]), schema=sch))
        self.assertEqual(out, [])


# ---------------------------------------------------------------------------
# DynamicFrame end-to-end — needs a SparkSession
# ---------------------------------------------------------------------------
class TestDynamicFrameParallelize(SparkTestCase):
    """``DynamicFrame.parallelize`` over a real local SparkSession."""

    def test_parallelize_applies_function_across_inputs(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x * x,
            inputs=[1, 2, 3, 4, 5],
            spark_session=self.spark,
        )

        # Schema is the canonical pickle schema regardless of payload.
        self.assertTrue(_spark_schema_is_dynamic(df.schema))
        self.assertEqual(sorted(df.collect()), [1, 4, 9, 16, 25])

    def test_parallelize_preserves_string_payloads(self) -> None:
        df = DynamicFrame.parallelize(
            function=str.upper,
            inputs=["abc", "def", "ghi"],
            spark_session=self.spark,
        )
        self.assertEqual(sorted(df.collect()), ["ABC", "DEF", "GHI"])

    def test_parallelize_handles_empty_inputs(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[],
            spark_session=self.spark,
        )
        self.assertEqual(df.collect(), [])
        self.assertEqual(df.df.count(), 0)

    def test_parallelize_returns_dynamic_frame_instance(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[1],
            spark_session=self.spark,
        )
        self.assertIsInstance(df, DynamicFrame)

    def test_parallelize_resolves_session_from_environment(self) -> None:
        # No spark_session passed — must reach for PyEnv, which finds the
        # global session set up by SparkTestCase.
        df = DynamicFrame.parallelize(
            function=lambda x: x + 1,
            inputs=[10, 20],
        )
        self.assertEqual(sorted(df.collect()), [11, 21])

    def test_parallelize_supports_dict_payloads(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda d: {"k": d["k"], "n": d["n"] + 1},
            inputs=[{"k": "a", "n": 1}, {"k": "b", "n": 2}],
            spark_session=self.spark,
        )
        result = sorted(df.collect(), key=lambda d: d["k"])
        self.assertEqual(result, [{"k": "a", "n": 2}, {"k": "b", "n": 3}])


class TestDynamicFrameMap(SparkTestCase):
    """Chained ``.map`` over an existing ``DynamicFrame``."""

    def test_map_applies_second_function(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x + 1,
            inputs=[1, 2, 3],
            spark_session=self.spark,
        ).map(lambda x: x * 10)

        self.assertEqual(sorted(df.collect()), [20, 30, 40])

    def test_map_returns_dynamic_frame(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[1],
            spark_session=self.spark,
        ).map(lambda x: x)

        self.assertIsInstance(df, DynamicFrame)
        self.assertTrue(_spark_schema_is_dynamic(df.schema))

    def test_map_identity_preserves_values(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[1, 2, 3],
            spark_session=self.spark,
        ).map(lambda x: x)

        self.assertEqual(sorted(df.collect()), [1, 2, 3])


class TestDynamicFrameExplode(SparkTestCase):
    """``.explode`` flattens an ``Iterable[T]`` payload into one row per element."""

    def test_explode_flattens_lists(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda n: list(range(n)),
            inputs=[1, 3, 0, 2],
            spark_session=self.spark,
        ).explode()

        # range(1)=[0]; range(3)=[0,1,2]; range(0)=[]; range(2)=[0,1]
        self.assertEqual(sorted(df.collect()), [0, 0, 0, 0, 1, 1, 2])

    def test_explode_empty_iterables_drop_rows(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: [],
            inputs=[1, 2, 3],
            spark_session=self.spark,
        ).explode()

        self.assertEqual(df.collect(), [])

    def test_explode_returns_dynamic_frame(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: [x],
            inputs=[1],
            spark_session=self.spark,
        ).explode()
        self.assertIsInstance(df, DynamicFrame)
        self.assertTrue(_spark_schema_is_dynamic(df.schema))


class TestDynamicFrameCollectAndIterator(SparkTestCase):
    """Materialisation paths: ``.collect`` and ``.to_local_iterator``."""

    def test_collect_returns_list(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[10, 20, 30],
            spark_session=self.spark,
        )
        out = df.collect()
        self.assertIsInstance(out, list)
        self.assertEqual(sorted(out), [10, 20, 30])

    def test_to_local_iterator_yields_each_value(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=["a", "b", "c"],
            spark_session=self.spark,
        )
        out = sorted(df.to_local_iterator())
        self.assertEqual(out, ["a", "b", "c"])

    def test_to_local_iterator_is_lazy(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[1, 2, 3],
            spark_session=self.spark,
        )
        it = df.to_local_iterator()
        # Hasn't been materialised into a list — pulling once advances.
        first = next(it)
        self.assertIn(first, {1, 2, 3})


class TestDynamicFrameCastAndArrow(SparkTestCase):
    """``.cast`` produces a typed Spark DataFrame; ``.toArrow`` lands a pa.Table."""

    def _people_frame(self) -> DynamicFrame:
        return DynamicFrame.parallelize(
            function=lambda d: [{"id": d["id"], "name": d["name"]}],
            inputs=[{"id": 1, "name": "ada"}, {"id": 2, "name": "alan"}],
            spark_session=self.spark,
        )

    def test_cast_returns_typed_spark_dataframe(self) -> None:
        sch = schema_builder([
            field_builder("id", arrow_type=pa.int64(), nullable=False),
            field_builder("name", arrow_type=pa.string()),
        ])

        df = self._people_frame().cast(sch)

        # The result is a Spark DataFrame, not a DynamicFrame.
        from pyspark.sql import DataFrame as SparkDF

        self.assertIsInstance(df, SparkDF)
        self.assertEqual(set(df.columns), {"id", "name"})

        rows = sorted((r["id"], r["name"]) for r in df.collect())
        self.assertEqual(rows, [(1, "ada"), (2, "alan")])

    def test_toArrow_returns_arrow_table(self) -> None:
        # DataFrame.toArrow() landed in Spark 4.0 — older runtimes skip cleanly
        # rather than failing with an AttributeError.
        from pyspark.sql import DataFrame as SparkDF

        if not hasattr(SparkDF, "toArrow"):
            self.skipTest("DataFrame.toArrow() requires pyspark>=4.0")

        sch = schema_builder([
            field_builder("id", arrow_type=pa.int64(), nullable=False),
            field_builder("name", arrow_type=pa.string()),
        ])

        table = self._people_frame().toArrow(schema=sch)

        self.assertIsInstance(table, pa.Table)
        self.assertEqual(set(table.column_names), {"id", "name"})

        records = sorted(
            zip(table.column("id").to_pylist(), table.column("name").to_pylist())
        )
        self.assertEqual(records, [(1, "ada"), (2, "alan")])


class TestDynamicFrameDataclass(SparkTestCase):
    """Direct construction and attribute surface of the frozen dataclass."""

    def test_construct_with_existing_dataframe(self) -> None:
        spark_df = self.spark.createDataFrame(
            [(dumps("hello"),), (dumps("world"),)],
            schema=_spark_dynamic_schema(),
        )
        df = DynamicFrame(df=spark_df)

        self.assertIs(df.df, spark_df)
        self.assertTrue(_spark_schema_is_dynamic(df.schema))
        self.assertEqual(sorted(df.collect()), ["hello", "world"])

    def test_dataclass_is_frozen(self) -> None:
        df = DynamicFrame.parallelize(
            function=lambda x: x,
            inputs=[1],
            spark_session=self.spark,
        )
        with self.assertRaises(Exception):
            df.df = None  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
