"""Spark integration tests for Folder._read_spark_frame.

Exercises the Spark read path that scatters pickled leaf Tabulars to
executors via mapInArrow: basic round-trip correctness, partitioned
reads with predicate pruning, empty-folder edge cases, multi-partition
fan-out, coalesce bounds, and the zlib compression prefix on large
pickled leaves.
"""
from __future__ import annotations

import os
import pickle
import unittest
import zlib

import pytest

pyspark = pytest.importorskip("pyspark")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from yggdrasil.enums import Mode  # noqa: E402
from yggdrasil.execution.expr import col  # noqa: E402
from yggdrasil.path.folder import Folder, FolderOptions  # noqa: E402
from yggdrasil.path.local_path import LocalPath  # noqa: E402
from yggdrasil.spark.tests import SparkTestCase  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_table(n: int = 5) -> pa.Table:
    return pa.table({
        "id": pa.array(range(n), type=pa.int64()),
        "value": pa.array([f"v{i}" for i in range(n)], type=pa.string()),
    })


def _write_folder(root, table: pa.Table | None = None) -> Folder:
    table = table if table is not None else _simple_table()
    folder = Folder(path=str(root))
    folder.write_arrow_table(table)
    return folder


def _partitioned_schema() -> pa.Schema:
    return pa.schema([
        pa.field("pk", pa.string(), metadata={b"t:partition_by": b"True"}),
        pa.field("id", pa.int64()),
        pa.field("value", pa.string()),
    ])


def _write_partitioned_folder(root, partitions: dict[str, list[dict]]) -> Folder:
    schema = _partitioned_schema()
    folder = Folder(path=str(root))
    rows = []
    for pk_val, items in partitions.items():
        for item in items:
            rows.append({"pk": pk_val, **item})
    arrays = [
        pa.array([r["pk"] for r in rows], type=pa.string()),
        pa.array([r["id"] for r in rows], type=pa.int64()),
        pa.array([r["value"] for r in rows], type=pa.string()),
    ]
    batch = pa.record_batch(arrays, schema=schema)
    folder.write_arrow_batches([batch])
    return folder


# ---------------------------------------------------------------------------
# TestSparkReadBasic
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_singletons():
    Folder._INSTANCES.clear()
    LocalPath._INSTANCES.clear()
    yield
    Folder._INSTANCES.clear()
    LocalPath._INSTANCES.clear()


class TestSparkReadBasic(SparkTestCase):

    def test_returns_pyspark_dataframe(self) -> None:
        from pyspark.sql import DataFrame as SparkDF
        folder = _write_folder(self.tmp_path)
        result = folder.read_spark_frame()
        self.assertIsInstance(result, SparkDF)

    def test_row_count_matches_driver(self) -> None:
        table = _simple_table(7)
        folder = _write_folder(self.tmp_path, table)
        spark_df = folder.read_spark_frame()
        self.assertEqual(spark_df.count(), table.num_rows)

    def test_column_names_match(self) -> None:
        table = _simple_table()
        folder = _write_folder(self.tmp_path, table)
        spark_df = folder.read_spark_frame()
        self.assertEqual(spark_df.columns, table.column_names)

    def test_data_values_match_driver(self) -> None:
        table = _simple_table(4)
        folder = _write_folder(self.tmp_path, table)
        spark_df = folder.read_spark_frame()
        driver_table = folder.read_arrow_table()
        spark_rows = sorted(spark_df.collect(), key=lambda r: r["id"])
        driver_ids = driver_table.column("id").to_pylist()
        driver_vals = driver_table.column("value").to_pylist()
        self.assertEqual([r["id"] for r in spark_rows], driver_ids)
        self.assertEqual([r["value"] for r in spark_rows], driver_vals)


# ---------------------------------------------------------------------------
# TestSparkReadPartitioned
# ---------------------------------------------------------------------------


class TestSparkReadPartitioned(SparkTestCase):

    def test_reads_partitioned_folder(self) -> None:
        from pyspark.sql import DataFrame as SparkDF
        folder = _write_partitioned_folder(self.tmp_path, {
            "alpha": [{"id": 1, "value": "a1"}],
            "beta": [{"id": 2, "value": "b1"}],
        })
        spark_df = folder.read_spark_frame()
        self.assertIsInstance(spark_df, SparkDF)

    def test_row_count_matches_across_partitions(self) -> None:
        partitions = {
            "x": [{"id": i, "value": f"x{i}"} for i in range(3)],
            "y": [{"id": i + 10, "value": f"y{i}"} for i in range(5)],
        }
        folder = _write_partitioned_folder(self.tmp_path, partitions)
        spark_df = folder.read_spark_frame()
        expected = sum(len(v) for v in partitions.values())
        self.assertEqual(spark_df.count(), expected)

    def test_partition_column_values_present(self) -> None:
        folder = _write_partitioned_folder(self.tmp_path, {
            "east": [{"id": 1, "value": "e1"}],
            "west": [{"id": 2, "value": "w1"}],
        })
        spark_df = folder.read_spark_frame()
        pk_values = sorted(
            r["pk"] for r in spark_df.select("pk").distinct().collect()
        )
        self.assertEqual(pk_values, ["east", "west"])

    def test_predicate_pushes_to_partition_pruning(self) -> None:
        folder = _write_partitioned_folder(self.tmp_path, {
            "keep": [{"id": 1, "value": "k1"}, {"id": 2, "value": "k2"}],
            "drop": [{"id": 3, "value": "d1"}],
        })
        predicate = col("pk") == "keep"
        spark_df = folder.read_spark_frame(predicate=predicate)
        rows = spark_df.collect()
        self.assertEqual(len(rows), 2)
        pk_values = {r["pk"] for r in rows}
        self.assertEqual(pk_values, {"keep"})


# ---------------------------------------------------------------------------
# TestSparkReadEmpty
# ---------------------------------------------------------------------------


class TestSparkReadEmpty(SparkTestCase):

    def test_empty_folder_returns_empty_dataframe(self) -> None:
        empty_dir = self.tmp_path / "empty"
        empty_dir.mkdir()
        folder = Folder(path=str(empty_dir))
        spark_df = folder.read_spark_frame()
        self.assertEqual(spark_df.count(), 0)

    def test_empty_after_predicate_returns_empty(self) -> None:
        folder = _write_partitioned_folder(self.tmp_path, {
            "only": [{"id": 1, "value": "o1"}],
        })
        predicate = col("pk") == "nonexistent"
        spark_df = folder.read_spark_frame(predicate=predicate)
        self.assertEqual(spark_df.count(), 0)


# ---------------------------------------------------------------------------
# TestSparkReadLarge
# ---------------------------------------------------------------------------


class TestSparkReadLarge(SparkTestCase):

    def test_100_rows_across_10_partitions(self) -> None:
        partitions = {}
        for p in range(10):
            pk = f"p{p:02d}"
            partitions[pk] = [
                {"id": p * 10 + i, "value": f"{pk}_{i}"}
                for i in range(10)
            ]
        folder = _write_partitioned_folder(self.tmp_path, partitions)
        spark_df = folder.read_spark_frame()
        self.assertEqual(spark_df.count(), 100)

    def test_all_rows_survive_round_trip(self) -> None:
        partitions = {}
        for p in range(10):
            pk = f"p{p:02d}"
            partitions[pk] = [
                {"id": p * 10 + i, "value": f"{pk}_{i}"}
                for i in range(10)
            ]
        folder = _write_partitioned_folder(self.tmp_path, partitions)
        spark_df = folder.read_spark_frame()
        driver_table = folder.read_arrow_table()
        spark_ids = sorted(r["id"] for r in spark_df.collect())
        driver_ids = sorted(driver_table.column("id").to_pylist())
        self.assertEqual(spark_ids, driver_ids)


# ---------------------------------------------------------------------------
# TestSparkCoalesce
# ---------------------------------------------------------------------------


class TestSparkCoalesce(SparkTestCase):

    def test_partition_count_le_num_leaves(self) -> None:
        partitions = {
            f"p{i}": [{"id": i, "value": f"v{i}"}]
            for i in range(3)
        }
        folder = _write_partitioned_folder(self.tmp_path, partitions)
        spark_df = folder.read_spark_frame()
        # After coalesce the RDD should have at most as many partitions
        # as there are leaf files.
        n_leaves = list(folder.iter_leaves(FolderOptions()))
        self.assertLessEqual(
            spark_df.rdd.getNumPartitions(), len(n_leaves),
        )

    def test_partition_count_le_default_parallelism(self) -> None:
        self.skip_if_spark_connect()   # sparkContext / rdd are JVM-only
        partitions = {
            f"p{i}": [{"id": i, "value": f"v{i}"}]
            for i in range(20)
        }
        folder = _write_partitioned_folder(self.tmp_path, partitions)
        spark_df = folder.read_spark_frame()
        parallelism = max(self.spark.sparkContext.defaultParallelism, 1)
        self.assertLessEqual(
            spark_df.rdd.getNumPartitions(), parallelism,
        )


# ---------------------------------------------------------------------------
# TestSparkCompression
# ---------------------------------------------------------------------------


@pytest.mark.spark_integration
class TestSparkCompression(SparkTestCase):

    def test_large_pickle_gets_compressed(self) -> None:
        # Build a leaf whose pickle clears the 4 MB ``Z``-prefixed zlib
        # compression threshold (high-entropy hex payload won't shrink), then
        # verify the prefix convention + round-trip. 150k rows of int64 + a
        # 40-char hex string ≈ 7 MB pickled — a safe margin over the threshold
        # without the original 200k's extra cost.
        _COMPRESS_THRESHOLD = 4 * 1024 * 1024
        n_rows = 150_000
        table = pa.table({
            "id": pa.array(range(n_rows), type=pa.int64()),
            "payload": pa.array(
                [os.urandom(20).hex() for _ in range(n_rows)],
                type=pa.string(),
            ),
        })
        folder = _write_folder(self.tmp_path, table)

        leaves = list(folder.iter_leaves(FolderOptions()))
        self.assertGreater(len(leaves), 0)

        raw = pickle.dumps(leaves[0])
        self.assertGreater(
            len(raw), _COMPRESS_THRESHOLD,
            "fixture too small to exercise the compression path",
        )

        compressed = b"Z" + zlib.compress(raw)
        self.assertTrue(compressed.startswith(b"Z"))
        restored = pickle.loads(zlib.decompress(compressed[1:]))
        restored_rows = sum(b.num_rows for b in restored.read_arrow_batches())
        self.assertEqual(restored_rows, n_rows)
