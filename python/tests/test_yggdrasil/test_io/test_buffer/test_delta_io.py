"""Unit tests for :class:`yggdrasil.io.buffer.delta_io.DeltaIO`.

Covers:

* :class:`DeltaOptions` validation
* factory / path coercion
* roundtrip via ``write_arrow_table`` / ``read_arrow_table``
* partition pruning (partitions=) and row-level filter pushdown
* column projection
* time travel by ``version``
* deletion-vector reads (Polars fallback path)
* SaveMode: ``APPEND``, ``OVERWRITE``, ``IGNORE``, ``ERROR_IF_EXISTS``
* schema collection
* ``count_rows`` on dataset + filtered paths
* batched iteration (``read_arrow_batches(batch_size=...)``)
* :meth:`DeltaIO.execute` SQL — GROUP BY, WHERE, projection, engine
  selection (polars / arrow / pandas), and a round-trip over a table
  with deletion vectors enabled

All tests require ``deltalake`` to be installed; the class is skipped
otherwise with an install hint. The two tests that exercise the
Polars SQL fallback path additionally require ``polars``.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

from yggdrasil.arrow.tests import ArrowTestCase


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_DELTALAKE = _has("deltalake")
HAS_POLARS = _has("polars")
HAS_PANDAS = _has("pandas")


@unittest.skipUnless(
    HAS_DELTALAKE,
    "'deltalake' is not installed. pip install deltalake",
)
class DeltaIOTestCase(ArrowTestCase):
    """Shared base: :class:`ArrowTestCase` + a fresh table directory per test.

    ``self.table_dir`` is a temp directory dedicated to the Delta table
    under test. Starts empty — ``DeltaIO.write_arrow_table`` will
    create the ``_delta_log/`` layout on first write.
    """

    dt_cls: ClassVar[Any]
    DeltaIO: ClassVar[Any]
    DeltaOptions: ClassVar[Any]
    SaveMode: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from deltalake import DeltaTable

        from yggdrasil.io.buffer.delta_io import DeltaIO, DeltaOptions
        from yggdrasil.io.enums import SaveMode

        cls.dt_cls = DeltaTable
        cls.DeltaIO = DeltaIO
        cls.DeltaOptions = DeltaOptions
        cls.SaveMode = SaveMode

    def setUp(self) -> None:
        super().setUp()
        self.table_dir = self.tmp_path / "delta"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_io(self, path: Path | None = None):
        return self.DeltaIO.make(path or self.table_dir)

    def sample_table(self):
        return self.table(
            {
                "id": list(range(5)),
                "name": ["alice", "bob", "carol", "dave", "eve"],
                "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

    def partitioned_table(self):
        return self.table(
            {
                "id": list(range(10)),
                "val": [f"v{i}" for i in range(10)],
                "part": ["a"] * 5 + ["b"] * 5,
            }
        )


# =====================================================================
# Options
# =====================================================================

class TestDeltaOptions(DeltaIOTestCase):
    def test_defaults(self):
        opt = self.DeltaOptions()
        self.assertIsNone(opt.version)
        self.assertIsNone(opt.timestamp)
        self.assertIsNone(opt.partitions)
        self.assertIsNone(opt.storage_options)
        self.assertFalse(opt.without_files)
        self.assertFalse(opt.as_large_types)
        # Inherited from PathOptions:
        self.assertTrue(opt.recursive)
        self.assertEqual(opt.partitioning, "hive")

    def test_version_and_timestamp_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.DeltaOptions(version=1, timestamp="2024-01-01T00:00:00Z")

    def test_version_must_be_non_negative_int(self):
        with self.assertRaisesRegex(ValueError, "version must be >= 0"):
            self.DeltaOptions(version=-1)
        with self.assertRaisesRegex(TypeError, "version must be int"):
            self.DeltaOptions(version="0")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "version must be int"):
            self.DeltaOptions(version=True)  # type: ignore[arg-type]

    def test_partitions_shape(self):
        with self.assertRaisesRegex(TypeError, "partitions must be"):
            self.DeltaOptions(partitions=[("col", "bad")])  # type: ignore[list-item]
        with self.assertRaisesRegex(TypeError, "partitions must be"):
            self.DeltaOptions(partitions=[(1, "=", "x")])  # type: ignore[list-item]
        ok = self.DeltaOptions(partitions=[("part", "=", "a")])
        self.assertEqual(ok.partitions, (("part", "=", "a"),))

    def test_schema_mode_values(self):
        with self.assertRaisesRegex(ValueError, "schema_mode"):
            self.DeltaOptions(schema_mode="weird")
        self.DeltaOptions(schema_mode="merge")
        self.DeltaOptions(schema_mode="overwrite")

    def test_target_file_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "target_file_size"):
            self.DeltaOptions(target_file_size=0)
        with self.assertRaisesRegex(ValueError, "target_file_size"):
            self.DeltaOptions(target_file_size=-100)
        self.DeltaOptions(target_file_size=1024)


# =====================================================================
# Factory & path coercion
# =====================================================================

class TestMake(DeltaIOTestCase):
    def test_coerces_string_to_path(self):
        io = self.DeltaIO.make(str(self.table_dir))
        self.assertIsInstance(io.path, Path)
        self.assertEqual(io.path, self.table_dir)

    def test_defaults_media_to_delta(self):
        from yggdrasil.io.enums import MimeTypes

        io = self.DeltaIO.make(self.table_dir)
        self.assertIs(io.media_type.mime_type, MimeTypes.DELTA)

    def test_table_exists_false_before_write(self):
        io = self.make_io()
        self.assertFalse(io.table_exists)

    def test_is_delta_table_probe_on_non_table(self):
        self.assertFalse(self.DeltaIO.is_delta_table(self.tmp_path))


# =====================================================================
# Roundtrip
# =====================================================================

class TestRoundtrip(DeltaIOTestCase):
    def test_write_then_read(self):
        io = self.make_io()
        expected = self.sample_table()
        io.write_arrow_table(expected, mode=self.SaveMode.OVERWRITE)
        self.assertTrue(io.table_exists)

        actual = io.read_arrow_table()
        self.assertFrameEqual(
            actual.sort_by("id"),
            expected,
            check_schema=False,  # delta-rs may promote nullability
        )

    def test_count_rows(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        self.assertEqual(io.count_rows(), 5)

    def test_schema_collection(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        schema = io._collect_arrow_schema()
        self.assertEqual(
            sorted(schema.names), ["id", "name", "score"]
        )

    def test_iter_files_yields_active_parquet(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        files = list(io.iter_files())
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].path.exists())
        self.assertTrue(files[0].path.suffix.endswith("parquet"))

    def test_batched_read(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": list(range(50))}),
            mode=self.SaveMode.OVERWRITE,
        )
        batches = list(io.read_arrow_batches(batch_size=10))
        self.assertEqual(sum(b.num_rows for b in batches), 50)
        # At least 2 batches (deltalake chunking isn't strictly
        # bounded but should honor our requested size).
        self.assertGreaterEqual(len(batches), 2)


# =====================================================================
# Column projection / row filter / partition pruning
# =====================================================================

class TestPushdown(DeltaIOTestCase):
    def test_column_projection(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        projected = io.read_arrow_table(columns=["id", "name"])
        self.assertEqual(projected.column_names, ["id", "name"])

    def test_row_filter_tuple(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        filtered = io.read_arrow_table(filter=("id", ">", 2))
        ids = sorted(filtered.column("id").to_pylist())
        self.assertEqual(ids, [3, 4])

    def test_partition_pruning(self):
        io = self.make_io()
        io.write_arrow_table(
            self.partitioned_table(),
            mode=self.SaveMode.OVERWRITE,
            partition_by="part",
        )
        only_a = io.read_arrow_table(partitions=[("part", "=", "a")])
        self.assertEqual(only_a.num_rows, 5)
        self.assertEqual(set(only_a.column("part").to_pylist()), {"a"})

    def test_count_rows_with_filter(self):
        io = self.make_io()
        io.write_arrow_table(self.sample_table(), mode=self.SaveMode.OVERWRITE)
        self.assertEqual(io.count_rows(filter=("id", ">", 2)), 2)


# =====================================================================
# Time travel
# =====================================================================

class TestTimeTravel(DeltaIOTestCase):
    def test_version_rollback(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"x": [1, 2, 3]}), mode=self.SaveMode.OVERWRITE
        )
        io.write_arrow_table(
            self.table({"x": [4, 5]}), mode=self.SaveMode.APPEND
        )
        self.assertEqual(io.version(), 1)

        latest = sorted(io.read_arrow_table().column("x").to_pylist())
        self.assertEqual(latest, [1, 2, 3, 4, 5])

        snapshot_v0 = sorted(
            io.read_arrow_table(version=0).column("x").to_pylist()
        )
        self.assertEqual(snapshot_v0, [1, 2, 3])


# =====================================================================
# Deletion vectors (exercises the Polars fallback path)
# =====================================================================

@unittest.skipUnless(HAS_POLARS, "polars required for DV fallback path")
class TestDeletionVectors(DeltaIOTestCase):
    def test_read_after_delete_with_dvs(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": list(range(100))}),
            mode=self.SaveMode.OVERWRITE,
            configuration={"delta.enableDeletionVectors": "true"},
        )
        # Use delta-rs directly to trigger a real delete commit.
        dt = self.dt_cls(str(io.path))
        dt.delete("id < 30")

        # The table's protocol now declares deletionVectors as a
        # reader feature; delta-rs refuses to_pyarrow_dataset on it,
        # so DeltaIO must dispatch through the Polars fallback.
        result = io.read_arrow_table()
        ids = sorted(result.column("id").to_pylist())
        self.assertEqual(len(ids), 70)
        self.assertEqual(min(ids), 30)

    def test_batched_read_with_dvs(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": list(range(80))}),
            mode=self.SaveMode.OVERWRITE,
            configuration={"delta.enableDeletionVectors": "true"},
        )
        dt = self.dt_cls(str(io.path))
        dt.delete("id % 4 = 0")

        batches = list(io.read_arrow_batches(batch_size=10))
        self.assertEqual(sum(b.num_rows for b in batches), 60)


# =====================================================================
# Save modes
# =====================================================================

class TestSaveModes(DeltaIOTestCase):
    def test_append(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": [1, 2]}), mode=self.SaveMode.OVERWRITE
        )
        io.write_arrow_table(
            self.table({"id": [3, 4]}), mode=self.SaveMode.APPEND
        )
        self.assertEqual(
            sorted(io.read_arrow_table().column("id").to_pylist()),
            [1, 2, 3, 4],
        )

    def test_overwrite_replaces_all_rows(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": [1, 2, 3]}), mode=self.SaveMode.OVERWRITE
        )
        io.write_arrow_table(
            self.table({"id": [99]}), mode=self.SaveMode.OVERWRITE
        )
        self.assertEqual(
            io.read_arrow_table().column("id").to_pylist(), [99]
        )

    def test_error_if_exists_raises(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": [1]}), mode=self.SaveMode.OVERWRITE
        )
        with self.assertRaises(Exception):
            io.write_arrow_table(
                self.table({"id": [2]}),
                mode=self.SaveMode.ERROR_IF_EXISTS,
            )

    def test_ignore_is_noop_when_table_exists(self):
        io = self.make_io()
        io.write_arrow_table(
            self.table({"id": [1, 2]}), mode=self.SaveMode.OVERWRITE
        )
        io.write_arrow_table(
            self.table({"id": [999]}), mode=self.SaveMode.IGNORE
        )
        self.assertEqual(
            sorted(io.read_arrow_table().column("id").to_pylist()),
            [1, 2],
        )

    def test_upsert_raises(self):
        io = self.make_io()
        with self.assertRaisesRegex(NotImplementedError, "UPSERT"):
            io.write_arrow_table(
                self.table({"id": [1]}), mode=self.SaveMode.UPSERT
            )


# =====================================================================
# execute — the ask-for-tests target
# =====================================================================

@unittest.skipUnless(HAS_POLARS, "polars required for MediaIO.execute")
class TestExecute(DeltaIOTestCase):
    def _seed(self, io, *, with_dvs: bool = False) -> None:
        config = (
            {"delta.enableDeletionVectors": "true"} if with_dvs else None
        )
        io.write_arrow_table(
            self.table(
                {
                    "id": list(range(10)),
                    "name": [f"u{i}" for i in range(10)],
                    "score": [float(i) for i in range(10)],
                    "part": ["a"] * 5 + ["b"] * 5,
                }
            ),
            mode=self.SaveMode.OVERWRITE,
            configuration=config,
        )

    def test_execute_returns_polars_by_default(self):
        import polars as pl

        io = self.make_io()
        self._seed(io)

        result = io.execute("SELECT id, name FROM self WHERE id > 6 ORDER BY id")
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result.columns, ["id", "name"])
        self.assertEqual(result["id"].to_list(), [7, 8, 9])

    def test_execute_arrow_engine(self):
        import pyarrow as pa

        io = self.make_io()
        self._seed(io)
        result = io.execute(
            "SELECT COUNT(*) AS n FROM self",
            engine="arrow",
        )
        self.assertIsInstance(result, pa.Table)
        self.assertEqual(result.column("n").to_pylist(), [10])

    @unittest.skipUnless(HAS_PANDAS, "pandas required for pandas engine")
    def test_execute_pandas_engine(self):
        import pandas as pd

        io = self.make_io()
        self._seed(io)
        result = io.execute(
            "SELECT COUNT(*) AS n FROM self",
            engine="pandas",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(int(result["n"].iloc[0]), 10)

    def test_execute_aggregation_and_group_by(self):
        io = self.make_io()
        self._seed(io)

        result = io.execute(
            "SELECT part, COUNT(*) AS n, SUM(score) AS total "
            "FROM self GROUP BY part ORDER BY part"
        )
        self.assertEqual(result["part"].to_list(), ["a", "b"])
        self.assertEqual(result["n"].to_list(), [5, 5])
        self.assertEqual(result["total"].to_list(), [10.0, 35.0])

    def test_execute_projection_and_filter(self):
        io = self.make_io()
        self._seed(io)

        result = io.execute(
            "SELECT name FROM self WHERE part = 'a' ORDER BY id"
        )
        self.assertEqual(
            result["name"].to_list(),
            ["u0", "u1", "u2", "u3", "u4"],
        )

    def test_execute_empty_statement_raises(self):
        io = self.make_io()
        self._seed(io)
        with self.assertRaisesRegex(ValueError, "non-empty SQL statement"):
            io.execute("")

    def test_execute_bad_engine_raises(self):
        io = self.make_io()
        self._seed(io)
        with self.assertRaisesRegex(ValueError, "engine must be one of"):
            io.execute("SELECT 1", engine="duckdb")

    def test_execute_on_table_with_deletion_vectors(self):
        """Full SQL round-trip on a table that requires the DV fallback.

        After a delete that creates a deletion-vector-capable protocol,
        the Arrow dataset path raises — ``execute`` must still return
        the correct post-delete rowcount via the Polars scan path.
        """
        io = self.make_io()
        self._seed(io, with_dvs=True)

        dt = self.dt_cls(str(io.path))
        dt.delete("id < 4")

        result = io.execute(
            "SELECT COUNT(*) AS n FROM self",
            engine="arrow",
        )
        self.assertEqual(result.column("n").to_pylist(), [6])

        grouped = io.execute(
            "SELECT part, COUNT(*) AS n FROM self GROUP BY part ORDER BY part"
        )
        self.assertEqual(grouped["part"].to_list(), ["a", "b"])
        self.assertEqual(grouped["n"].to_list(), [1, 5])

    def test_execute_uses_custom_name(self):
        io = self.make_io()
        self._seed(io)
        result = io.execute(
            "SELECT COUNT(*) AS n FROM t",
            name="t",
        )
        self.assertEqual(result["n"].to_list(), [10])


if __name__ == "__main__":
    unittest.main()
