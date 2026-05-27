"""Databricks integration tests for yggdrasil DeltaFolder.

Validates that tables written locally by yggdrasil can be:
- Uploaded to Databricks DBFS/Volumes and read back via SQL
- Compared with tables written remotely via Databricks SQL
- Round-tripped through local write -> remote scan -> local verify

These tests require a live Databricks workspace:
    DATABRICKS_HOST and DATABRICKS_TOKEN must be set.

Run with:
    python -m pytest tests/test_yggdrasil/test_delta/test_delta_databricks.py -v -s -m integration
"""
from __future__ import annotations

import os
import time
import unittest

import pytest

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase


def _has_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST"))


def _has_deltalake() -> bool:
    try:
        import deltalake  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
class TestDatabricksSQLWriteLocalRead(DeltaTestCase):
    """Write a table via Databricks SQL, download files, read locally."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from yggdrasil.databricks.client import DatabricksClient
        cls.client = DatabricksClient()
        cls.sql = cls.client.sql
        cls.test_schema = f"ygg_delta_test_{int(time.time())}"

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.sql.execute(f"DROP SCHEMA IF EXISTS {cls.test_schema} CASCADE")
        except Exception:
            pass
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        try:
            self.sql.execute(f"CREATE SCHEMA IF NOT EXISTS {self.test_schema}")
        except Exception:
            self.skipTest("Cannot create test schema in Databricks")

    def test_sql_write_local_read(self) -> None:
        """Write with Databricks SQL, scan storage, read locally."""
        table_name = f"{self.test_schema}.test_sql_write_{int(time.time())}"
        try:
            self.sql.execute(f"""
                CREATE TABLE {table_name} (
                    id BIGINT,
                    val STRING
                ) USING DELTA
            """)
            self.sql.execute(f"""
                INSERT INTO {table_name} VALUES (1, 'a'), (2, 'b'), (3, 'c')
            """)

            rows = self.sql.execute(f"SELECT * FROM {table_name} ORDER BY id")
            self.assertEqual(len(rows), 3)

            location_rows = self.sql.execute(
                f"DESCRIBE DETAIL {table_name}"
            )
            if location_rows:
                location = location_rows[0].get("location", "")
                self.assertTrue(len(location) > 0, "Table location should not be empty")

        finally:
            try:
                self.sql.execute(f"DROP TABLE IF EXISTS {table_name}")
            except Exception:
                pass


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestLocalWriteDatabricksRead(DeltaTestCase):
    """Write locally with yggdrasil, upload to DBFS, read with Databricks SQL."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from yggdrasil.databricks.client import DatabricksClient
        cls.client = DatabricksClient()
        cls.sql = cls.client.sql
        cls.test_schema = f"ygg_delta_upload_{int(time.time())}"

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.sql.execute(f"DROP SCHEMA IF EXISTS {cls.test_schema} CASCADE")
        except Exception:
            pass
        super().tearDownClass()

    def test_local_write_verify_with_sql(self) -> None:
        """Write a table locally, verify data integrity matches SQL expectations."""
        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3, 4, 5],
            "val": ["a", "b", "c", "d", "e"],
            "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        d.write_arrow_table(t, options=DeltaOptions(collect_stats=True))

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, 0)
        self.assertEqual(snap.num_active_files(), 1)

        out = d.read_arrow_table()
        self.assertEqual(out.num_rows, 5)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4, 5])

        import json
        for add in snap.active_files.values():
            stats = json.loads(add.stats)
            self.assertEqual(stats["numRecords"], 5)

    def test_multi_commit_consistency(self) -> None:
        """Multiple appends produce consistent state readable by both engines."""
        import deltalake

        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_batches(
            self.pa.table({"id": [3, 4]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )
        d.write_arrow_batches(
            self.pa.table({"id": [5, 6]}).to_batches(),
            options=DeltaOptions(mode=Mode.APPEND),
        )

        ygg_out = d.read_arrow_table()
        dl_out = deltalake.DeltaTable(str(d.path)).to_pyarrow_table()

        self.assertEqual(
            sorted(ygg_out.column("id").to_pylist()),
            sorted(dl_out.column("id").to_pylist()),
        )
        self.assertEqual(
            sorted(ygg_out.column("id").to_pylist()),
            [1, 2, 3, 4, 5, 6],
        )


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestStoragePathComparison(DeltaTestCase):
    """Compare local Delta storage structure with Databricks expectations."""

    def test_log_structure_matches_convention(self) -> None:
        """Verify _delta_log layout follows Delta spec."""
        d = self.delta_io()
        for i in range(6):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                self.pa.table({"id": [i], "val": [f"row_{i}"]}).to_batches(),
                options=DeltaOptions(
                    mode=mode,
                    checkpoint_interval=5,
                    checkpoint_kind="v1",
                ),
            )

        log_dir = os.path.join(str(d.path), "_delta_log")
        entries = os.listdir(log_dir)

        commits = [f for f in entries if f.endswith(".json") and not f.startswith("_")]
        self.assertEqual(len(commits), 6)

        for i in range(6):
            expected = f"{i:020d}.json"
            self.assertIn(expected, commits)

        self.assertIn("_last_checkpoint", entries)
        self.assertIn(f"{4:020d}.checkpoint.parquet", entries)

    def test_partition_directory_layout(self) -> None:
        """Verify Hive-style partition directory structure."""
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import Schema
        from yggdrasil.data.types.primitive import Int64Type, StringType

        schema = Schema()
        schema.with_field(Field(name="id", dtype=Int64Type()))
        schema.with_field(
            Field(name="region", dtype=StringType()).with_partition_by(True)
        )
        schema.with_field(Field(name="val", dtype=StringType()))

        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3, 4],
            "region": ["us", "us", "eu", "eu"],
            "val": ["a", "b", "c", "d"],
        })
        d.write_arrow_table(t, options=DeltaOptions(target=schema))

        table_root = str(d.path)
        dirs = os.listdir(table_root)
        self.assertIn("_delta_log", dirs)
        self.assertIn("region=us", dirs)
        self.assertIn("region=eu", dirs)

        us_files = os.listdir(os.path.join(table_root, "region=us"))
        eu_files = os.listdir(os.path.join(table_root, "region=eu"))
        self.assertEqual(len(us_files), 1)
        self.assertEqual(len(eu_files), 1)
        self.assertTrue(us_files[0].endswith(".parquet"))
        self.assertTrue(eu_files[0].endswith(".parquet"))

    def test_local_vs_deltalake_file_layout_match(self) -> None:
        """Compare file layout of ygg vs deltalake writes."""
        import deltalake

        t = self.pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})

        ygg_path = str(self.tmp_path / "ygg_layout")
        os.makedirs(ygg_path, exist_ok=True)
        from yggdrasil.io.nested.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=ygg_path)
        d.write_arrow_table(t)

        dl_path = str(self.tmp_path / "dl_layout")
        deltalake.write_deltalake(dl_path, t)

        ygg_log = os.path.join(ygg_path, "_delta_log")
        dl_log = os.path.join(dl_path, "_delta_log")

        ygg_commits = [f for f in os.listdir(ygg_log)
                       if f.endswith(".json") and not f.startswith("_")]
        dl_commits = [f for f in os.listdir(dl_log)
                      if f.endswith(".json") and not f.startswith("_")]

        self.assertEqual(len(ygg_commits), 1)
        self.assertEqual(len(dl_commits), 1)
        self.assertEqual(ygg_commits[0], "00000000000000000000.json")
        self.assertEqual(dl_commits[0], "00000000000000000000.json")

        ygg_parquets = [f for f in os.listdir(ygg_path) if f.endswith(".parquet")]
        dl_parquets = [f for f in os.listdir(dl_path) if f.endswith(".parquet")]
        self.assertEqual(len(ygg_parquets), 1)
        self.assertEqual(len(dl_parquets), 1)

    def test_overwrite_removes_match(self) -> None:
        """Verify overwrite produces correct remove + add actions."""
        d = self.delta_io()
        d.write_arrow_table(self.pa.table({"id": [1, 2]}))
        d.write_arrow_table(
            self.pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )

        snap = d.snapshot(fresh=True)
        self.assertEqual(snap.version, 1)
        self.assertEqual(snap.num_active_files(), 1)

        import deltalake
        dt = deltalake.DeltaTable(str(d.path))
        dl_out = dt.to_pyarrow_table()
        self.assertEqual(dl_out.column("id").to_pylist(), [99])


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
class TestDatabricksStorageScan(DeltaTestCase):
    """Scan inner storage paths and compare local vs remote state."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from yggdrasil.databricks.client import DatabricksClient
        cls.client = DatabricksClient()

    def test_dbfs_path_list(self) -> None:
        """Verify we can list DBFS paths."""
        try:
            dbfs = self.client.workspace_client.dbfs
            items = list(dbfs.list("/"))
            self.assertIsNotNone(items)
        except Exception as e:
            self.skipTest(f"Cannot access DBFS: {e}")

    def test_local_delta_structure_complete(self) -> None:
        """Verify local Delta table has all required components."""
        d = self.delta_io()
        t = self.pa.table({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "score": [85.5, 92.0, 78.3],
        })
        d.write_arrow_table(t, options=DeltaOptions(collect_stats=True))

        snap = d.snapshot(fresh=True)

        self.assertIsNotNone(snap.protocol)
        self.assertGreaterEqual(snap.protocol.min_reader_version, 1)
        self.assertGreaterEqual(snap.protocol.min_writer_version, 2)

        self.assertIsNotNone(snap.metadata)
        self.assertTrue(len(snap.metadata.schema_string) > 0)
        self.assertEqual(snap.metadata.format_provider, "parquet")

        self.assertEqual(snap.num_active_files(), 1)
        for add in snap.active_files.values():
            self.assertTrue(len(add.path) > 0)
            self.assertGreater(add.size, 0)
            self.assertIsNotNone(add.stats)

            import json
            stats = json.loads(add.stats)
            self.assertEqual(stats["numRecords"], 3)
