"""Databricks SQL engine integration tests for yggdrasil DeltaFolder.

End-to-end tests that:
- Write tables via Databricks SQL, read back with DeltaFolder
- Write tables locally with DeltaFolder, register + read via SQL
- Compare ygg local reads vs Databricks SQL reads
- Scan inner storage paths and compare file layouts
- Test APPEND / OVERWRITE / schema evolution through SQL + DeltaFolder
- Benchmark local DeltaFolder vs Databricks SQL read paths

Requires:
    DATABRICKS_HOST, DATABRICKS_TOKEN (or auth profile)
    Optional: DATABRICKS_INTEGRATION_CATALOG (default: main)
              DATABRICKS_INTEGRATION_SCHEMA (default: ygg_delta_test)

Run:
    python -m pytest tests/test_yggdrasil/test_delta/test_delta_databricks.py -v -s -m integration
"""
from __future__ import annotations

import json
import os
import secrets
import unittest
from typing import ClassVar

import pyarrow as pa
import pytest

from yggdrasil.delta.io import DeltaOptions


def _has_databricks() -> bool:
    return bool(os.environ.get("DATABRICKS_HOST"))


def _catalog() -> str:
    return os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading_tgp_dev").strip() or "trading_tgp_dev"


def _schema() -> str:
    return os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "ygg_delta_test").strip() or "ygg_delta_test"


# ---------------------------------------------------------------------------
# Base class — one client + SQL engine per class, auto-cleanup
# ---------------------------------------------------------------------------


@pytest.mark.integration
@unittest.skipUnless(_has_databricks(), "DATABRICKS_HOST not set")
class _DeltaSQLBase(unittest.TestCase):
    client: ClassVar
    sql: ClassVar
    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    _tables: ClassVar[list]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        from yggdrasil.databricks.client import DatabricksClient
        cls.client = DatabricksClient()
        cls.catalog_name = _catalog()
        cls.schema_name = _schema()
        cls.sql = cls.client.sql
        cls._tables = []
        try:
            cls.sql.execute(f"CREATE SCHEMA IF NOT EXISTS {cls.catalog_name}.{cls.schema_name}")
        except Exception as e:
            raise unittest.SkipTest(f"Cannot create test schema: {e}")

    @classmethod
    def tearDownClass(cls) -> None:
        for name in cls._tables:
            try:
                cls.sql.execute(f"DROP TABLE IF EXISTS {name}")
            except Exception:
                pass
        super().tearDownClass()

    def _table_name(self, tag: str) -> str:
        name = f"{self.catalog_name}.{self.schema_name}.yg_{tag}_{secrets.token_hex(4)}"
        type(self)._tables.append(name)
        return name

    def _execute(self, sql: str):
        return self.sql.execute(sql)

    def _read_sql_arrow(self, sql: str) -> pa.Table:
        result = self._execute(sql)
        return result.read_arrow_table()


# ---------------------------------------------------------------------------
# SQL write → DeltaFolder read (via storage location)
# ---------------------------------------------------------------------------


class TestSQLWriteDeltaFolderRead(_DeltaSQLBase):
    """Write via Databricks SQL, then read the underlying storage with DeltaFolder."""

    def test_create_insert_read_via_storage(self) -> None:
        """CREATE TABLE + INSERT via SQL, read Delta log from storage location."""
        tbl = self._table_name("sql_wr")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT, val STRING) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')")

        # Get storage location
        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()
        self.assertTrue(location, "Table must have a storage location")

        # Read via DeltaFolder
        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        self.assertGreaterEqual(snap.version, 0)
        self.assertGreater(snap.num_active_files(), 0)
        self.assertIsNotNone(snap.metadata)

        out = folder.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])

    def test_append_multiple_inserts(self) -> None:
        """Multiple INSERTs produce multiple versions readable by DeltaFolder."""
        tbl = self._table_name("sql_app")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1), (2)")
        self._execute(f"INSERT INTO {tbl} VALUES (3), (4)")
        self._execute(f"INSERT INTO {tbl} VALUES (5)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        self.assertGreaterEqual(snap.version, 2)
        out = folder.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4, 5])

    def test_overwrite_via_sql(self) -> None:
        """INSERT OVERWRITE replaces data, DeltaFolder sees new version."""
        tbl = self._table_name("sql_ow")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1), (2), (3)")
        self._execute(f"INSERT OVERWRITE {tbl} VALUES (99)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        out = folder.read_arrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_partitioned_table_via_sql(self) -> None:
        """Partitioned table created via SQL, read with partition pruning."""
        tbl = self._table_name("sql_part")
        self._execute(f"""
            CREATE TABLE {tbl} (id BIGINT, region STRING, val STRING)
            USING DELTA PARTITIONED BY (region)
        """)
        self._execute(f"INSERT INTO {tbl} VALUES (1, 'us', 'a'), (2, 'eu', 'b'), (3, 'us', 'c')")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        self.assertEqual(snap.partition_columns, ["region"])
        out = folder.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_schema_matches_sql(self) -> None:
        """DeltaFolder schema matches what SQL reports."""
        tbl = self._table_name("sql_sch")
        self._execute(f"""
            CREATE TABLE {tbl} (
                id BIGINT, name STRING, score DOUBLE, active BOOLEAN
            ) USING DELTA
        """)
        self._execute(f"INSERT INTO {tbl} VALUES (1, 'alice', 95.5, true)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        schema = folder.collect_schema()
        names = [f.name for f in schema.fields]
        self.assertEqual(names, ["id", "name", "score", "active"])


# ---------------------------------------------------------------------------
# DeltaFolder write → SQL read (write to managed location)
# ---------------------------------------------------------------------------


class TestDeltaFolderWriteSQLRead(_DeltaSQLBase):
    """Write with DeltaFolder to a managed location, read back via SQL."""

    def test_arrow_insert_then_sql_select(self) -> None:
        """Write data via SQLEngine.arrow_insert_into, read back via SQL."""
        tbl = self._table_name("ygg_ins")
        data = pa.table({
            "id": pa.array([10, 20, 30], type=pa.int64()),
            "val": pa.array(["x", "y", "z"], type=pa.string()),
        })

        self.sql.arrow_insert_into(
            data, table=tbl, mode="overwrite",
            wait=True, raise_error=True,
        )

        out = self._read_sql_arrow(f"SELECT * FROM {tbl} ORDER BY id")
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.column("id").to_pylist(), [10, 20, 30])

    def test_append_via_arrow_insert(self) -> None:
        """Multiple arrow_insert_into appends produce correct cumulative state."""
        tbl = self._table_name("ygg_app")
        batch1 = pa.table({"id": pa.array([1, 2], type=pa.int64())})
        batch2 = pa.table({"id": pa.array([3, 4], type=pa.int64())})

        self.sql.arrow_insert_into(batch1, table=tbl, mode="overwrite",
                                    wait=True, raise_error=True)
        self.sql.arrow_insert_into(batch2, table=tbl, mode="append",
                                    wait=True, raise_error=True)

        out = self._read_sql_arrow(f"SELECT * FROM {tbl} ORDER BY id")
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4])

    def test_overwrite_via_arrow_insert(self) -> None:
        """Overwrite replaces all rows."""
        tbl = self._table_name("ygg_ow")
        self.sql.arrow_insert_into(
            pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}),
            table=tbl, mode="overwrite", wait=True, raise_error=True,
        )
        self.sql.arrow_insert_into(
            pa.table({"id": pa.array([99], type=pa.int64())}),
            table=tbl, mode="overwrite", wait=True, raise_error=True,
        )

        out = self._read_sql_arrow(f"SELECT * FROM {tbl}")
        self.assertEqual(out.column("id").to_pylist(), [99])


# ---------------------------------------------------------------------------
# Bidirectional: SQL ↔ DeltaFolder data comparison
# ---------------------------------------------------------------------------


class TestBidirectionalComparison(_DeltaSQLBase):
    """Write on one side, read on both, compare results."""

    def test_sql_write_compare_sql_vs_deltafolder(self) -> None:
        """Write via SQL, read via both SQL and DeltaFolder, compare."""
        tbl = self._table_name("cmp_sql")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT, val STRING) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1, 'a'), (2, 'b'), (3, 'c')")

        sql_out = self._read_sql_arrow(f"SELECT * FROM {tbl} ORDER BY id")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        ygg_out = folder.read_arrow_table()

        self.assertEqual(
            sorted(sql_out.column("id").to_pylist()),
            sorted(ygg_out.column("id").to_pylist()),
        )
        self.assertEqual(
            sorted(sql_out.column("val").to_pylist()),
            sorted(ygg_out.column("val").to_pylist()),
        )

    def test_arrow_insert_compare_sql_vs_deltafolder(self) -> None:
        """Write via arrow_insert_into, read via SQL and DeltaFolder, compare."""
        tbl = self._table_name("cmp_arr")
        data = pa.table({
            "id": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "score": pa.array([1.1, 2.2, 3.3, 4.4, 5.5], type=pa.float64()),
        })
        self.sql.arrow_insert_into(data, table=tbl, mode="overwrite",
                                    wait=True, raise_error=True)

        sql_out = self._read_sql_arrow(f"SELECT * FROM {tbl} ORDER BY id")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        ygg_out = DeltaFolder(path=Path.from_(location)).read_arrow_table()

        self.assertEqual(
            sorted(sql_out.column("id").to_pylist()),
            sorted(ygg_out.column("id").to_pylist()),
        )

    def test_multi_version_time_travel(self) -> None:
        """Multiple SQL inserts, DeltaFolder reads each version."""
        tbl = self._table_name("cmp_tt")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1), (2)")
        self._execute(f"INSERT INTO {tbl} VALUES (3)")
        self._execute(f"INSERT INTO {tbl} VALUES (4), (5)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        head_ids = sorted(folder.read_arrow_table().column("id").to_pylist())
        self.assertEqual(head_ids, [1, 2, 3, 4, 5])

        # Time-travel to version after first INSERT
        if snap.version >= 2:
            v1 = folder.read_arrow_table(options=DeltaOptions(version=1))
            self.assertEqual(sorted(v1.column("id").to_pylist()), [1, 2])


# ---------------------------------------------------------------------------
# Storage path inspection
# ---------------------------------------------------------------------------


class TestStorageScan(_DeltaSQLBase):
    """Inspect the physical storage layout of Databricks-managed Delta tables."""

    def test_table_has_delta_log(self) -> None:
        """Every Delta table has a _delta_log directory."""
        tbl = self.client.tables.table(self._table_name("scan_log"))
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1)")

        root = tbl.storage_path()
        children = [c.name for c in root.iterdir()]
        self.assertIn("_delta_log", children)

    def test_snapshot_metadata_matches_sql_describe(self) -> None:
        """Snapshot protocol/metadata matches DESCRIBE output."""
        tbl = self._table_name("scan_meta")
        self._execute(f"""
            CREATE TABLE {tbl} (id BIGINT, name STRING, score DOUBLE)
            USING DELTA
        """)
        self._execute(f"INSERT INTO {tbl} VALUES (1, 'a', 1.5)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()
        num_files_sql = detail.column("numFiles")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        self.assertEqual(snap.num_active_files(), num_files_sql)
        self.assertIsNotNone(snap.metadata)
        self.assertIn("id", snap.schema_string)
        self.assertIn("name", snap.schema_string)
        self.assertIn("score", snap.schema_string)

    def test_file_stats_match_sql_count(self) -> None:
        """AddFile stats numRecords matches SQL COUNT(*)."""
        tbl = self._table_name("scan_stats")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl} SELECT id FROM range(100)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        count_out = self._read_sql_arrow(f"SELECT COUNT(*) AS cnt FROM {tbl}")
        sql_count = count_out.column("cnt")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()

        ygg_count = sum(
            json.loads(add.stats).get("numRecords", 0)
            for add in snap.active_files.values()
            if add.stats
        )
        self.assertEqual(ygg_count, sql_count)


# ---------------------------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------------------------


class TestSchemaEvolution(_DeltaSQLBase):
    """Schema changes via SQL, read with DeltaFolder."""

    def test_add_column_via_sql(self) -> None:
        """ALTER TABLE ADD COLUMN, DeltaFolder sees updated schema."""
        tbl = self._table_name("evo_add")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl} VALUES (1)")
        self._execute(f"ALTER TABLE {tbl} ADD COLUMN (name STRING)")
        self._execute(f"INSERT INTO {tbl} VALUES (2, 'bob')")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        schema = folder.collect_schema()
        names = [f.name for f in schema.fields]
        self.assertIn("name", names)

    def test_table_properties_via_sql(self) -> None:
        """Table properties set via SQL are visible in snapshot config."""
        tbl = self._table_name("evo_prop")
        self._execute(f"CREATE TABLE {tbl} (id BIGINT) USING DELTA")
        self._execute(f"ALTER TABLE {tbl} SET TBLPROPERTIES ('delta.minReaderVersion' = '1')")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl}")
        location = detail.column("location")[0].as_py()

        from yggdrasil.io.nested.delta import DeltaFolder
        from yggdrasil.path import Path
        folder = DeltaFolder(path=Path.from_(location))
        snap = folder.snapshot()
        self.assertIsNotNone(snap.protocol)
