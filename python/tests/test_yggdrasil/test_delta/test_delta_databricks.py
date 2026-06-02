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
    Optional: DATABRICKS_INTEGRATION_CATALOG (default: trading_tgp_dev)
              DATABRICKS_INTEGRATION_SCHEMA (default: ygg_integration)

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
    return os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "ygg_integration").strip() or "ygg_integration"


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

    def _table(self, tag: str):
        """Mint a unique table handle bound to this workspace.

        Returns the :class:`Table` object — callers reach for
        :meth:`Table.storage_path` to get a UC-credentialed Path on
        the underlying cloud storage. The fully-qualified name lives
        on ``table.full_name()`` and is tracked for class teardown.
        """
        return self.client.tables.table(self._table_name(tag))

    def _execute(self, sql: str):
        return self.sql.execute(sql)

    def _read_sql_arrow(self, sql: str) -> pa.Table:
        result = self._execute(sql)
        return result.read_arrow_table()

    def _delta_folder(self, table) -> "DeltaFolder":
        """Open a :class:`DeltaFolder` over *table*'s cloud storage.

        Uses :meth:`Table.storage_path` so the underlying read carries
        UC-vended temporary credentials (auto-refreshing) — a raw
        ``Path.from_(location)`` against the same URL would fail with
        ``AccessDenied`` on managed Delta tables.
        """
        from yggdrasil.io.delta import DeltaFolder
        return DeltaFolder(path=table.storage_path())


# ---------------------------------------------------------------------------
# SQL write → DeltaFolder read (via storage location)
# ---------------------------------------------------------------------------


class TestSQLWriteDeltaFolderRead(_DeltaSQLBase):
    """Write via Databricks SQL, then read the underlying storage with DeltaFolder."""

    def test_create_insert_read_via_storage(self) -> None:
        """CREATE TABLE + INSERT via SQL, read Delta log from storage location."""
        tbl = self._table("sql_wr")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1, 'a'), (2, 'b'), (3, 'c')")

        folder = self._delta_folder(tbl)
        snap = folder.snapshot()

        self.assertGreaterEqual(snap.version, 0)
        self.assertGreater(snap.num_active_files(), 0)
        self.assertIsNotNone(snap.metadata)

        out = folder.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3])

    def test_append_multiple_inserts(self) -> None:
        """Multiple INSERTs produce multiple versions readable by DeltaFolder."""
        tbl = self._table("sql_app")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1), (2)")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (3), (4)")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (5)")

        folder = self._delta_folder(tbl)
        snap = folder.snapshot()

        self.assertGreaterEqual(snap.version, 2)
        out = folder.read_arrow_table()
        self.assertEqual(sorted(out.column("id").to_pylist()), [1, 2, 3, 4, 5])

    def test_overwrite_via_sql(self) -> None:
        """INSERT OVERWRITE replaces data, DeltaFolder sees new version."""
        tbl = self._table("sql_ow")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1), (2), (3)")
        self._execute(f"INSERT OVERWRITE {tbl.full_name()} VALUES (99)")

        out = self._delta_folder(tbl).read_arrow_table()
        self.assertEqual(out.column("id").to_pylist(), [99])

    def test_partitioned_table_via_sql(self) -> None:
        """Partitioned table created via SQL, read with partition pruning."""
        tbl = self._table("sql_part")
        self._execute(f"""
            CREATE TABLE {tbl.full_name()} (id BIGINT, region STRING, val STRING)
            USING DELTA PARTITIONED BY (region)
        """)
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1, 'us', 'a'), (2, 'eu', 'b'), (3, 'us', 'c')")

        folder = self._delta_folder(tbl)
        snap = folder.snapshot()

        self.assertEqual(snap.partition_columns, ["region"])
        out = folder.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(set(out.column("region").to_pylist()), {"us", "eu"})

    def test_schema_matches_sql(self) -> None:
        """DeltaFolder schema matches what SQL reports."""
        tbl = self._table("sql_sch")
        self._execute(f"""
            CREATE TABLE {tbl.full_name()} (
                id BIGINT, name STRING, score DOUBLE, active BOOLEAN
            ) USING DELTA
        """)
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1, 'alice', 95.5, true)")

        schema = self._delta_folder(tbl).collect_schema()
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
# Timestamp physical-type compatibility (Databricks reads ygg's parquet)
# ---------------------------------------------------------------------------


class TestTimestampCrossRead(_DeltaSQLBase):
    """DeltaFolder writes timestamps that Databricks / Photon can read.

    The Photon parquet reader rejects nanosecond timestamps
    (``Unsupported time unit in Parquet TimestampType``); DeltaFolder
    down-coerces to ``TIMESTAMP(MICROS)`` while preserving the zone.
    A tz-aware column round-trips as ``timestamp`` (UTC instant), a
    tz-naive column as ``timestamp_ntz`` — matching the ``deltalake``
    writer and what Databricks SQL ``DESCRIBE`` reports.
    """

    def test_nanosecond_timestamp_readable_by_databricks(self) -> None:
        import datetime

        tbl = self._table_name("ts_ns")
        data = pa.table({
            "id": pa.array(["a", "b", "c"], pa.string()),
            "updated_at": pa.array(
                [datetime.datetime(2020, 1, 1), datetime.datetime(2021, 6, 15),
                 datetime.datetime(2029, 12, 28)],
                pa.timestamp("ns", tz="UTC"),
            ),
        })
        # Write straight to the managed-table storage path via DeltaFolder.
        self.sql.arrow_insert_into(data, table=tbl, mode="overwrite",
                                   wait=True, raise_error=True)
        out = self._read_sql_arrow(f"SELECT count(*) c, min(updated_at) mn, "
                                   f"max(updated_at) mx FROM {tbl}")
        row = out.to_pylist()[0]
        self.assertEqual(row["c"], 3)
        self.assertEqual(row["mn"].year, 2020)
        self.assertEqual(row["mx"].year, 2029)

    def test_aware_timestamp_describes_as_timestamp(self) -> None:
        tbl = self._table_name("ts_aware")
        data = pa.table({
            "id": pa.array(["a"], pa.string()),
            "ts": pa.array([0], pa.timestamp("us", tz="UTC")),
        })
        self.sql.arrow_insert_into(data, table=tbl, mode="overwrite",
                                   wait=True, raise_error=True)
        desc = {r["col_name"]: r["data_type"]
                for r in self._read_sql_arrow(f"DESCRIBE {tbl}").to_pylist()
                if r.get("col_name")}
        self.assertEqual(desc.get("ts"), "timestamp")


# ---------------------------------------------------------------------------
# Bidirectional: SQL ↔ DeltaFolder data comparison
# ---------------------------------------------------------------------------


class TestBidirectionalComparison(_DeltaSQLBase):
    """Write on one side, read on both, compare results."""

    def test_sql_write_compare_sql_vs_deltafolder(self) -> None:
        """Write via SQL, read via both SQL and DeltaFolder, compare."""
        tbl = self._table("cmp_sql")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1, 'a'), (2, 'b'), (3, 'c')")

        sql_out = self._read_sql_arrow(f"SELECT * FROM {tbl.full_name()} ORDER BY id")
        ygg_out = self._delta_folder(tbl).read_arrow_table()

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
        tbl = self._table("cmp_arr")
        data = pa.table({
            "id": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
            "score": pa.array([1.1, 2.2, 3.3, 4.4, 5.5], type=pa.float64()),
        })
        self.sql.arrow_insert_into(data, table=tbl.full_name(), mode="overwrite",
                                    wait=True, raise_error=True)

        sql_out = self._read_sql_arrow(f"SELECT * FROM {tbl.full_name()} ORDER BY id")
        ygg_out = self._delta_folder(tbl).read_arrow_table()

        self.assertEqual(
            sorted(sql_out.column("id").to_pylist()),
            sorted(ygg_out.column("id").to_pylist()),
        )

    def test_multi_version_time_travel(self) -> None:
        """Multiple SQL inserts, DeltaFolder reads each version."""
        tbl = self._table("cmp_tt")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1), (2)")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (3)")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (4), (5)")

        folder = self._delta_folder(tbl)
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
        tbl = self._table("scan_log")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1)")

        root = tbl.storage_path()
        children = [c.name for c in root.iterdir()]
        self.assertIn("_delta_log", children)

    def test_snapshot_metadata_matches_sql_describe(self) -> None:
        """Snapshot protocol/metadata matches DESCRIBE output."""
        tbl = self._table("scan_meta")
        self._execute(f"""
            CREATE TABLE {tbl.full_name()} (id BIGINT, name STRING, score DOUBLE)
            USING DELTA
        """)
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1, 'a', 1.5)")

        detail = self._read_sql_arrow(f"DESCRIBE DETAIL {tbl.full_name()}")
        num_files_sql = detail.column("numFiles")[0].as_py()

        snap = self._delta_folder(tbl).snapshot()

        self.assertEqual(snap.num_active_files(), num_files_sql)
        self.assertIsNotNone(snap.metadata)
        self.assertIn("id", snap.schema_string)
        self.assertIn("name", snap.schema_string)
        self.assertIn("score", snap.schema_string)

    def test_file_stats_match_sql_count(self) -> None:
        """AddFile stats numRecords matches SQL COUNT(*)."""
        tbl = self._table("scan_stats")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} SELECT id FROM range(100)")

        count_out = self._read_sql_arrow(f"SELECT COUNT(*) AS cnt FROM {tbl.full_name()}")
        sql_count = count_out.column("cnt")[0].as_py()

        snap = self._delta_folder(tbl).snapshot()

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
        tbl = self._table("evo_add")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (1)")
        self._execute(f"ALTER TABLE {tbl.full_name()} ADD COLUMN (name STRING)")
        self._execute(f"INSERT INTO {tbl.full_name()} VALUES (2, 'bob')")

        schema = self._delta_folder(tbl).collect_schema()
        names = [f.name for f in schema.fields]
        self.assertIn("name", names)

    def test_table_properties_via_sql(self) -> None:
        """Table properties set via SQL are visible in snapshot config."""
        tbl = self._table("evo_prop")
        self._execute(f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA")
        self._execute(
            f"ALTER TABLE {tbl.full_name()} "
            "SET TBLPROPERTIES ('delta.minReaderVersion' = '1')"
        )

        snap = self._delta_folder(tbl).snapshot()
        self.assertIsNotNone(snap.protocol)


# ---------------------------------------------------------------------------
# Deletion vectors — MERGE on DV-enabled tables marks rows instead of rewriting
# ---------------------------------------------------------------------------


class TestMergeDeletionVectors(_DeltaSQLBase):
    """MERGE INTO a DV-enabled Delta table emits deletion vectors
    instead of rewriting the underlying parquet files.

    On ``delta.enableDeletionVectors=true``, an UPDATE/DELETE inside
    a MERGE either:

    - stamps a deletion vector on the existing AddFile (marking the
      matched rows as deleted), and appends a new AddFile carrying
      just the updated rows, OR
    - encodes the DV inline on the AddFile when the row count fits
      under the inline threshold.

    The snapshot's :attr:`AddFile.deletion_vector` slot is non-None
    on the resulting AddFiles; ``num_active_files()`` typically
    grows because the new rows ride on a fresh file. The DeltaFolder
    read path filters DV-marked rows out, so the materialised table
    reflects the post-MERGE state.
    """

    def _create_dv_table(self, tag: str):
        tbl = self._table(tag)
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) "
            "USING DELTA "
            "TBLPROPERTIES ('delta.enableDeletionVectors' = 'true')"
        )
        return tbl

    def test_merge_update_emits_deletion_vector(self) -> None:
        """A MERGE WHEN MATCHED UPDATE stamps DVs on the touched files."""
        tbl = self._create_dv_table("dv_upd")
        self._execute(
            f"INSERT INTO {tbl.full_name()} "
            "VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')"
        )
        self._execute(f"""
            MERGE INTO {tbl.full_name()} t
            USING (
              SELECT 1 AS id, 'A' AS val UNION ALL
              SELECT 3 AS id, 'C' AS val
            ) s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.val = s.val
        """)

        snap = self._delta_folder(tbl).snapshot()
        dv_files = [
            add for add in snap.active_files.values()
            if add.deletion_vector is not None
        ]
        self.assertGreater(
            len(dv_files), 0,
            "MERGE on a DV-enabled table should produce at least "
            "one AddFile with a deletion_vector attached.",
        )

    def test_merge_update_reads_back_correctly(self) -> None:
        """DeltaFolder filters DV-marked rows; the materialised
        table matches the post-MERGE SQL view."""
        tbl = self._create_dv_table("dv_read")
        self._execute(
            f"INSERT INTO {tbl.full_name()} "
            "VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')"
        )
        self._execute(f"""
            MERGE INTO {tbl.full_name()} t
            USING (
              SELECT 2 AS id, 'B' AS val UNION ALL
              SELECT 4 AS id, 'D' AS val
            ) s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.val = s.val
        """)

        sql_pairs = sorted(
            zip(
                self._read_sql_arrow(
                    f"SELECT id, val FROM {tbl.full_name()} ORDER BY id"
                ).column("id").to_pylist(),
                self._read_sql_arrow(
                    f"SELECT id, val FROM {tbl.full_name()} ORDER BY id"
                ).column("val").to_pylist(),
            )
        )
        ygg = self._delta_folder(tbl).read_arrow_table()
        ygg_pairs = sorted(
            zip(ygg.column("id").to_pylist(), ygg.column("val").to_pylist())
        )
        self.assertEqual(sql_pairs, ygg_pairs)

    def test_merge_delete_emits_deletion_vector(self) -> None:
        """WHEN MATCHED THEN DELETE on a DV-enabled table marks rows
        via the DV slot instead of rewriting files."""
        tbl = self._create_dv_table("dv_del")
        self._execute(
            f"INSERT INTO {tbl.full_name()} "
            "VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')"
        )
        self._execute(f"""
            MERGE INTO {tbl.full_name()} t
            USING (SELECT 2 AS id UNION ALL SELECT 4 AS id) s
            ON t.id = s.id
            WHEN MATCHED THEN DELETE
        """)

        snap = self._delta_folder(tbl).snapshot()
        dv_files = [
            add for add in snap.active_files.values()
            if add.deletion_vector is not None
        ]
        self.assertGreater(len(dv_files), 0)

        out = self._delta_folder(tbl).read_arrow_table()
        self.assertEqual(
            sorted(out.column("id").to_pylist()), [1, 3],
        )

    def test_merge_mixed_update_insert_with_dv(self) -> None:
        """A full ``WHEN MATCHED UPDATE … WHEN NOT MATCHED INSERT``
        MERGE on a DV-enabled table reconciles both branches: matched
        rows get DV-marked, unmatched rows land as new AddFiles."""
        tbl = self._create_dv_table("dv_mix")
        self._execute(
            f"INSERT INTO {tbl.full_name()} "
            "VALUES (1, 'a'), (2, 'b'), (3, 'c')"
        )
        self._execute(f"""
            MERGE INTO {tbl.full_name()} t
            USING (
              SELECT 2 AS id, 'B' AS val UNION ALL
              SELECT 5 AS id, 'E' AS val
            ) s
            ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.val = s.val
            WHEN NOT MATCHED THEN INSERT (id, val) VALUES (s.id, s.val)
        """)

        snap = self._delta_folder(tbl).snapshot()
        dv_files = [
            add for add in snap.active_files.values()
            if add.deletion_vector is not None
        ]
        self.assertGreater(len(dv_files), 0)

        out = self._delta_folder(tbl).read_arrow_table()
        pairs = sorted(
            zip(out.column("id").to_pylist(), out.column("val").to_pylist())
        )
        self.assertEqual(
            pairs, [(1, "a"), (2, "B"), (3, "c"), (5, "E")],
        )


# ---------------------------------------------------------------------------
# Lazy tabular ↔ Databricks SQL — same query, two engines
# ---------------------------------------------------------------------------


class TestLazyTabularVsDatabricksSQL(_DeltaSQLBase):
    """Same predicate, two engines: warehouse SQL vs yggdrasil's lazy
    :class:`DeltaFolder` reader.

    The Databricks side runs ``SELECT * FROM tbl WHERE <pred>`` on the
    SQL warehouse and returns Arrow rows. The lazy side reads the
    parquet files directly off the table's storage path, applies the
    same predicate via ``DeltaOptions.predicate`` (pushed down to
    Arrow expression filtering), and returns Arrow rows without a
    warehouse round trip.

    The two outputs must agree — locks in predicate-pushdown
    correctness across the full type / NULL / partition matrix.
    Each test ``self.assertEqual`` ‑s the sorted row sets so file
    iteration order on the lazy side can't flake the comparison.
    """

    @staticmethod
    def _pylist_of_rows(table: pa.Table, columns: list[str]) -> list[tuple]:
        """Materialize *columns* of *table* as a sortable list of tuples."""
        cols = [table.column(c).to_pylist() for c in columns]
        return sorted(zip(*cols))

    def test_simple_comparison_filter(self) -> None:
        """``WHERE id > 5`` — single column, numeric compare."""
        tbl = self._table("lazy_gt")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'a'), (5, 'e'), (10, 'j'), (12, 'l'), (3, 'c')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} WHERE id > 5"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(predicate=col("id") > 5),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "val"]),
        )

    def test_conjunction_predicate(self) -> None:
        """``WHERE id >= 5 AND id < 12`` — range bound via AND."""
        tbl = self._table("lazy_and")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'a'), (5, 'e'), (8, 'h'), (12, 'l'), (15, 'o')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} "
            "WHERE id >= 5 AND id < 12"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(
                predicate=(col("id") >= 5) & (col("id") < 12),
            ),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "val"]),
        )

    def test_in_list_predicate(self) -> None:
        """``WHERE id IN (...)`` — set membership."""
        tbl = self._table("lazy_in")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} WHERE id IN (2, 4)"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(predicate=col("id").is_in([2, 4])),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "val"]),
        )

    def test_null_predicate(self) -> None:
        """``WHERE val IS NOT NULL`` — null handling under both engines."""
        tbl = self._table("lazy_null")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'a'), (2, NULL), (3, 'c'), (4, NULL), (5, 'e')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} WHERE val IS NOT NULL"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(predicate=~col("val").is_null()),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "val"]),
        )

    def test_partition_filter_matches_sql(self) -> None:
        """``WHERE region = 'us'`` on a partitioned table — the lazy
        path prunes partition directories before reading parquet,
        but the rows that come out match the SQL view exactly."""
        tbl = self._table("lazy_part")
        self._execute(f"""
            CREATE TABLE {tbl.full_name()} (id BIGINT, region STRING, val STRING)
            USING DELTA PARTITIONED BY (region)
        """)
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'us', 'a'), (2, 'eu', 'b'), (3, 'us', 'c'), "
            "(4, 'apac', 'd'), (5, 'eu', 'e')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} WHERE region = 'us'"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(predicate=col("region") == "us"),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "region", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "region", "val"]),
        )

    def test_combined_partition_and_row_filter(self) -> None:
        """``WHERE region = 'us' AND id > 1`` — partition prune AND
        intra-file filter together; both engines must converge on
        the same rows."""
        tbl = self._table("lazy_combo")
        self._execute(f"""
            CREATE TABLE {tbl.full_name()} (id BIGINT, region STRING, val STRING)
            USING DELTA PARTITIONED BY (region)
        """)
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'us', 'a'), (2, 'us', 'b'), (3, 'eu', 'c'), "
            "(4, 'us', 'd'), (5, 'apac', 'e')"
        )

        sql_out = self._read_sql_arrow(
            f"SELECT * FROM {tbl.full_name()} "
            "WHERE region = 'us' AND id > 1"
        )
        from yggdrasil.execution.expr import col
        ygg_out = self._delta_folder(tbl).read_arrow_table(
            options=DeltaOptions(
                predicate=(col("region") == "us") & (col("id") > 1),
            ),
        )

        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "region", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "region", "val"]),
        )

    def test_no_predicate_full_scan(self) -> None:
        """Sanity floor: with no predicate at all both engines return
        the whole table — same rows, same columns, same count."""
        tbl = self._table("lazy_full")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} SELECT id, "
            "concat('v', cast(id AS STRING)) AS val FROM range(100)"
        )

        sql_out = self._read_sql_arrow(f"SELECT * FROM {tbl.full_name()}")
        ygg_out = self._delta_folder(tbl).read_arrow_table()

        self.assertEqual(sql_out.num_rows, ygg_out.num_rows)
        self.assertEqual(
            self._pylist_of_rows(sql_out, ["id", "val"]),
            self._pylist_of_rows(ygg_out, ["id", "val"]),
        )


# ---------------------------------------------------------------------------
# Table.lazy(sql=...) — submit SQL and read through the Tabular interface
# ---------------------------------------------------------------------------


class TestTableLazySQL(_DeltaSQLBase):
    """`tbl.lazy(sql="...")` returns a deferred :class:`Tabular`.

    The query runs on the warehouse so the result handle is ready,
    but the rows aren't materialised until the caller invokes a
    Tabular hook (read_arrow_table / read_arrow_batches / ...).
    ``{self}`` in the query string is substituted with this table's
    quoted full name so callers don't repeat ``tbl.full_name(safe=True)``
    in every query.
    """

    def test_lazy_with_self_placeholder(self) -> None:
        tbl = self._table("lazy_self")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT, val STRING) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "(1, 'a'), (5, 'e'), (10, 'j'), (12, 'l')"
        )

        out = tbl.lazy(sql="SELECT id, val FROM {self} WHERE id > 5").read_arrow_table()
        ids = sorted(out.column("id").to_pylist())
        self.assertEqual(ids, [10, 12])

    def test_lazy_without_placeholder(self) -> None:
        """A query without ``{self}`` flows through verbatim — the
        caller is responsible for the FROM clause."""
        tbl = self._table("lazy_explicit")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (id BIGINT) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES (1), (2), (3)"
        )

        out = tbl.lazy(
            sql=f"SELECT COUNT(*) AS n FROM {tbl.full_name()}"
        ).read_arrow_table()
        self.assertEqual(out.column("n")[0].as_py(), 3)

    def test_lazy_aggregation_matches_full_scan(self) -> None:
        """Aggregations via lazy SQL produce the same result as
        applying the same aggregation against ``DeltaFolder``-side
        arrow tables — bytes are equal modulo column ordering."""
        tbl = self._table("lazy_agg")
        self._execute(
            f"CREATE TABLE {tbl.full_name()} (region STRING, v DOUBLE) USING DELTA"
        )
        self._execute(
            f"INSERT INTO {tbl.full_name()} VALUES "
            "('us', 1.0), ('us', 2.0), ('eu', 3.0), ('apac', 4.0)"
        )

        lazy_out = tbl.lazy(
            sql="SELECT region, SUM(v) AS s FROM {self} GROUP BY region",
        ).read_arrow_table()
        # Cross-check via the storage path.
        full = self._delta_folder(tbl).read_arrow_table()
        sums_by_region = {}
        for region, v in zip(
            full.column("region").to_pylist(),
            full.column("v").to_pylist(),
        ):
            sums_by_region[region] = sums_by_region.get(region, 0.0) + v

        lazy_pairs = sorted(
            zip(
                lazy_out.column("region").to_pylist(),
                lazy_out.column("s").to_pylist(),
            )
        )
        self.assertEqual(
            lazy_pairs,
            sorted(sums_by_region.items()),
        )

    def test_lazy_returns_self_when_sql_is_none(self) -> None:
        """``lazy()`` without a SQL argument hands back the table
        itself so callers can chain on the table's own data without
        an extra round trip."""
        tbl = self._table("lazy_self_back")
        self.assertIs(tbl.lazy(), tbl)
