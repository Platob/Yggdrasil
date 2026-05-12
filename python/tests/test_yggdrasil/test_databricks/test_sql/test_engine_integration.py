"""Live-integration tests for :class:`SQLEngine` and :class:`Table`.

Skipped unless ``DATABRICKS_HOST`` (and the matching credentials) are
exported via the standard SDK env vars — see
:class:`DatabricksIntegrationCase`.

Scope
-----
The fixture is pinned to ``trading.unittest`` (the catalog/schema can
be overridden via :envvar:`DATABRICKS_INTEGRATION_CATALOG` /
:envvar:`DATABRICKS_INTEGRATION_SCHEMA`). Each test touches a unique
managed table name so concurrent runs don't collide and a partial
failure leaves at most one orphan table behind.

Auto-create policy
------------------
The engine and table methods exercised here are *opportunistic*:
operations are attempted directly, and the catalog / schema / table
are only created on demand when the operation surfaces a missing
resource. ``setUpClass`` falls back to ``ensure_created`` when the
upfront probe fails so the run can still proceed against a pristine
workspace.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import secrets
from typing import ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound

from yggdrasil.data.enums import Mode
from yggdrasil.databricks.sql.engine import SQLEngine
from yggdrasil.databricks.sql.table import Table

from .. import DatabricksIntegrationCase


__all__ = [
    "TestSQLEngineIntegration",
    "TestSQLMergeStrategy",
    "TestSQLInsertFillMissingColumns",
    "TestSQLConcurrentWrites",
]


class _SQLIntegrationBase(DatabricksIntegrationCase):
    """Shared fixture + helpers for the SQL integration suites.

    Not collected by pytest (no ``Test`` prefix). Provisions
    ``trading.unittest`` once per concrete subclass via
    ``ensure_created``, registers minted tables for class-level
    cleanup, and exposes small builders for sample schemas / data.
    """

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    engine: ClassVar[SQLEngine]
    created_tables: ClassVar[list[str]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = os.environ.get(
            "DATABRICKS_INTEGRATION_CATALOG", "trading",
        ).strip() or "trading"
        cls.schema_name = os.environ.get(
            "DATABRICKS_INTEGRATION_SCHEMA", "unittest",
        ).strip() or "unittest"

        # Engine scoped to ``trading.unittest`` so every method that takes
        # ``catalog_name`` / ``schema_name`` keyword args inherits the right
        # default and unqualified table names resolve correctly.
        cls.engine = cls.client.sql(
            catalog_name=cls.catalog_name,
            schema_name=cls.schema_name,
        )
        cls.created_tables = []

        # Best-effort ensure_created on the catalog + schema. The engine /
        # table methods will retry the same path on per-test demand if a
        # transient miss surfaces later.
        cls._ensure_catalog_schema()

    @classmethod
    def _ensure_catalog_schema(cls) -> None:
        """Make sure ``trading.unittest`` exists; skip the suite if we
        can't get there (no permission to create the catalog, etc.)."""
        catalog = cls.engine.catalogs.catalog(cls.catalog_name)
        try:
            catalog.ensure_created(
                comment="yggdrasil integration-test catalog",
            )
        except DatabricksError as exc:
            # Most realistic miss: no metastore-admin grant. Skip the
            # suite cleanly so the rest of the local run isn't blocked.
            import unittest as _ut
            raise _ut.SkipTest(
                f"Cannot create or access catalog {cls.catalog_name!r}: "
                f"{exc}. Set DATABRICKS_INTEGRATION_CATALOG to a catalog "
                "the test identity can write to, or pre-provision "
                f"{cls.catalog_name}.{cls.schema_name}."
            ) from exc

        schema = cls.engine.schemas.schema(
            f"{cls.catalog_name}.{cls.schema_name}"
        )
        schema.ensure_created(
            comment="yggdrasil integration-test schema",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            for full_name in cls.created_tables:
                try:
                    cls.engine.table(full_name).delete(raise_error=False)
                except DatabricksError:
                    pass
        finally:
            super().tearDownClass()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unique_table(self, prefix: str) -> Table:
        """Return a fresh :class:`Table` handle with a unique name.

        The handle is registered for class-level cleanup before any
        DDL runs, so a test that fails mid-flight still leaves the
        teardown loop with something to drop.
        """
        name = f"yg_{prefix}_{secrets.token_hex(4)}"
        full_name = f"{self.catalog_name}.{self.schema_name}.{name}"
        type(self).created_tables.append(full_name)
        return self.engine.table(full_name)

    @staticmethod
    def _sample_schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("label", pa.string()),
                pa.field("amount", pa.float64()),
            ]
        )

    @staticmethod
    def _sample_data() -> pa.Table:
        return pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
                "amount": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            }
        )

    def _ensure_table(self, table: Table, definition: pa.Schema) -> Table:
        """Create the managed table if a probe says it's missing.

        Mirrors the "try operation; on miss, auto-create" contract the
        caller asked for: we read ``.exists`` and only call
        ``ensure_created`` when the answer is no.
        """
        if not table.exists:
            table.ensure_created(definition)
        return table


class TestSQLEngineIntegration(_SQLIntegrationBase):
    """Engine + table CRUD against a real workspace."""

    # ------------------------------------------------------------------
    # Catalog / schema lifecycle
    # ------------------------------------------------------------------

    def test_catalog_and_schema_exist(self) -> None:
        catalog = self.engine.catalogs.catalog(self.catalog_name)
        self.assertTrue(catalog.exists)

        schema = self.engine.schemas.schema(
            f"{self.catalog_name}.{self.schema_name}"
        )
        self.assertTrue(schema.exists)

    # ------------------------------------------------------------------
    # Engine.execute — basic query
    # ------------------------------------------------------------------

    def test_execute_select_literal(self) -> None:
        result = self.engine.execute("SELECT 1 AS one, 'hello' AS greeting")
        arrow_table = result.to_arrow_table()

        self.assertEqual(arrow_table.num_rows, 1)
        self.assertEqual(arrow_table.column("one")[0].as_py(), 1)
        self.assertEqual(arrow_table.column("greeting")[0].as_py(), "hello")

    def test_execute_uses_engine_scope(self) -> None:
        """SQL run through the engine should resolve unqualified
        identifiers against the engine's catalog/schema scope."""
        result = self.engine.execute("SELECT current_catalog() AS c, current_schema() AS s")
        row = result.to_arrow_table().to_pylist()[0]

        self.assertEqual(row["c"], self.catalog_name)
        self.assertEqual(row["s"], self.schema_name)

    def test_execute_empty_result_preserves_metadata_schema(self) -> None:
        """A SELECT that yields zero rows still surfaces a typed
        :class:`pa.Schema` derived from the warehouse statement
        manifest, not a schema-less empty table.

        Guards against the regression where ``read_arrow_table`` would
        collapse to ``Schema.empty()`` when the result chunk iterator
        was empty (the warehouse never streams a header batch on an
        empty result set, so the schema has to come from the manifest).
        """
        result = self.engine.execute(
            "SELECT CAST(1 AS BIGINT) AS id, CAST('x' AS STRING) AS label, "
            "CAST(NULL AS DOUBLE) AS amount WHERE 1 = 0"
        )

        arrow_table = result.to_arrow_table()
        self.assertEqual(arrow_table.num_rows, 0)
        self.assertEqual(
            arrow_table.schema.names, ["id", "label", "amount"],
        )
        self.assertEqual(arrow_table.schema.field("id").type, pa.int64())
        self.assertEqual(arrow_table.schema.field("label").type, pa.string())
        self.assertEqual(arrow_table.schema.field("amount").type, pa.float64())

    # ------------------------------------------------------------------
    # Engine.create_table — auto-create on miss
    # ------------------------------------------------------------------

    def test_engine_create_table_auto_creates_managed_table(self) -> None:
        table = self._unique_table("create")
        self.assertFalse(table.exists)

        created = self.engine.create_table(
            self._sample_schema(),
            full_name=table.full_name(),
        )
        self.assertTrue(created.exists)

        # Idempotent: a second create_table call with if_not_exists=True
        # (the engine default) must not raise.
        again = self.engine.create_table(
            self._sample_schema(),
            full_name=table.full_name(),
        )
        self.assertTrue(again.exists)

    # ------------------------------------------------------------------
    # Table.create / Table.ensure_created
    # ------------------------------------------------------------------

    def test_table_ensure_created_is_idempotent(self) -> None:
        table = self._unique_table("ensure")
        table.ensure_created(self._sample_schema())
        self.assertTrue(table.exists)

        # Second call is a no-op (or schema-merge no-op) on an existing
        # table with the same definition.
        table.ensure_created(self._sample_schema())
        self.assertTrue(table.exists)

    # ------------------------------------------------------------------
    # Table.insert / Engine.insert_into — auto-create on miss, then read back
    # ------------------------------------------------------------------

    def test_table_insert_auto_creates_then_round_trips(self) -> None:
        """Insert against a missing table: the helper materializes it
        on demand, then the inserted rows round-trip through SELECT."""
        table = self._unique_table("insert")
        data = self._sample_data()

        self._ensure_table(table, data.schema)
        table.insert(data, mode=Mode.OVERWRITE)

        read = self.engine.execute(
            f"SELECT id, label, amount FROM {table.full_name(safe=True)} "
            "ORDER BY id"
        ).to_arrow_table()

        self.assertEqual(read.num_rows, 3)
        self.assertEqual(read.column("id").to_pylist(), [1, 2, 3])
        self.assertEqual(read.column("label").to_pylist(), ["a", "b", "c"])
        self.assertEqual(read.column("amount").to_pylist(), [1.5, 2.5, 3.5])

    def test_engine_insert_into_resolves_target(self) -> None:
        table = self._unique_table("engine_insert")
        data = self._sample_data()

        self._ensure_table(table, data.schema)
        self.engine.insert_into(
            data,
            location=table.full_name(),
            mode=Mode.OVERWRITE,
        )

        read = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}"
        ).to_arrow_table()
        self.assertEqual(read.column("n")[0].as_py(), 3)

    # ------------------------------------------------------------------
    # Table.delete — drops the table; the second drop is a no-op
    # ------------------------------------------------------------------

    def test_table_delete_drops_then_missing(self) -> None:
        table = self._unique_table("delete")
        table.ensure_created(self._sample_schema())
        self.assertTrue(table.exists)

        table.delete()
        # Re-resolve through the service so we don't read a stale cached
        # ``_infos`` from the in-memory handle that just dropped itself.
        self.assertFalse(self.engine.table(table.full_name()).exists)

    # ------------------------------------------------------------------
    # external_data — the generic ExternalStatementData binding
    # ------------------------------------------------------------------

    def test_execute_with_arrow_external_data_round_trips(self) -> None:
        """A tabular passed via ``external_data`` is staged as a
        temporary Parquet volume and the ``{alias}`` placeholder in the
        statement text is rewritten to point at it.  The same pathway
        also populates the new generic
        :attr:`PreparedStatement.external_data` registry — exercised
        below by inspecting the prepared statement after execution.
        """
        from yggdrasil.databricks.warehouse.statement import (
            WarehousePreparedStatement,
        )

        data = pa.table(
            {
                "id": pa.array([10, 20, 30], type=pa.int64()),
                "label": pa.array(["x", "y", "z"], type=pa.string()),
            }
        )
        prepared = WarehousePreparedStatement.prepare(
            "SELECT id, label FROM {ext} ORDER BY id",
            external_data={"ext": data},
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )
        # Both registries are populated by ``prepare``: the legacy
        # ``external_volume_paths`` (a real staged path) and the generic
        # ``external_data`` (text_value pre-baked from that path).
        self.assertIn("ext", prepared.external_volume_paths or {})
        self.assertIn("ext", prepared.external_data or {})
        self.assertTrue(
            prepared.external_data["ext"].text_value.startswith("parquet.`")
        )

        try:
            arrow_table = self.engine.execute(prepared).to_arrow_table()
            self.assertEqual(arrow_table.num_rows, 3)
            self.assertEqual(arrow_table.column("id").to_pylist(), [10, 20, 30])
            self.assertEqual(
                arrow_table.column("label").to_pylist(), ["x", "y", "z"],
            )
        finally:
            prepared.clear_temporary_resources()

    def test_execute_with_external_statement_data_text_value(self) -> None:
        """An :class:`ExternalStatementData` with a pre-baked
        ``text_value`` (here a ``VALUES`` clause) is substituted
        verbatim — no staging round-trip, no warehouse volume created.
        """
        from yggdrasil.data.statement import ExternalStatementData
        from yggdrasil.databricks.warehouse.statement import (
            WarehousePreparedStatement,
        )

        prepared = WarehousePreparedStatement.prepare(
            "SELECT id FROM {src} ORDER BY id",
            external_data={
                "src": ExternalStatementData(
                    "src", text_value="(VALUES (1), (2), (3)) AS t(id)",
                ),
            },
        )
        # No staging happened — only the generic registry is populated.
        self.assertIsNone(prepared.external_volume_paths)
        self.assertEqual(
            prepared.external_data["src"].text_value,
            "(VALUES (1), (2), (3)) AS t(id)",
        )

        arrow_table = self.engine.execute(prepared).to_arrow_table()
        self.assertEqual(arrow_table.column("id").to_pylist(), [1, 2, 3])

    def test_engine_execute_api_mode_with_external_data(self) -> None:
        """``engine.execute(external_data=..., engine="api")`` stages
        an Arrow table to a temporary volume and substitutes ``{alias}``
        with ``parquet.`<full>`` in the statement text — same path the
        warehouse-only ``WarehousePreparedStatement.prepare`` exercises,
        but driven from the engine entry point so the
        ``engine="api"`` branch picks it up explicitly.
        """
        data = pa.table(
            {
                "id": pa.array([100, 200, 300], type=pa.int64()),
                "label": pa.array(["p", "q", "r"], type=pa.string()),
            }
        )
        result = self.engine.execute(
            "SELECT id, label FROM {ext} ORDER BY id",
            external_data={"ext": data},
            engine="api",
        )
        try:
            arrow_table = result.to_arrow_table()
            self.assertEqual(arrow_table.num_rows, 3)
            self.assertEqual(
                arrow_table.column("id").to_pylist(), [100, 200, 300],
            )
            self.assertEqual(
                arrow_table.column("label").to_pylist(), ["p", "q", "r"],
            )
        finally:
            result.statement.clear_temporary_resources()

    def test_engine_execute_spark_mode_with_arrow_external_data(self) -> None:
        """``engine.execute(external_data=..., engine="spark")``
        registers the Arrow table as a Spark temp view and substitutes
        ``{alias}`` with the view name. No volume staging happens — the
        Spark path does not need to round-trip through Parquet.
        """
        try:
            import pyspark  # noqa: F401
        except ImportError:
            self.skipTest("pyspark is not installed in this environment")

        data = pa.table(
            {
                "id": pa.array([7, 8, 9], type=pa.int64()),
                "label": pa.array(["s", "t", "u"], type=pa.string()),
            }
        )
        result = self.engine.execute(
            "SELECT id, label FROM {ext} ORDER BY id",
            external_data={"ext": data},
            engine="spark",
        )
        try:
            self.assertIsNotNone(result.statement.external_data)
            self.assertIn("ext", result.statement.external_data)
            # Spark fills in text_value with the temp-view name during
            # ``start``; it is non-None and is *not* the volume parquet
            # form the warehouse path produces.
            view_text = result.statement.external_data["ext"].text_value
            self.assertIsNotNone(view_text)
            self.assertFalse(view_text.startswith("parquet.`"))

            arrow_table = result.to_arrow_table()
            self.assertEqual(arrow_table.num_rows, 3)
            self.assertEqual(arrow_table.column("id").to_pylist(), [7, 8, 9])
            self.assertEqual(
                arrow_table.column("label").to_pylist(), ["s", "t", "u"],
            )
        finally:
            result.clear_temporary_resources()

    def test_engine_execute_spark_mode_with_volume_path_external_data(self) -> None:
        """A pre-existing :class:`VolumePath` passed via
        ``external_data`` on the Spark path is text-substituted as
        ``parquet.`<full>`` — Spark reads parquet by path, no temp view
        registered."""
        try:
            import pyspark  # noqa: F401
        except ImportError:
            self.skipTest("pyspark is not installed in this environment")

        from yggdrasil.databricks.fs import VolumePath

        data = pa.table(
            {
                "id": pa.array([42, 43], type=pa.int64()),
                "label": pa.array(["m", "n"], type=pa.string()),
            }
        )
        # Stage the volume up-front via the new factory shortcut so the
        # caller can hand the engine a ready-to-read VolumePath.
        path = VolumePath.staging_path(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            resource_name="engine_spark_volume",
            tabular=data,
        )
        try:
            result = self.engine.execute(
                "SELECT id, label FROM {ext} ORDER BY id",
                external_data={"ext": path},
                engine="spark",
            )
            try:
                self.assertIn("ext", result.statement.external_data or {})
                self.assertTrue(
                    result.statement.external_data["ext"].text_value.startswith(
                        "parquet.`"
                    )
                )
                arrow_table = result.to_arrow_table()
                self.assertEqual(
                    arrow_table.column("id").to_pylist(), [42, 43],
                )
                self.assertEqual(
                    arrow_table.column("label").to_pylist(), ["m", "n"],
                )
            finally:
                result.clear_temporary_resources()
        finally:
            path.unlink(missing_ok=True)

    def test_volume_path_staging_path_writes_tabular_in_one_call(self) -> None:
        """``VolumePath.staging_path(tabular=...)`` mints the path
        *and* writes the supplied tabular as Parquet — replaces the
        previous two-step ``staging_path(...)`` followed by
        ``as_media(PARQUET).write_table(...)`` pattern."""
        from yggdrasil.databricks.fs import VolumePath

        data = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )
        path = VolumePath.staging_path(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            resource_name="staging_factory",
            tabular=data,
        )
        try:
            # File exists with the written bytes (size > 0).
            self.assertGreater(path.size, 0)

            # And the warehouse can read it back as a parquet table —
            # that's the contract callers depend on.
            arrow_table = self.engine.execute(
                "SELECT id, label FROM parquet.`%s` ORDER BY id"
                % path.full_path()
            ).to_arrow_table()
            self.assertEqual(arrow_table.num_rows, 3)
            self.assertEqual(arrow_table.column("id").to_pylist(), [1, 2, 3])
            self.assertEqual(
                arrow_table.column("label").to_pylist(), ["a", "b", "c"],
            )
        finally:
            path.unlink(missing_ok=True)

    def test_engine_drop_missing_table_is_no_op(self) -> None:
        """``drop_table`` on a name that was never created should not
        raise — UC's DROP TABLE IF EXISTS contract."""
        ghost = (
            f"{self.catalog_name}.{self.schema_name}."
            f"yg_ghost_{secrets.token_hex(4)}"
        )
        # Pre-condition: table really doesn't exist.
        with self.assertRaises(NotFound):
            self.engine.table(ghost).infos  # noqa: B018 — assert raises

        # ``drop_table`` swallows the NotFound and returns cleanly.
        self.engine.drop_table(ghost, raise_error=False)


# =====================================================================
# MERGE strategy
# =====================================================================


class TestSQLMergeStrategy(_SQLIntegrationBase):
    """Each save-mode + ``match_by`` combination through ``Table.insert``.

    Reuses the catalog/schema fixture and per-test cleanup from
    :class:`TestSQLEngineIntegration` and adds a small helper to
    seed a fresh table with a known initial row set. The expected
    DML each branch generates is documented inline in
    ``yggdrasil.databricks.sql.table._build_dml_statements``; this
    suite verifies the *observable* outcome (row counts + values).
    """

    @staticmethod
    def _initial_data() -> pa.Table:
        """Three rows with keys 1/2/3 — the seed every test starts from."""
        return pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
                "amount": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
            }
        )

    def _seed(self, prefix: str) -> Table:
        """Build a unique table, create it, and seed with ``_initial_data``."""
        data = self._initial_data()
        table = self._unique_table(prefix)
        self._ensure_table(table, data.schema)
        # OVERWRITE so the seed is deterministic regardless of any
        # auto-create state the helper may have left behind.
        table.insert(data, mode=Mode.OVERWRITE)
        return table

    def _read_rows(self, table: Table) -> list[dict]:
        return self.engine.execute(
            f"SELECT id, label, amount FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

    # ------------------------------------------------------------------
    # MERGE — full UPSERT (update matching, insert new)
    # ------------------------------------------------------------------

    def test_merge_upsert_updates_matching_and_inserts_new(self) -> None:
        """``Mode.UPSERT`` + ``match_by=['id']`` → MERGE INTO with both
        WHEN MATCHED UPDATE and WHEN NOT MATCHED INSERT branches. Rows
        with overlapping keys take the source values; new keys land."""
        table = self._seed("merge_upsert")

        update = pa.table(
            {
                # id=2 overlaps (must be updated); id=4 is new (must be inserted).
                "id": pa.array([2, 4], type=pa.int64()),
                "label": pa.array(["beta-2", "delta"], type=pa.string()),
                "amount": pa.array([200.0, 40.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.UPSERT, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha", "amount": 10.0},
                {"id": 2, "label": "beta-2", "amount": 200.0},
                {"id": 3, "label": "gamma", "amount": 30.0},
                {"id": 4, "label": "delta", "amount": 40.0},
            ],
        )

    def test_merge_alias_matches_upsert(self) -> None:
        """``Mode.MERGE`` is treated as a synonym for ``UPSERT`` —
        same MERGE-with-update branch."""
        table = self._seed("merge_alias")

        update = pa.table(
            {
                "id": pa.array([1], type=pa.int64()),
                "label": pa.array(["alpha-merged"], type=pa.string()),
                "amount": pa.array([100.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.MERGE, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(rows[0], {"id": 1, "label": "alpha-merged", "amount": 100.0})
        self.assertEqual(len(rows), 3)

    # ------------------------------------------------------------------
    # MERGE — INSERT-ONLY branches
    # ------------------------------------------------------------------

    def test_merge_append_with_match_by_skips_existing_keys(self) -> None:
        """``Mode.APPEND`` + ``match_by`` → MERGE WHEN NOT MATCHED INSERT
        only; rows whose key already exists are dropped silently."""
        table = self._seed("merge_append_keyed")

        update = pa.table(
            {
                # id=2 already exists → skipped; id=5 is new → inserted.
                "id": pa.array([2, 5], type=pa.int64()),
                "label": pa.array(["beta-DROPPED", "epsilon"], type=pa.string()),
                "amount": pa.array([999.0, 50.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.APPEND, match_by=["id"])

        rows = self._read_rows(table)
        # Original beta still wins because APPEND+match_by never updates.
        self.assertEqual(rows[1], {"id": 2, "label": "beta", "amount": 20.0})
        self.assertEqual(
            rows[-1], {"id": 5, "label": "epsilon", "amount": 50.0}
        )
        self.assertEqual(len(rows), 4)

    def test_merge_auto_with_match_by_skips_existing_keys(self) -> None:
        """``Mode.AUTO`` + ``match_by`` behaves the same as APPEND —
        insert-only MERGE."""
        table = self._seed("merge_auto_keyed")

        update = pa.table(
            {
                "id": pa.array([3, 6], type=pa.int64()),
                "label": pa.array(["gamma-DROPPED", "zeta"], type=pa.string()),
                "amount": pa.array([999.0, 60.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.AUTO, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(rows[2], {"id": 3, "label": "gamma", "amount": 30.0})
        self.assertEqual(rows[-1], {"id": 6, "label": "zeta", "amount": 60.0})
        self.assertEqual(len(rows), 4)

    # ------------------------------------------------------------------
    # TRUNCATE + match_by → keyed DELETE then INSERT
    # ------------------------------------------------------------------

    def test_truncate_with_match_by_deletes_matching_then_inserts(self) -> None:
        table = self._seed("merge_truncate_keyed")

        update = pa.table(
            {
                # Wipe id=1 and id=3 (matched by key) and re-insert with new values.
                # id=2 is untouched because the source doesn't carry that key.
                "id": pa.array([1, 3], type=pa.int64()),
                "label": pa.array(["alpha-v2", "gamma-v2"], type=pa.string()),
                "amount": pa.array([11.0, 33.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.TRUNCATE, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha-v2", "amount": 11.0},
                {"id": 2, "label": "beta", "amount": 20.0},
                {"id": 3, "label": "gamma-v2", "amount": 33.0},
            ],
        )

    def test_truncate_without_match_by_wipes_table(self) -> None:
        """Plain ``Mode.TRUNCATE`` without keys: full TRUNCATE + INSERT."""
        table = self._seed("merge_truncate_full")

        replacement = pa.table(
            {
                "id": pa.array([99], type=pa.int64()),
                "label": pa.array(["only"], type=pa.string()),
                "amount": pa.array([0.0], type=pa.float64()),
            }
        )


        table.insert(replacement, mode=Mode.TRUNCATE)

        rows = self._read_rows(table)
        self.assertEqual(rows, [{"id": 99, "label": "only", "amount": 0.0}])

    # ------------------------------------------------------------------
    # safe_merge=True — sidestep MERGE entirely (DELETE+INSERT semantics)
    # ------------------------------------------------------------------

    def test_safe_merge_upsert_round_trip(self) -> None:
        """``safe_merge=True`` + UPSERT runs keyed DELETE + INSERT
        instead of MERGE; outcome is identical for non-overlapping
        writers."""
        table = self._seed("safe_merge_upsert")

        update = pa.table(
            {
                "id": pa.array([2, 4], type=pa.int64()),
                "label": pa.array(["beta-safe", "delta-safe"], type=pa.string()),
                "amount": pa.array([222.0, 44.0], type=pa.float64()),
            }
        )
        table.insert(
            update, mode=Mode.UPSERT, match_by=["id"], safe_merge=True,
        )

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha", "amount": 10.0},
                {"id": 2, "label": "beta-safe", "amount": 222.0},
                {"id": 3, "label": "gamma", "amount": 30.0},
                {"id": 4, "label": "delta-safe", "amount": 44.0},
            ],
        )

    def test_safe_merge_append_uses_anti_join_insert(self) -> None:
        """``safe_merge=True`` + APPEND runs ``INSERT ... WHERE NOT
        EXISTS`` against the target — same observable result as the
        native MERGE insert-only branch."""
        table = self._seed("safe_merge_append")

        update = pa.table(
            {
                "id": pa.array([1, 7], type=pa.int64()),
                "label": pa.array(["alpha-DROPPED", "eta"], type=pa.string()),
                "amount": pa.array([999.0, 70.0], type=pa.float64()),
            }
        )
        table.insert(
            update, mode=Mode.APPEND, match_by=["id"], safe_merge=True,
        )

        rows = self._read_rows(table)
        # id=1 untouched; id=7 added.
        self.assertEqual(rows[0], {"id": 1, "label": "alpha", "amount": 10.0})
        self.assertEqual(rows[-1], {"id": 7, "label": "eta", "amount": 70.0})
        self.assertEqual(len(rows), 4)

    # ------------------------------------------------------------------
    # Multi-key match_by
    # ------------------------------------------------------------------

    def test_merge_upsert_composite_key(self) -> None:
        """``match_by`` accepts multiple columns — MERGE ON joins on the
        full key tuple."""
        table = self._unique_table("merge_composite")
        seed = pa.table(
            {
                "tenant": pa.array(["a", "a", "b"], type=pa.string()),
                "id": pa.array([1, 2, 1], type=pa.int64()),
                "value": pa.array([10.0, 20.0, 100.0], type=pa.float64()),
            }
        )
        self._ensure_table(table, seed.schema)
        table.insert(seed, mode=Mode.OVERWRITE)

        update = pa.table(
            {
                # (a, 2) overlaps → updated; (b, 2) is new → inserted.
                "tenant": pa.array(["a", "b"], type=pa.string()),
                "id": pa.array([2, 2], type=pa.int64()),
                "value": pa.array([222.0, 200.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.UPSERT, match_by=["tenant", "id"])

        rows = self.engine.execute(
            f"SELECT tenant, id, value FROM {table.full_name(safe=True)} "
            "ORDER BY tenant, id"
        ).to_arrow_table().to_pylist()
        self.assertEqual(
            rows,
            [
                {"tenant": "a", "id": 1, "value": 10.0},
                {"tenant": "a", "id": 2, "value": 222.0},
                {"tenant": "b", "id": 1, "value": 100.0},
                {"tenant": "b", "id": 2, "value": 200.0},
            ],
        )


# =====================================================================
# Schema fill — source columns are a subset of the target schema
# =====================================================================


class TestSQLInsertFillMissingColumns(_SQLIntegrationBase):
    """Insert source rows whose schema is a strict subset of the target.

    The Arrow staging cast (``cast_arrow_tabular`` in
    ``yggdrasil.data.types.nested.struct_arrow``) materialises a default
    array for every target column the source doesn't carry, so the
    staged Parquet — and therefore the ``INSERT INTO target (cols)
    SELECT ...`` projection that ``arrow_insert`` builds against the
    target schema — always has a value for every target column. These
    tests pin that contract by inserting subset payloads through the
    public ``Table.insert`` API and reading the rows back: missing
    columns must land as ``NULL`` for inserted rows, and the
    ``update_column_names`` knob must protect target-only columns on
    MERGE-matched rows.
    """

    @staticmethod
    def _wide_schema() -> pa.Schema:
        """Target schema with one key + three nullable payload columns.

        The key (``id``) is non-nullable so the tests can fail loudly
        if the missing-column fill ever leaks onto the key.
        """
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("label", pa.string()),
                pa.field("amount", pa.float64()),
                pa.field("active", pa.bool_()),
            ]
        )

    def _read_rows(self, table: Table) -> list[dict]:
        return self.engine.execute(
            f"SELECT id, label, amount, active "
            f"FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

    # ------------------------------------------------------------------
    # APPEND — fill one missing column with NULL
    # ------------------------------------------------------------------

    def test_append_fills_single_missing_column_with_null(self) -> None:
        """Source carries id+label+active; target also has ``amount``.
        Inserted rows must land with ``amount = NULL``."""
        table = self._unique_table("fill_one_append")
        table.ensure_created(self._wide_schema())

        partial = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["a", "b"], type=pa.string()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(partial, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "a", "amount": None, "active": True},
                {"id": 2, "label": "b", "amount": None, "active": False},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND — fill multiple missing columns with NULL
    # ------------------------------------------------------------------

    def test_append_fills_multiple_missing_columns_with_null(self) -> None:
        """Source carries the key column only — every payload column
        on the target must land as ``NULL``."""
        table = self._unique_table("fill_many_append")
        table.ensure_created(self._wide_schema())

        partial = pa.table(
            {"id": pa.array([10, 20], type=pa.int64())}
        )
        table.insert(partial, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 10, "label": None, "amount": None, "active": None},
                {"id": 20, "label": None, "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND — source columns in a different order than target
    # ------------------------------------------------------------------

    def test_append_reorders_source_columns_to_target_schema(self) -> None:
        """Cast is by name, not by position — a reordered + truncated
        source still lines up with the target columns."""
        table = self._unique_table("fill_reorder_append")
        table.ensure_created(self._wide_schema())

        reordered = pa.table(
            {
                # Intentionally label, id (reversed) and no amount/active.
                "label": pa.array(["x", "y"], type=pa.string()),
                "id": pa.array([100, 200], type=pa.int64()),
            }
        )
        table.insert(reordered, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 100, "label": "x", "amount": None, "active": None},
                {"id": 200, "label": "y", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # OVERWRITE — fill missing columns on a replace
    # ------------------------------------------------------------------

    def test_overwrite_fills_missing_columns_with_null(self) -> None:
        """``Mode.OVERWRITE`` drops the target before insert; the
        subset replacement must keep the wider target schema and fill
        the absent columns with NULL."""
        table = self._unique_table("fill_overwrite")
        table.ensure_created(self._wide_schema())

        # Seed so the OVERWRITE actually replaces something.
        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["seed-a", "seed-b"], type=pa.string()),
                "amount": pa.array([1.5, 2.5], type=pa.float64()),
                "active": pa.array([True, True], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        partial = pa.table(
            {
                "id": pa.array([5, 6], type=pa.int64()),
                "label": pa.array(["fresh-a", "fresh-b"], type=pa.string()),
            }
        )
        table.insert(partial, mode=Mode.OVERWRITE)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 5, "label": "fresh-a", "amount": None, "active": None},
                {"id": 6, "label": "fresh-b", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # UPSERT — match by key, missing source columns become NULL on the
    # matched rows (incoming wins, full row replaced)
    # ------------------------------------------------------------------

    def test_upsert_missing_columns_become_null_on_match(self) -> None:
        """MERGE ... WHEN MATCHED THEN UPDATE SET T.col = S.col touches
        every non-key column. A source missing ``amount`` / ``active``
        therefore overwrites the matched row's payload columns with
        the NULL the staging cast filled in — the documented MERGE
        semantics for callers that didn't scope updates with
        ``update_column_names``."""
        table = self._unique_table("fill_upsert")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        # id=2 overlaps → updated; id=3 is new → inserted. Both should
        # land with ``amount = NULL`` and ``active = NULL`` because the
        # source doesn't carry those columns.
        partial = pa.table(
            {
                "id": pa.array([2, 3], type=pa.int64()),
                "label": pa.array(["beta-2", "gamma"], type=pa.string()),
            }
        )
        table.insert(partial, mode=Mode.UPSERT, match_by=["id"])

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                {"id": 2, "label": "beta-2", "amount": None, "active": None},
                {"id": 3, "label": "gamma", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # UPSERT + update_column_names — protect target-only payload columns
    # ------------------------------------------------------------------

    def test_upsert_update_column_names_protects_target_only_columns(self) -> None:
        """``update_column_names=['label']`` restricts the MERGE UPDATE
        SET clause to just ``label``. Matched rows keep their existing
        ``amount`` / ``active`` even though the source is missing them;
        unmatched rows still land with the staged NULL fills."""
        table = self._unique_table("fill_upsert_scoped")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        partial = pa.table(
            {
                "id": pa.array([2, 3], type=pa.int64()),
                "label": pa.array(["beta-2", "gamma"], type=pa.string()),
            }
        )
        table.insert(
            partial,
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=["label"],
        )

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                # id=2: label updated; amount/active preserved by scope.
                {"id": 2, "label": "beta-2", "amount": 20.0, "active": False},
                # id=3: new row → NULL fill stands.
                {"id": 3, "label": "gamma", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND + match_by — insert-only MERGE drops keyed dupes; survivors
    # carry the NULL fill for missing columns
    # ------------------------------------------------------------------

    def test_keyed_append_fills_missing_columns_for_new_rows(self) -> None:
        """Insert-only MERGE: existing keys are skipped, new keys land
        with the staged NULL fills for columns the source omitted."""
        table = self._unique_table("fill_append_keyed")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, True], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        partial = pa.table(
            {
                # id=2 already exists → skipped (existing row kept).
                # id=5 is new → inserted with NULL amount / active.
                "id": pa.array([2, 5], type=pa.int64()),
                "label": pa.array(["beta-DROPPED", "epsilon"], type=pa.string()),
            }
        )
        table.insert(partial, mode=Mode.APPEND, match_by=["id"])

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                # Original row wins on key conflict.
                {"id": 2, "label": "beta", "amount": 20.0, "active": True},
                # New row carries the staged NULL fill.
                {"id": 5, "label": "epsilon", "amount": None, "active": None},
            ],
        )


# =====================================================================
# Concurrent writes
# =====================================================================


class TestSQLConcurrentWrites(_SQLIntegrationBase):
    """Drive the same Delta target from N threads at once.

    Delta serializes commits at the table level — each writer that
    races a successful commit retries against the latest snapshot, so
    the final state must match a serial run of the same workload.
    These tests assert that contract through the public ``insert``
    surface: no lost rows on disjoint appends, no duplicate keys on
    overlapping upserts.
    """

    PARALLELISM: ClassVar[int] = 4
    ROWS_PER_WRITER: ClassVar[int] = 25

    @staticmethod
    def _writer_chunk(writer_id: int, n: int, *, key_offset: int = 0) -> pa.Table:
        """Build a deterministic chunk for *writer_id*.

        ``key_offset`` lets two writers either own disjoint keys
        (offset = writer_id * n) or share keys (offset = 0).
        """
        ids = list(range(key_offset, key_offset + n))
        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "writer": pa.array([writer_id] * n, type=pa.int32()),
                "amount": pa.array(
                    [float(writer_id * 1000 + i) for i in range(n)],
                    type=pa.float64(),
                ),
            }
        )

    @staticmethod
    def _empty_schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("writer", pa.int32()),
                pa.field("amount", pa.float64()),
            ]
        )

    def _run_in_parallel(self, fn, args_iter):
        """Run ``fn`` concurrently and re-raise the first exception."""
        with cf.ThreadPoolExecutor(max_workers=self.PARALLELISM) as pool:
            futures = [pool.submit(fn, *args) for args in args_iter]
            for fut in cf.as_completed(futures):
                fut.result()  # surface exceptions from the workers

    # ------------------------------------------------------------------
    # Disjoint appends — no row should be lost
    # ------------------------------------------------------------------

    def test_concurrent_appends_disjoint_keys_preserve_all_rows(self) -> None:
        table = self._unique_table("concurrent_append")
        self._ensure_table(table, self._empty_schema())

        def append(writer_id: int) -> None:
            chunk = self._writer_chunk(
                writer_id,
                self.ROWS_PER_WRITER,
                key_offset=writer_id * self.ROWS_PER_WRITER,
            )
            table.insert(chunk, mode=Mode.APPEND)

        self._run_in_parallel(
            append, [(i,) for i in range(self.PARALLELISM)],
        )

        count = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(count, self.PARALLELISM * self.ROWS_PER_WRITER)

        per_writer = self.engine.execute(
            f"SELECT writer, COUNT(*) AS n FROM {table.full_name(safe=True)} "
            "GROUP BY writer ORDER BY writer"
        ).to_arrow_table().to_pylist()
        self.assertEqual(
            per_writer,
            [{"writer": i, "n": self.ROWS_PER_WRITER} for i in range(self.PARALLELISM)],
        )

    # ------------------------------------------------------------------
    # Overlapping upserts — final state has one row per key
    # ------------------------------------------------------------------

    def test_concurrent_upserts_overlapping_keys_no_duplicates(self) -> None:
        """Every writer upserts the same key range. After all commits
        land, every key must be present exactly once and the surviving
        ``writer`` value is one of the writer ids in [0, PARALLELISM)."""
        table = self._unique_table("concurrent_upsert")
        self._ensure_table(table, self._empty_schema())

        # Seed so the first MERGE has a non-empty target — exercises the
        # WHEN MATCHED branch on at least one writer.
        seed = self._writer_chunk(-1, self.ROWS_PER_WRITER, key_offset=0)
        table.insert(seed, mode=Mode.OVERWRITE)

        def upsert(writer_id: int) -> None:
            chunk = self._writer_chunk(
                writer_id, self.ROWS_PER_WRITER, key_offset=0,
            )
            table.insert(chunk, mode=Mode.UPSERT, match_by=["id"])

        self._run_in_parallel(
            upsert, [(i,) for i in range(self.PARALLELISM)],
        )

        rows = self.engine.execute(
            f"SELECT id, writer FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

        self.assertEqual(len(rows), self.ROWS_PER_WRITER)
        self.assertEqual(
            [r["id"] for r in rows], list(range(self.ROWS_PER_WRITER)),
        )
        valid_writers = set(range(self.PARALLELISM))
        for r in rows:
            self.assertIn(
                r["writer"], valid_writers,
                f"row {r!r} kept writer={r['writer']!r} which never wrote",
            )

    # ------------------------------------------------------------------
    # Mixed APPEND + UPSERT — neither writer should lose rows
    # ------------------------------------------------------------------

    def test_concurrent_mixed_append_and_upsert(self) -> None:
        """One writer appends a disjoint key range; another upserts an
        overlapping range. After both commit, the appender's keys are
        all present and the upsert range has exactly one row per key."""
        table = self._unique_table("concurrent_mixed")
        self._ensure_table(table, self._empty_schema())

        # Seed with keys [0, ROWS_PER_WRITER) so the upsert path has
        # something to update.
        seed = self._writer_chunk(0, self.ROWS_PER_WRITER, key_offset=0)
        table.insert(seed, mode=Mode.OVERWRITE)

        def appender() -> None:
            chunk = self._writer_chunk(
                100, self.ROWS_PER_WRITER,
                key_offset=10 * self.ROWS_PER_WRITER,
            )
            table.insert(chunk, mode=Mode.APPEND)

        def upserter() -> None:
            chunk = self._writer_chunk(
                200, self.ROWS_PER_WRITER, key_offset=0,
            )
            table.insert(chunk, mode=Mode.UPSERT, match_by=["id"])

        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(appender), pool.submit(upserter)]
            for fut in cf.as_completed(futures):
                fut.result()

        # Total rows = original seed (all upserted in place) + appended chunk.
        count = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(count, 2 * self.ROWS_PER_WRITER)

        # Upsert range survived as a single row per key, all written by 200.
        upserted = self.engine.execute(
            f"SELECT writer FROM {table.full_name(safe=True)} "
            f"WHERE id < {self.ROWS_PER_WRITER}"
        ).to_arrow_table().column("writer").to_pylist()
        self.assertEqual(len(upserted), self.ROWS_PER_WRITER)
        self.assertTrue(all(w == 200 for w in upserted))

        # Append range fully landed.
        appended = self.engine.execute(
            f"SELECT COUNT(*) AS n FROM {table.full_name(safe=True)} "
            f"WHERE id >= {10 * self.ROWS_PER_WRITER}"
        ).to_arrow_table().column("n")[0].as_py()
        self.assertEqual(appended, self.ROWS_PER_WRITER)
