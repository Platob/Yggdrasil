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

import os
import secrets
from typing import ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound

from yggdrasil.data.enums import Mode
from yggdrasil.databricks.sql.engine import SQLEngine
from yggdrasil.databricks.sql.table import Table

from .. import DatabricksIntegrationCase


__all__ = ["TestSQLEngineIntegration"]


class TestSQLEngineIntegration(DatabricksIntegrationCase):
    """Round-trip the SQL engine and per-table API against a real workspace.

    Catalog / schema default to ``trading.unittest``; the suite
    auto-creates them on first miss using ``ensure_created`` so the
    tests work against a clean workspace as well as one where the
    fixture is already provisioned.
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
