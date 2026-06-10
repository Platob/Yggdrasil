"""Core :class:`SQLEngine` + :class:`Table` round-trips (create / insert / read / DDL).

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

import secrets

from databricks.sdk.errors import NotFound
import pyarrow as pa

from yggdrasil.databricks.table.table import Table
from yggdrasil.enums import Mode

from ._sql_integration import SQLIntegrationCase


class TestSQLEngineIntegration(SQLIntegrationCase):
    """Engine + table CRUD against a real workspace."""

    # ------------------------------------------------------------------
    # Catalog / schema lifecycle
    # ------------------------------------------------------------------

    def test_catalog_and_schema_exist(self) -> None:
        catalog = self.engine.catalogs.catalog(self.catalog_name)
        self.assertTrue(catalog.exists())

        schema = self.engine.schemas.schema(
            f"{self.catalog_name}.{self.schema_name}"
        )
        self.assertTrue(schema.exists())

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

    def test_execute_many(self):
        batch = self.engine.execute_many(
            ["SELECT 1 AS one, 'hello' AS greeting" for _ in range(4)],
            parallel=True
        )

        assert batch.done

        table = batch.to_arrow_table()
        self.assertEqual(table.num_rows, 4)
        self.assertEqual(table.column("one").to_pylist(), [1, 1, 1, 1])
        self.assertEqual(table.column("greeting").to_pylist(), ["hello", "hello", "hello", "hello"])

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
        self.assertFalse(table.exists())

        created = self.engine.create_table(
            self._sample_schema(),
            full_name=table.full_name(),
        )
        self.assertTrue(created.exists())

        # Idempotent: a second create_table call with missing_ok=True
        # (the engine default) must not raise.
        again = self.engine.create_table(
            self._sample_schema(),
            full_name=table.full_name(),
        )
        self.assertTrue(again.exists())

    # ------------------------------------------------------------------
    # Table.create / Table.ensure_created
    # ------------------------------------------------------------------

    def test_table_ensure_created_is_idempotent(self) -> None:
        table = self._unique_table("ensure")
        table.ensure_created(self._sample_schema())
        self.assertTrue(table.exists())

        # Second call is a no-op (or schema-merge no-op) on an existing
        # table with the same definition.
        table.ensure_created(self._sample_schema())
        self.assertTrue(table.exists())

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
        self.assertTrue(table.exists())

        table.delete()
        # Re-resolve through the service so we don't read a stale cached
        # ``_infos`` from the in-memory handle that just dropped itself.
        self.assertFalse(self.engine.table(table.full_name()).exists())

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
            client=self.client,
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
            client=self.client,
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
            import databricks.connect  # noqa: F401
        except ImportError:
            self.skipTest(
                "pyspark / databricks-connect not installed — Spark-mode "
                "execution unavailable in this environment"
            )
        if self.spark is None:
            self.skipTest("no Spark session available for this workspace")

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
            import databricks.connect  # noqa: F401
        except ImportError:
            self.skipTest(
                "pyspark / databricks-connect not installed — Spark-mode "
                "execution unavailable in this environment"
            )
        if self.spark is None:
            self.skipTest("no Spark session available for this workspace")

        from yggdrasil.enums import Mode
        from yggdrasil.enums.media_type import MediaTypes

        data = pa.table(
            {
                "id": pa.array([42, 43], type=pa.int64()),
                "label": pa.array(["m", "n"], type=pa.string()),
            }
        )
        # Stage the volume up-front through the table's staging path so
        # the caller can hand the engine a ready-to-read VolumePath.
        table = Table(
            service=self.client.tables,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name="engine_spark_volume",
        )
        path = table.insert_volume_path(temporary=False)
        path.as_media(media_type=MediaTypes.PARQUET).write_table(
            data, mode=Mode.OVERWRITE,
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

    def test_table_insert_volume_path_writes_parquet(self) -> None:
        """``Table.insert_volume_path`` mints the staging path; writing
        the Parquet through ``as_media(PARQUET).write_table`` is the
        two-step pattern the warehouse insert path drives — verify the
        warehouse can read the file back."""
        from yggdrasil.enums import Mode
        from yggdrasil.enums.media_type import MediaTypes

        data = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )
        table = Table(
            service=self.client.tables,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name="staging_factory",
        )
        path = table.insert_volume_path(temporary=False)
        path.as_media(media_type=MediaTypes.PARQUET).write_table(
            data, mode=Mode.OVERWRITE,
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
            _ = self.engine.table(ghost).infos  # noqa: B018 — assert raises

        # ``drop_table`` swallows the NotFound and returns cleanly.
        self.engine.drop_table(ghost, raise_error=False)


# =====================================================================
# MERGE strategy
# =====================================================================
