import datetime as dt
import unittest
from dataclasses import dataclass, field

import pyarrow as pa
import pytest
from yggdrasil.data.statement import Statement as BaseStatement

from yggdrasil.data import Schema
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.sql.engine import (
    _apply_temporary_table_aliases,
    _staging_parquet_ref,
)
from yggdrasil.databricks.sql.exceptions import SQLError
from yggdrasil.databricks.sql.staging import StagingPath
from ..conftest import DatabricksCase, requires_databricks

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Unit tests — no live workspace required
# ---------------------------------------------------------------------------


class TestTemporaryTableAliasing(unittest.TestCase):
    """Pure-function coverage for the alias substitution helpers."""

    def test_no_substitutions_returns_statement_unchanged(self):
        stmt = "SELECT * FROM t"
        self.assertIs(_apply_temporary_table_aliases(stmt, {}), stmt)

    def test_replaces_multiple_aliases(self):
        stmt = "SELECT * FROM {left} JOIN {right} ON {left}.id = {right}.id"
        out = _apply_temporary_table_aliases(
            stmt,
            {"left": "parquet.`/a.parquet`", "right": "parquet.`/b.parquet`"},
        )
        self.assertNotIn("{left}", out)
        self.assertNotIn("{right}", out)
        self.assertIn("parquet.`/a.parquet`", out)
        self.assertIn("parquet.`/b.parquet`", out)

    def test_missing_alias_leaves_placeholder_intact(self):
        stmt = "SELECT * FROM {foo} WHERE {bar} > 1"
        out = _apply_temporary_table_aliases(stmt, {"foo": "X"})
        self.assertEqual(out, "SELECT * FROM X WHERE {bar} > 1")


class _FakeCleanupResource:
    """Minimal staging-like object to drive cleanup semantics under test."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def cleanup(self, *, allow_not_found: bool = True) -> None:
        self.calls.append(allow_not_found)


@dataclass
class _FakeStatement(BaseStatement):
    """Concrete ``StatementResult`` used to exercise base-class behavior."""

    _fake_done: bool = field(default=False)

    @property
    def done(self) -> bool:
        return self._fake_done

    @property
    def failed(self) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def refresh_status(self) -> None:
        return None

    def collect_schema(self, full=False) -> Schema:
        return Schema.from_any_fields([], metadata={})

    def to_arrow_reader(self, **_: object) -> pa.RecordBatchReader:  # pragma: no cover
        raise NotImplementedError


class TestTemporaryTableCleanup(unittest.TestCase):
    """Behavior of ``attach_temporary_tables`` / lazy cleanup on base result."""

    def test_cleanup_skipped_while_not_done(self):
        result = _FakeStatement()
        resource = _FakeCleanupResource()
        result.attach_temporary_tables([resource])

        result._maybe_cleanup_temporary_tables()
        self.assertEqual(resource.calls, [])
        self.assertFalse(result._temporary_tables_cleaned)

    def test_cleanup_runs_once_when_done(self):
        resource_a = _FakeCleanupResource()
        resource_b = _FakeCleanupResource()
        result = _FakeStatement()
        result.attach_temporary_tables([resource_a, resource_b])

        object.__setattr__(result, "_fake_done", True)

        result._maybe_cleanup_temporary_tables()
        result._maybe_cleanup_temporary_tables()

        self.assertEqual(resource_a.calls, [True])
        self.assertEqual(resource_b.calls, [True])
        self.assertTrue(result._temporary_tables_cleaned)
        self.assertEqual(result._temporary_tables, ())

    def test_cleanup_swallows_per_resource_errors(self):
        class Broken:
            def cleanup(self, *, allow_not_found: bool = True) -> None:
                raise RuntimeError("boom")

        ok = _FakeCleanupResource()
        result = _FakeStatement()
        result.attach_temporary_tables([Broken(), ok])
        object.__setattr__(result, "_fake_done", True)

        result._maybe_cleanup_temporary_tables()

        self.assertEqual(ok.calls, [True])
        self.assertTrue(result._temporary_tables_cleaned)

    def test_wait_triggers_cleanup(self):
        result = _FakeStatement()
        resource = _FakeCleanupResource()
        result.attach_temporary_tables([resource])
        object.__setattr__(result, "_fake_done", True)

        result.wait(wait=False)

        self.assertEqual(resource.calls, [True])

    def test_attach_no_tables_is_noop(self):
        result = _FakeStatement()
        result.attach_temporary_tables([])
        self.assertEqual(result._temporary_tables, ())
        self.assertFalse(result._temporary_tables_cleaned)


class TestStagingParquetRef(unittest.TestCase):
    """Unit-level check on the parquet source-clause formatting."""

    def test_wraps_path_in_backticked_parquet_ref(self):
        class _FakePath:
            def __str__(self) -> str:
                return "/Volumes/c/s/tmp/.sql/c/s/t/tmp-0-1-abc.parquet"

        path = _FakePath()
        ref = _staging_parquet_ref(
            StagingPath(
                path=path,  # type: ignore[arg-type]
                catalog_name="c",
                schema_name="s",
                table_name="t",
                start_ts=0,
                end_ts=1,
                token="abc",
            )
        )
        self.assertTrue(ref.startswith("parquet.`"))
        self.assertIn("/Volumes/c/s/tmp/.sql/c/s/t/tmp-0-1-abc.parquet", ref)
        self.assertTrue(ref.endswith("`"))


# ---------------------------------------------------------------------------
# Integration tests — require a live Databricks workspace
# ---------------------------------------------------------------------------


@requires_databricks
@integration
class TestSQLEngine(DatabricksCase):
    """End-to-end tests that hit a real Databricks warehouse."""

    engine: SQLEngine

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

    @staticmethod
    def _sample_table() -> pa.Table:
        return pa.table(
            [
                pa.array(["a", None, "c"]),
                pa.array([1, 2, 4]),
                pa.array(
                    [{"q": dt.datetime.now(dt.timezone.utc), "v": 1.0}, None, None]
                ),
                pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                pa.array(
                    [{"k": "v"}, None, None],
                    type=pa.map_(pa.string(), pa.string()),
                ),
            ],
            names=["c0", "id", "c2", "c3", "map column"],
        )

    def test_insert_then_read_roundtrip(self):
        data = self._sample_table()
        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        try:
            handle = self.engine.table(table_name="test_insert")
            result = self.engine.execute(f"SELECT * FROM {handle.full_name(safe=True)}")

            self.assertEqual(result.to_arrow_table().num_rows, data.num_rows)
            self.assertEqual(result.to_arrow_dataset().to_table().num_rows, data.num_rows)
            self.assertEqual(result.to_polars(stream=False).shape[0], data.num_rows)
        finally:
            self.engine.drop_table(table_name="test_insert")

    def test_insert_schema_evolution_append(self):
        data = self._sample_table()
        self.engine.insert_into(data, table_name="test_insert", mode="overwrite")

        try:
            extra = pa.table(
                [
                    pa.array(["1", "2", "4"]),
                    pa.array([{"q": dt.datetime.now(dt.timezone.utc)}, None, None]),
                    pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                    pa.array(
                        [{"k": "v"}, None, None],
                        type=pa.map_(pa.string(), pa.string()),
                    ),
                ],
                names=["id", "new column", "c3", "map column"],
            )

            self.engine.insert_into(
                extra,
                table_name="test_insert",
                mode="append",
                schema_mode="append",
            )

            handle = self.engine.table(table_name="test_insert")
            result = self.engine.execute(f"SELECT * FROM {handle.full_name(safe=True)}")
            self.assertEqual(
                result.to_arrow_table().num_rows,
                data.num_rows + extra.num_rows,
            )
        finally:
            self.engine.drop_table(table_name="test_insert")

    def test_execute_unknown_table_raises(self):
        with pytest.raises(SQLError):
            self.engine.execute("SELECT * FROM unknown_table")

    def test_warehouse_api_create_drop(self):
        data = pa.table(
            [
                pa.array(["a", None, "c"]),
                pa.array([1, 2, 4]),
                pa.array([{"q": dt.datetime.now()}, None, None]),
                pa.array([[{"list_nest": dt.datetime.now()}], None, None]),
                pa.array(
                    [{"k": "v"}, None, None],
                    type=pa.map_(pa.string(), pa.string()),
                ),
            ],
            names=["c0", "c1", "c2", "c3", "map column"],
        )
        self.engine.create_table(data, table_name="test_warehouse_api")
        self.engine.drop_table(table_name="test_warehouse_api")

    def test_warehouse_crud(self):
        warehouses = self.workspace.warehouses()
        warehouse = None
        try:
            warehouse = warehouses.create(name="tmp warehouse", wait=False)
            self.assertEqual(warehouse.warehouse_name, "tmp warehouse")
        finally:
            if warehouse:
                warehouse.delete()


@requires_databricks
@integration
class TestSQLEngineTemporaryTables(DatabricksCase):
    """Temporary-table staging through ``execute`` and ``execute_many``."""

    engine: SQLEngine

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.engine = cls.workspace.sql(catalog_name="trading", schema_name="unittest")

    @staticmethod
    def _make_left() -> pa.Table:
        return pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "name": pa.array(["a", "b", "c"]),
            }
        )

    @staticmethod
    def _make_right() -> pa.Table:
        return pa.table(
            {
                "id": pa.array([2, 3, 4], type=pa.int64()),
                "score": pa.array([20.0, 30.0, 40.0]),
            }
        )

    def test_execute_with_tabular_data_substitutes_and_cleans(self):
        data = self._make_left()

        result = self.engine.execute(
            "SELECT id, name FROM {src} ORDER BY id",
            temporary_tables={"src": data},
            engine="api",
        )

        out = result.to_arrow_table()
        self.assertEqual(out.num_rows, data.num_rows)
        self.assertEqual(out.column("id").to_pylist(), [1, 2, 3])

        # Staging is owned by the engine and cleaned up once the result is done.
        self.assertTrue(result.done)
        self.assertTrue(result._temporary_tables_cleaned)
        self.assertEqual(result._temporary_tables, ())

    def test_execute_accepts_prebuilt_staging_path(self):
        data = self._make_left()

        staging = StagingPath.for_table(
            client=self.engine.client,
            catalog_name="trading",
            schema_name="unittest",
            table_name="test_temp_prebuilt",
        )
        staging.register_shutdown_cleanup()
        staging.write_table(data)

        try:
            result = self.engine.execute(
                "SELECT COUNT(*) AS n FROM {src}",
                temporary_tables={"src": staging},
                engine="api",
            )
            row = result.to_arrow_table().to_pylist()[0]
            self.assertEqual(row["n"], data.num_rows)

            # User-owned StagingPaths are not attached for cleanup.
            self.assertEqual(result._temporary_tables, ())

            # The staging file is still present because the engine did not own it.
            self.assertTrue(staging.path.exists())
        finally:
            staging.cleanup(allow_not_found=True)

    def test_execute_many_shares_temporary_tables(self):
        left = self._make_left()
        right = self._make_right()

        batch = self.engine.execute_many(
            [
                "SELECT COUNT(*) AS n FROM {left}",
                (
                    "SELECT l.id, l.name, r.score "
                    "FROM {left} AS l JOIN {right} AS r ON l.id = r.id "
                    "ORDER BY l.id"
                ),
            ],
            temporary_tables={"left": left, "right": right},
            engine="api",
        )

        first, second = list(batch.results.values())
        self.assertEqual(first.to_arrow_table().to_pylist()[0]["n"], left.num_rows)

        joined = second.to_arrow_table()
        self.assertEqual(joined.num_rows, 2)
        self.assertEqual(joined.column("id").to_pylist(), [2, 3])

        # Staging is cleaned up after the batch resolves.
        for result in batch.results.values():
            self.assertTrue(result._temporary_tables_cleaned)

    def test_execute_many_parallel_cleans_up(self):
        left = self._make_left()

        batch = self.engine.execute_many(
            {
                "q1": "SELECT COUNT(*) AS n FROM {src}",
                "q2": "SELECT MAX(id) AS m FROM {src}",
            },
            temporary_tables={"src": left},
            parallel=True,
            engine="api",
        )

        self.assertEqual(batch["q1"].to_arrow_table().to_pylist()[0]["n"], left.num_rows)
        self.assertEqual(batch["q2"].to_arrow_table().to_pylist()[0]["m"], 3)

        for result in batch.results.values():
            self.assertTrue(result._temporary_tables_cleaned)

    def test_temporary_tables_requires_catalog_and_schema(self):
        bare_engine = self.workspace.sql()  # no catalog/schema bound

        with self.assertRaises(ValueError):
            bare_engine.execute(
                "SELECT * FROM {src}",
                temporary_tables={"src": self._make_left()},
                engine="api",
            )
