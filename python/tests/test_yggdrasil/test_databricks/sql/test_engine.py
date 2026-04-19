import datetime as dt
import unittest
from dataclasses import dataclass, field

import pyarrow as pa
import pytest

from yggdrasil.data import Schema
from yggdrasil.data.statement import StatementResult as BaseStatementResult
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.sql.engine import (
    _apply_external_table_aliases,
    _build_match_condition,
    _delta_cluster_columns,
    _delta_partition_columns,
    _is_concurrent_append,
    _narrow_target_columns,
    _narrowing_predicate_from_values,
    _narrowing_predicates_from_polars,
    _narrowing_predicates_via_subquery,
    _retry_concurrent_append,
    _staging_parquet_ref,
)
from yggdrasil.databricks.sql.exceptions import SQLError
from yggdrasil.databricks.sql.staging import StagingPath
from ..conftest import DatabricksCase, requires_databricks

integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Unit tests — no live workspace required
# ---------------------------------------------------------------------------


class TestExternalTableAliasing(unittest.TestCase):
    """Pure-function coverage for the alias substitution helpers."""

    def test_no_substitutions_returns_statement_unchanged(self):
        stmt = "SELECT * FROM t"
        self.assertIs(_apply_external_table_aliases(stmt, {}), stmt)

    def test_replaces_multiple_aliases(self):
        stmt = "SELECT * FROM {left} JOIN {right} ON {left}.id = {right}.id"
        out = _apply_external_table_aliases(
            stmt,
            {"left": "parquet.`/a.parquet`", "right": "parquet.`/b.parquet`"},
        )
        self.assertNotIn("{left}", out)
        self.assertNotIn("{right}", out)
        self.assertIn("parquet.`/a.parquet`", out)
        self.assertIn("parquet.`/b.parquet`", out)

    def test_missing_alias_leaves_placeholder_intact(self):
        stmt = "SELECT * FROM {foo} WHERE {bar} > 1"
        out = _apply_external_table_aliases(stmt, {"foo": "X"})
        self.assertEqual(out, "SELECT * FROM X WHERE {bar} > 1")


class _FakeCleanupResource:
    """Minimal staging-like object to drive cleanup semantics under test."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    def cleanup(self, *, allow_not_found: bool = True) -> None:
        self.calls.append(allow_not_found)


@dataclass
class _FakeStatement(BaseStatementResult):
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

    def start(self, *, wait=True, raise_error=True, **_kwargs) -> "_FakeStatement":
        return self

    def cancel(self) -> "_FakeStatement":
        return self

    def collect_schema(self, full=False) -> Schema:
        return Schema.from_any_fields([], metadata={})

    def to_arrow_reader(self, **_: object) -> pa.RecordBatchReader:  # pragma: no cover
        raise NotImplementedError


class TestSqlInsertIntoSmartDispatch(unittest.TestCase):
    """``sql_insert_into`` should route cached/Spark/warehouse paths correctly."""

    def _make_engine(self):
        from unittest.mock import MagicMock
        engine = MagicMock(spec=["sql_insert_into", "insert_into", "spark_insert_into"])
        # Use the real sql_insert_into wired onto the mock.
        from yggdrasil.databricks.sql.engine import SQLEngine
        engine.sql_insert_into = SQLEngine.sql_insert_into.__get__(engine)
        engine.catalog_name = None
        engine.schema_name = None
        return engine

    def test_cached_spark_df_is_reused(self):
        from unittest.mock import MagicMock, patch
        from yggdrasil.databricks.sql.statement import StatementResult
        from yggdrasil.data.statement import PreparedStatement

        engine = self._make_engine()
        result = StatementResult(statement=PreparedStatement(text="SELECT 1"))
        spark_df = MagicMock()
        object.__setattr__(result, "_spark_df", spark_df)

        with patch("yggdrasil.environ.PyEnv.spark_session", return_value=None):
            engine.sql_insert_into(result, location="main.s.t")

        engine.insert_into.assert_called_once()
        kwargs = engine.insert_into.call_args.kwargs
        self.assertIs(kwargs["data"], spark_df)

    def test_spark_session_runs_sql_and_dispatches_to_spark_insert(self):
        from unittest.mock import MagicMock, patch
        from yggdrasil.data.statement import PreparedStatement

        engine = self._make_engine()
        stmt = PreparedStatement(text="SELECT 1")

        spark_session = MagicMock()
        df = MagicMock()
        spark_session.sql.return_value = df

        with patch(
            "yggdrasil.environ.PyEnv.spark_session",
            return_value=spark_session,
        ):
            engine.sql_insert_into(stmt, location="main.s.t")

        spark_session.sql.assert_called_once_with("SELECT 1")
        engine.spark_insert_into.assert_called_once()
        kwargs = engine.spark_insert_into.call_args.kwargs
        self.assertIs(kwargs["data"], df)


class TestExternalTableCleanup(unittest.TestCase):
    """Behavior of ``attach_external_tables`` / lazy cleanup on base result."""

    def test_cleanup_skipped_while_not_done(self):
        result = _FakeStatement()
        resource = _FakeCleanupResource()
        result.attach_external_tables([resource])

        result._maybe_cleanup_external_tables()
        self.assertEqual(resource.calls, [])
        self.assertFalse(result._external_tables_cleaned)

    def test_cleanup_runs_once_when_done(self):
        resource_a = _FakeCleanupResource()
        resource_b = _FakeCleanupResource()
        result = _FakeStatement()
        result.attach_external_tables([resource_a, resource_b])

        object.__setattr__(result, "_fake_done", True)

        result._maybe_cleanup_external_tables()
        result._maybe_cleanup_external_tables()

        self.assertEqual(resource_a.calls, [True])
        self.assertEqual(resource_b.calls, [True])
        self.assertTrue(result._external_tables_cleaned)
        self.assertEqual(result._external_tables, ())

    def test_cleanup_swallows_per_resource_errors(self):
        class Broken:
            def cleanup(self, *, allow_not_found: bool = True) -> None:
                raise RuntimeError("boom")

        ok = _FakeCleanupResource()
        result = _FakeStatement()
        result.attach_external_tables([Broken(), ok])
        object.__setattr__(result, "_fake_done", True)

        result._maybe_cleanup_external_tables()

        self.assertEqual(ok.calls, [True])
        self.assertTrue(result._external_tables_cleaned)

    def test_wait_triggers_cleanup(self):
        result = _FakeStatement()
        resource = _FakeCleanupResource()
        result.attach_external_tables([resource])
        object.__setattr__(result, "_fake_done", True)

        result.wait(wait=False)

        self.assertEqual(resource.calls, [True])

    def test_attach_no_tables_is_noop(self):
        result = _FakeStatement()
        result.attach_external_tables([])
        self.assertEqual(result._external_tables, ())
        self.assertFalse(result._external_tables_cleaned)


class TestDeltaConcurrentAppendNarrowing(unittest.TestCase):
    """Coverage for the DELTA_CONCURRENT_APPEND.WHOLE_TABLE_READ mitigation."""

    def test_build_match_condition_appends_extra_predicates(self):
        on_clause = _build_match_condition(
            ["id"],
            left_alias="T",
            right_alias="S",
            extra_predicates=["T.`dt` IN (SELECT DISTINCT `dt` FROM {src})"],
        )
        self.assertIn("T.`id` <=> S.`id`", on_clause)
        self.assertIn(
            "T.`dt` IN (SELECT DISTINCT `dt` FROM {src})",
            on_clause,
        )
        self.assertTrue(on_clause.count(" AND ") == 1)

    def test_build_match_condition_skips_falsy_extras(self):
        on_clause = _build_match_condition(
            ["id"],
            left_alias="T",
            right_alias="S",
            extra_predicates=["", None, "T.`dt` IS NOT NULL"],  # type: ignore[list-item]
        )
        self.assertEqual(
            on_clause, "T.`id` <=> S.`id` AND T.`dt` IS NOT NULL"
        )

    def test_narrowing_predicate_from_values_emits_in_list(self):
        pred = _narrowing_predicate_from_values(
            "dt", ["2026-04-01", "2026-04-02", None],
            target_alias="T", max_in_values=500,
        )
        assert pred == "(T.`dt` IN ('2026-04-01', '2026-04-02') OR T.`dt` IS NULL)"

    def test_narrowing_predicate_from_values_falls_back_to_between(self):
        values = list(range(600))
        pred = _narrowing_predicate_from_values(
            "dt", values,
            target_alias="T", max_in_values=500,
        )
        assert pred == "T.`dt` BETWEEN 0 AND 599"

    def test_narrowing_predicate_from_values_all_null_returns_is_null(self):
        pred = _narrowing_predicate_from_values(
            "dt", [None, None], target_alias="T", max_in_values=500,
        )
        assert pred == "T.`dt` IS NULL"

    def test_narrowing_predicate_from_values_empty_returns_none(self):
        assert _narrowing_predicate_from_values(
            "dt", [], target_alias="T", max_in_values=500,
        ) is None

    def test_narrowing_predicates_from_polars_reads_distinct_values(self):
        import polars as pl

        frame = pl.DataFrame(
            {
                "dt": ["2026-04-01", "2026-04-02", "2026-04-01"],
                "region": ["EU", "EU", "US"],
            }
        )

        predicates = _narrowing_predicates_from_polars(
            frame, ["dt", "region"], target_alias="T",
        )

        assert len(predicates) == 2
        assert "T.`dt` IN (" in predicates[0]
        assert "'2026-04-01'" in predicates[0]
        assert "'2026-04-02'" in predicates[0]
        assert "T.`region` IN (" in predicates[1]

    def test_narrowing_predicates_via_subquery_produces_in_clause(self):
        predicates = _narrowing_predicates_via_subquery(
            ["dt", "region"],
            target_alias="T",
            source_expr="{src}",
        )
        self.assertEqual(
            predicates,
            [
                "T.`dt` IN (SELECT DISTINCT `dt` FROM {src})",
                "T.`region` IN (SELECT DISTINCT `region` FROM {src})",
            ],
        )

    def test_delta_partition_columns_reads_from_table_info(self):
        class _Col:
            def __init__(self, name, idx=None):
                self.name = name
                self.partition_index = idx

        class _Info:
            columns = [
                _Col("id", None),
                _Col("dt", 0),
                _Col("region", 1),
                _Col("value", None),
            ]

        class _Table:
            infos = _Info()

            def full_name(self, safe=False):
                return "c.s.t"

        self.assertEqual(_delta_partition_columns(_Table()), ["dt", "region"])

    def test_delta_cluster_columns_parses_tbl_properties_json(self):
        class _Info:
            properties = {
                "clusteringColumns": '[["dt"], ["region"]]',
                "other": "value",
            }

        class _Table:
            infos = _Info()

            def full_name(self, safe=False):
                return "c.s.t"

        self.assertEqual(_delta_cluster_columns(_Table()), ["dt", "region"])

    def test_delta_cluster_columns_handles_missing_or_bad_props(self):
        class _Info:
            properties = {"clusteringColumns": "not-json"}

        class _Table:
            infos = _Info()

            def full_name(self, safe=False):
                return "c.s.t"

        self.assertEqual(_delta_cluster_columns(_Table()), [])

    def test_narrow_target_columns_skips_match_keys_and_dedupes(self):
        class _Col:
            def __init__(self, name, idx=None):
                self.name = name
                self.partition_index = idx

        class _Info:
            columns = [_Col("id"), _Col("dt", 0), _Col("region", 1)]
            properties = {"clusteringColumns": '[["dt"], ["bucket"]]'}

        class _Table:
            infos = _Info()

            def full_name(self, safe=False):
                return "c.s.t"

        cols = _narrow_target_columns(_Table(), match_by=["id", "region"])
        # Partitions first (minus ``region`` which is a match key),
        # then cluster columns (``dt`` deduped, ``bucket`` kept).
        self.assertEqual(cols, ["dt", "bucket"])

    def test_is_concurrent_append_matches_delta_error_text(self):
        class _Err(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self.error_code = "INTERNAL_ERROR"
                self.message = msg

        exc = _Err(
            "[DELTA_CONCURRENT_APPEND.WHOLE_TABLE_READ] Transaction conflict"
        )
        self.assertTrue(_is_concurrent_append(exc))

        self.assertFalse(
            _is_concurrent_append(RuntimeError("something totally unrelated"))
        )

        class _SparkLike(Exception):
            pass

        _SparkLike.__name__ = "ConcurrentAppendException"
        self.assertTrue(_is_concurrent_append(_SparkLike("boom")))

    def test_retry_concurrent_append_retries_then_succeeds(self):
        class _Err(Exception):
            error_code = "CONCURRENT_DELTA_TABLE_WRITE"
            message = "DELTA_CONCURRENT_APPEND"

        calls = {"n": 0}

        def _op() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise _Err("DELTA_CONCURRENT_APPEND transient")
            return "ok"

        from unittest.mock import patch

        with patch("yggdrasil.databricks.sql.engine.time.sleep"):
            out = _retry_concurrent_append(_op, attempts=5, base_delay=0.0)

        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    def test_retry_concurrent_append_propagates_other_errors(self):
        def _op() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            _retry_concurrent_append(_op, attempts=3, base_delay=0.0)

    def test_retry_concurrent_append_reraises_after_last_attempt(self):
        class _Err(Exception):
            error_code = "DELTA_CONCURRENT_APPEND"
            message = "DELTA_CONCURRENT_APPEND keeps losing"

        def _op() -> None:
            raise _Err("DELTA_CONCURRENT_APPEND keeps losing")

        from unittest.mock import patch

        with patch("yggdrasil.databricks.sql.engine.time.sleep"):
            with self.assertRaises(_Err):
                _retry_concurrent_append(_op, attempts=2, base_delay=0.0)


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
class TestSQLEngineExternalTables(DatabricksCase):
    """External-table staging through ``execute`` and ``execute_many``."""

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
            external_tables={"src": data},
            engine="api",
        )

        out = result.to_arrow_table()
        self.assertEqual(out.num_rows, data.num_rows)
        self.assertEqual(out.column("id").to_pylist(), [1, 2, 3])

        # Staging is owned by the engine and cleaned up once the result is done.
        self.assertTrue(result.done)
        self.assertTrue(result._external_tables_cleaned)
        self.assertEqual(result._external_tables, ())

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
                external_tables={"src": staging},
                engine="api",
            )
            row = result.to_arrow_table().to_pylist()[0]
            self.assertEqual(row["n"], data.num_rows)

            # User-owned StagingPaths are not attached for cleanup.
            self.assertEqual(result._external_tables, ())

            # The staging file is still present because the engine did not own it.
            self.assertTrue(staging.path.exists())
        finally:
            staging.cleanup(allow_not_found=True)

    def test_execute_many_shares_external_tables(self):
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
            external_tables={"left": left, "right": right},
            engine="api",
        )

        first, second = list(batch.results.values())
        self.assertEqual(first.to_arrow_table().to_pylist()[0]["n"], left.num_rows)

        joined = second.to_arrow_table()
        self.assertEqual(joined.num_rows, 2)
        self.assertEqual(joined.column("id").to_pylist(), [2, 3])

        # Staging is cleaned up after the batch resolves.
        for result in batch.results.values():
            self.assertTrue(result._external_tables_cleaned)

    def test_execute_many_parallel_cleans_up(self):
        left = self._make_left()

        batch = self.engine.execute_many(
            {
                "q1": "SELECT COUNT(*) AS n FROM {src}",
                "q2": "SELECT MAX(id) AS m FROM {src}",
            },
            external_tables={"src": left},
            parallel=True,
            engine="api",
        )

        self.assertEqual(batch["q1"].to_arrow_table().to_pylist()[0]["n"], left.num_rows)
        self.assertEqual(batch["q2"].to_arrow_table().to_pylist()[0]["m"], 3)

        for result in batch.results.values():
            self.assertTrue(result._external_tables_cleaned)

    def test_external_tables_requires_catalog_and_schema(self):
        bare_engine = self.workspace.sql()  # no catalog/schema bound

        with self.assertRaises(ValueError):
            bare_engine.execute(
                "SELECT * FROM {src}",
                external_tables={"src": self._make_left()},
                engine="api",
            )
