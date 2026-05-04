"""Live-Postgres integration tests for :class:`PostgresEngine`.

Skipped automatically when ``POSTGRES_URI`` is unset *or* the
optional ``psycopg`` / ``adbc-driver-postgresql`` packages are
missing — see :class:`yggdrasil.postgres.tests.PostgresTestCase`.

Run only this suite with::

    pytest -m postgres_integration tests/test_yggdrasil/test_postgres/

Or skip the whole live tier with::

    pytest -m "not postgres_integration"
"""

from __future__ import annotations

import pytest

from yggdrasil.postgres.tests import PostgresTestCase

pytestmark = pytest.mark.postgres_integration


class TestEngineExecute(PostgresTestCase):
    """``execute`` / ``sql`` round-trips against a live Postgres."""

    def test_execute_select_one(self) -> None:
        result = self.engine.execute("SELECT 1 AS one")
        table = result.read_arrow_table()
        self.assertEqual(table.column_names, ["one"])
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(table.column("one").to_pylist(), [1])

    def test_execute_with_parameters(self) -> None:
        result = self.engine.execute(
            "SELECT %s::int AS x, %s::text AS s",
            parameters=(7, "hello"),
        )
        rows = result.read_arrow_table().to_pylist()
        self.assertEqual(rows, [{"x": 7, "s": "hello"}])

    def test_execute_ddl_returns_empty_table(self) -> None:
        # DDL emits no result-set; we still return an Arrow shape so
        # callers don't have to special-case ``None``.
        result = self.engine.execute(
            f'CREATE TABLE "{self.test_schema_name}"."ddl_smoke" (id int)',
            prefer_arrow=False,
        )
        self.assertEqual(result.read_arrow_table().num_rows, 0)

    def test_execute_returns_zero_rows_on_empty_select(self) -> None:
        # Empty result-set should still expose the projected columns.
        result = self.engine.execute("SELECT 1::int AS x WHERE FALSE")
        table = result.read_arrow_table()
        self.assertEqual(table.column_names, ["x"])
        self.assertEqual(table.num_rows, 0)

    def test_sql_helper_roundtrip(self) -> None:
        result = self.engine.sql("SELECT 'pg'::text AS who")
        self.assertEqual(
            result.read_arrow_table().column("who").to_pylist(),
            ["pg"],
        )

    def test_psycopg_path_matches_arrow_path(self) -> None:
        # Same query, two driver paths — the materialised values must
        # agree (modulo Arrow type promotion the row path doesn't do).
        arrow_rows = self.engine.execute(
            "SELECT 1::int AS a, 'x'::text AS b",
            prefer_arrow=True,
        ).read_arrow_table().to_pylist()
        psy_rows = self.engine.execute(
            "SELECT 1::int AS a, 'x'::text AS b",
            prefer_arrow=False,
        ).read_arrow_table().to_pylist()
        self.assertEqual(arrow_rows, psy_rows)


class TestEngineRescope(PostgresTestCase):
    """``engine(schema_name=...)`` reuses the underlying connection."""

    def test_rescope_shares_connection(self) -> None:
        scoped = self.engine(schema_name=self.test_schema_name)
        self.assertIs(scoped.connection, self.engine.connection)
        self.assertEqual(scoped.schema_name, self.test_schema_name)

    def test_rescope_noop_returns_self(self) -> None:
        # No knobs means no rebind — the call returns the same engine.
        self.assertIs(self.engine(), self.engine)


class TestEngineBatch(PostgresTestCase):
    """Batched execution against the live executor."""

    def test_batch_runs_each_statement(self) -> None:
        batch = self.engine.execute_many([
            "SELECT 1 AS x",
            "SELECT 2 AS x",
            "SELECT 3 AS x",
        ])
        values = [
            r.read_arrow_table().column("x").to_pylist()[0]
            for r in batch.results.values()
        ]
        self.assertEqual(values, [1, 2, 3])
