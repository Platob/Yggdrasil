"""Tests for the Databricks whole-query pushdown adapter.

The real :class:`yggdrasil.databricks.table.table.Table` requires the
Databricks SDK. To avoid pulling that in for unit tests we forge a
duck-typed stand-in whose class name + module path match the
identity check in :func:`yggdrasil.sql.databricks_pushdown.is_databricks_table`.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# Pushdown rewrite uses the SQL planner which depends on sqlglot.
pytest.importorskip("sqlglot")

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.tabular import ArrowTabular
from yggdrasil.io.tabular.execution.sql import Engine, system_catalog
from yggdrasil.io.tabular.execution.sql.databricks_pushdown import (
    is_databricks_table,
    try_databricks_pushdown,
)


# ---------------------------------------------------------------------------
# Test doubles — installed under the Databricks Table's module path so
# the duck-type check accepts them without importing the SDK.
# ---------------------------------------------------------------------------


class _FakeStatementResult:
    """Stand-in for :class:`WarehouseStatementResult`.

    Just records the executed SQL and returns the bound table on
    every read; that's enough for the pushdown tests to assert
    "the warehouse was hit with the rewritten SQL".
    """

    def __init__(self, sql: str, table: pa.Table) -> None:
        self.executed_sql = sql
        self._table = table

    def read_arrow_table(self, options=None, **kwargs):  # noqa: ARG002
        return self._table

    def read_pylist(self, options=None, **kwargs):  # noqa: ARG002
        return self._table.to_pylist()


class _FakeSqlEngine:
    """Records executed SQL on the parent client."""

    def __init__(self, client: "_FakeClient") -> None:
        self._client = client

    def execute(self, sql: str) -> _FakeStatementResult:
        self._client.executed.append(sql)
        return _FakeStatementResult(sql, self._client.fake_result)


class _FakeClient:
    """Minimal client surface: ``.sql()`` returns a SQLEngine-like."""

    def __init__(self, fake_result: pa.Table) -> None:
        self.executed: list[str] = []
        self.fake_result = fake_result

    def sql(self, **_kwargs) -> _FakeSqlEngine:
        return _FakeSqlEngine(self)


_CACHED_FAKE_TABLE: type | None = None


def _make_fake_table_class():
    """Build a class whose module + qualname match the real one.

    :func:`is_databricks_table` accepts any class whose
    ``__module__`` is ``yggdrasil.databricks.table.table`` and whose
    ``__name__`` is ``Table`` — it never reaches into
    ``sys.modules[...]`` to compare identity. So we forge a class with
    those two attributes and don't touch the real module's ``Table``
    binding (which would shadow the SDK class for any later test on
    the same session).

    Parent modules are imported through the real Python machinery
    first so we don't accidentally shadow the on-disk
    ``yggdrasil/databricks`` package (other tests rely on
    ``yggdrasil.databricks.fs`` resolving normally). Only when a
    parent genuinely isn't importable do we install a namespace-
    package stub.
    """
    global _CACHED_FAKE_TABLE
    if _CACHED_FAKE_TABLE is not None:
        return _CACHED_FAKE_TABLE

    import importlib

    mod_path = "yggdrasil.databricks.table.table"
    parts = mod_path.split(".")
    for i in range(1, len(parts)):
        sub = ".".join(parts[: i + 1])
        if sub in sys.modules:
            continue
        try:
            importlib.import_module(sub)
            continue
        except Exception:
            pass
        # Fall back to a namespace-package stub so child modules
        # resolve. ``__path__`` makes Python treat it as a package.
        mod = types.ModuleType(sub)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[sub] = mod

    class Table(ArrowTabular):
        """Fake :class:`yggdrasil.databricks.table.table.Table` stand-in."""

        def __init__(
            self,
            data: pa.Table,
            *,
            client: _FakeClient,
            catalog_name: str,
            schema_name: str,
            table_name: str,
        ) -> None:
            super().__init__(data)
            self.client = client
            self.catalog_name = catalog_name
            self.schema_name = schema_name
            self.table_name = table_name

        def full_name(self, safe: Any = False) -> str:
            if safe:
                return f"`{self.catalog_name}`.`{self.schema_name}`.`{self.table_name}`"
            return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    Table.__module__ = mod_path
    _CACHED_FAKE_TABLE = Table
    return Table


_FakeTable = _make_fake_table_class()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trades(client: _FakeClient) -> _FakeTable:
    return _FakeTable(
        pa.table({"symbol": ["AAPL", "GOOG"], "qty": [10, 5]}),
        client=client,
        catalog_name="main",
        schema_name="warehouse",
        table_name="trades",
    )


def _users(client: _FakeClient) -> _FakeTable:
    return _FakeTable(
        pa.table({"id": [1, 2], "name": ["alice", "bob"]}),
        client=client,
        catalog_name="main",
        schema_name="warehouse",
        table_name="users",
    )


# ---------------------------------------------------------------------------
# Identity check
# ---------------------------------------------------------------------------


class TestDuckType(ArrowTestCase):
    def test_recognises_databricks_table(self) -> None:
        client = _FakeClient(pa.table({"x": [1]}))
        self.assertTrue(is_databricks_table(_trades(client)))

    def test_rejects_arrow_tabular(self) -> None:
        self.assertFalse(is_databricks_table(ArrowTabular(pa.table({"x": [1]}))))


# ---------------------------------------------------------------------------
# Whole-query rewrite
# ---------------------------------------------------------------------------


class TestPushdownRewrite(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        system_catalog.clear()
        self.client = _FakeClient(pa.table({
            "symbol": ["AAPL"], "total": [100],
        }))

    def tearDown(self) -> None:
        system_catalog.clear()
        super().tearDown()

    def test_simple_rewrite(self) -> None:
        eng = Engine(sources={"trades": _trades(self.client)})
        eng.execute("SELECT symbol, qty FROM trades WHERE qty > 5")
        self.assertEqual(len(self.client.executed), 1)
        rewritten = self.client.executed[0]
        # Fully-qualified name appears in the rewritten SQL.
        self.assertIn("main.warehouse.trades", rewritten)

    def test_join_pushed_down(self) -> None:
        eng = Engine(sources={
            "trades": _trades(self.client),
            "users": _users(self.client),
        })
        eng.execute("SELECT * FROM trades JOIN users ON trades.symbol = users.name")
        self.assertEqual(len(self.client.executed), 1)
        rewritten = self.client.executed[0]
        self.assertIn("main.warehouse.trades", rewritten)
        self.assertIn("main.warehouse.users", rewritten)

    def test_where_kwarg_merged_into_pushed_sql(self) -> None:
        eng = Engine(sources={"trades": _trades(self.client)})
        eng.execute(
            "SELECT symbol FROM trades", where="qty > 5",
        )
        self.assertEqual(len(self.client.executed), 1)
        rewritten = self.client.executed[0]
        self.assertIn("WHERE", rewritten.upper())
        # Source predicate survives.
        self.assertIn("qty", rewritten)


# ---------------------------------------------------------------------------
# Negative paths — fallback to in-process planner
# ---------------------------------------------------------------------------


class TestPushdownFallback(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        system_catalog.clear()
        self.client = _FakeClient(pa.table({"x": [1]}))

    def tearDown(self) -> None:
        system_catalog.clear()
        super().tearDown()

    def test_mixed_sources_fall_back_to_arrow(self) -> None:
        eng = Engine(sources={
            "trades": _trades(self.client),
            "local": pa.table({"symbol": ["AAPL"], "lookup": ["yes"]}),
        })
        out = eng.execute(
            "SELECT t.symbol FROM trades t JOIN local l ON t.symbol = l.symbol"
        )
        # Pushdown rejected (mixed sources) → no warehouse call.
        self.assertEqual(len(self.client.executed), 0)
        # Result still materialized correctly via the in-process planner.
        self.assertEqual(out.read_arrow_table().num_rows, 1)

    def test_pushdown_disabled_runs_locally(self) -> None:
        eng = Engine(sources={"trades": _trades(self.client)})
        out = eng.execute(
            "SELECT symbol FROM trades", pushdown=False,
        )
        self.assertEqual(len(self.client.executed), 0)
        self.assertEqual(out.read_arrow_table().num_rows, 2)

    def test_two_clients_fall_back(self) -> None:
        client_b = _FakeClient(pa.table({"x": [1]}))
        eng = Engine(sources={
            "trades": _trades(self.client),
            "trades_b": _trades(client_b),
        })
        eng.execute("SELECT * FROM trades JOIN trades_b ON trades.symbol = trades_b.symbol")
        # Two different clients — pushdown skipped.
        self.assertEqual(len(self.client.executed), 0)
        self.assertEqual(len(client_b.executed), 0)


# ---------------------------------------------------------------------------
# Direct module entry point
# ---------------------------------------------------------------------------


class TestDirectEntry(ArrowTestCase):
    def setUp(self) -> None:
        super().setUp()
        system_catalog.clear()
        self.client = _FakeClient(pa.table({"x": [1]}))

    def tearDown(self) -> None:
        system_catalog.clear()
        super().tearDown()

    def test_returns_none_on_unknown_source(self) -> None:
        from yggdrasil.io.tabular.execution.sql.dynamic_catalog import DynamicCatalog
        catalog = DynamicCatalog(parents=[])
        out = try_databricks_pushdown(
            "SELECT * FROM nope", catalog=catalog,
        )
        self.assertIsNone(out)
