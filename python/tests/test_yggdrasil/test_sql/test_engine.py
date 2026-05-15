"""End-to-end tests for :class:`yggdrasil.sql.engine.Engine`."""
from __future__ import annotations

import pytest

# The SQL planner pulls in :mod:`sqlglot` for parsing — skip the whole
# module on installs without the optional ``[sql]`` extra so the
# project's base test run stays green.
pytest.importorskip("sqlglot")

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.io.tabular.execution.expr import col
from yggdrasil.io.tabular.execution.sql import (
    Aggregate,
    Engine,
    EnginePlan,
    Filter,
    Join,
    Limit,
    PlanNode,
    Project,
    Scan,
    Sort,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trades() -> pa.Table:
    return pa.table({
        "symbol": ["AAPL", "GOOG", "AAPL", "MSFT", "AAPL"],
        "qty": [10, 5, 7, 3, 2],
        "price": [150.0, 2800.0, 152.0, 300.0, 148.0],
    })


def _users() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["alice", "bob", "carol"]})


def _orders() -> pa.Table:
    return pa.table({
        "id": [10, 11, 12, 13],
        "user_id": [1, 1, 2, 4],
        "amount": [100, 200, 50, 500],
    })


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalog(ArrowTestCase):
    def test_register_and_resolve(self) -> None:
        eng = Engine()
        eng.register("trades", _trades())
        self.assertIn("trades", eng.names())
        result = eng.execute("SELECT symbol FROM trades")
        self.assertEqual(result.read_arrow_table().num_rows, 5)

    def test_register_via_kwarg(self) -> None:
        eng = Engine(sources={"t": _trades()})
        self.assertIn("t", eng.names())
        result = eng.execute("SELECT * FROM t")
        self.assertEqual(result.read_arrow_table().num_rows, 5)

    def test_unknown_source_raises(self) -> None:
        eng = Engine()
        with self.assertRaises(KeyError):
            eng.execute("SELECT * FROM nope")


# ---------------------------------------------------------------------------
# Plan inspection
# ---------------------------------------------------------------------------


class TestPlanShape(ArrowTestCase):
    def test_select_star_scan_only(self) -> None:
        eng = Engine(sources={"t": _trades()})
        ep = eng.prepare("SELECT * FROM t")
        # SELECT * collapses to a Scan with no Project on top.
        self.assertIsInstance(ep.plan, Scan)
        self.assertIsNone(ep.plan.projection)

    def test_where_lowered_into_scan(self) -> None:
        eng = Engine(sources={"t": _trades()})
        ep = eng.prepare("SELECT symbol FROM t WHERE qty > 5")
        self.assertIsInstance(ep.plan, Scan)
        self.assertIsNotNone(ep.plan.predicate)
        self.assertEqual(ep.plan.projection, ("symbol",))

    def test_group_by_builds_aggregate(self) -> None:
        eng = Engine(sources={"t": _trades()})
        ep = eng.prepare("SELECT symbol, COUNT(*) AS c FROM t GROUP BY symbol")
        # Aggregate is the top-most node when there's no ORDER BY / LIMIT
        # (the Project forwards the keyed columns).
        self.assertIsInstance(ep.plan, Project)
        self.assertIsInstance(ep.plan.children()[0], Aggregate)

    def test_order_by_limit_stack(self) -> None:
        eng = Engine(sources={"t": _trades()})
        ep = eng.prepare("SELECT symbol FROM t ORDER BY qty DESC LIMIT 2")
        # Limit → Project → Sort → Scan (Sort runs before Project so
        # ORDER BY can see physical columns the SELECT drops).
        self.assertIsInstance(ep.plan, Limit)
        project = ep.plan.children()[0]
        self.assertIsInstance(project, Project)
        sort = project.children()[0]
        self.assertIsInstance(sort, Sort)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution(ArrowTestCase):
    def test_select_star(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute("SELECT * FROM t").read_arrow_table()
        self.assertEqual(out.num_rows, 5)
        self.assertEqual(set(out.column_names), {"symbol", "qty", "price"})

    def test_where_filter(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute("SELECT symbol, qty FROM t WHERE qty > 5").read_arrow_table()
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(sorted(out.column("qty").to_pylist()), [7, 10])

    def test_group_by_sum(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol, SUM(qty) AS total FROM t GROUP BY symbol "
            "ORDER BY total DESC"
        ).read_arrow_table()
        rows = out.to_pylist()
        self.assertEqual(rows[0], {"symbol": "AAPL", "total": 19})

    def test_group_by_count_star(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol, COUNT(*) AS n FROM t GROUP BY symbol ORDER BY symbol"
        ).read_arrow_table()
        self.assertEqual(out.column("n").to_pylist(), [3, 1, 1])

    def test_global_aggregate(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute("SELECT COUNT(*) AS c, SUM(qty) AS s FROM t").read_arrow_table()
        self.assertEqual(out.to_pylist(), [{"c": 5, "s": 27}])

    def test_order_by_limit(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol, qty FROM t ORDER BY qty DESC LIMIT 2"
        ).read_arrow_table()
        self.assertEqual(out.column("qty").to_pylist(), [10, 7])

    def test_limit_offset(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol FROM t ORDER BY qty ASC LIMIT 2 OFFSET 1"
        ).read_arrow_table()
        # qty ASC: 2,3,5,7,10 → symbols MSFT, GOOG, AAPL, AAPL — offset 1
        # gives the 2nd and 3rd.
        self.assertEqual(out.num_rows, 2)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


class TestJoins(ArrowTestCase):
    def test_inner_join(self) -> None:
        eng = Engine(sources={"u": _users(), "o": _orders()})
        out = eng.execute(
            "SELECT u.name, o.amount FROM u JOIN o ON u.id = o.user_id "
            "ORDER BY u.name, o.amount"
        ).read_arrow_table()
        self.assertEqual(
            out.to_pylist(),
            [
                {"name": "alice", "amount": 100},
                {"name": "alice", "amount": 200},
                {"name": "bob", "amount": 50},
            ],
        )

    def test_left_join_keeps_unmatched(self) -> None:
        eng = Engine(sources={"u": _users(), "o": _orders()})
        out = eng.execute(
            "SELECT u.name, o.amount FROM u LEFT JOIN o ON u.id = o.user_id "
            "ORDER BY u.name, o.amount"
        ).read_arrow_table()
        rows = out.to_pylist()
        # carol (id=3) has no matching order — appears with NULL amount.
        unmatched = [r for r in rows if r["amount"] is None]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["name"], "carol")


# ---------------------------------------------------------------------------
# Predicate composition (where=)
# ---------------------------------------------------------------------------


class TestPredicateCompose(ArrowTestCase):
    def test_where_kwarg_intersects_with_sql(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol, qty FROM t",
            where=col("qty") > 5,
        ).read_arrow_table()
        self.assertEqual(sorted(out.column("qty").to_pylist()), [7, 10])

    def test_where_string_lifts_to_predicate(self) -> None:
        eng = Engine(sources={"t": _trades()})
        out = eng.execute(
            "SELECT symbol FROM t WHERE qty > 0",
            where="symbol = 'AAPL'",
        ).read_arrow_table()
        self.assertEqual(set(out.column("symbol").to_pylist()), {"AAPL"})


# ---------------------------------------------------------------------------
# Output engines (CastOptions integration)
# ---------------------------------------------------------------------------


class TestOutputEngines(ArrowTestCase):
    def test_to_arrow_table(self) -> None:
        eng = Engine(sources={"t": _trades()})
        result = eng.execute("SELECT symbol, qty FROM t WHERE qty > 5")
        out = result.read_arrow_table()
        self.assertIsInstance(out, pa.Table)

    def test_to_polars_frame(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")
        eng = Engine(sources={"t": _trades()})
        result = eng.execute("SELECT symbol, qty FROM t WHERE qty > 5")
        df = result.read_polars_frame()
        self.assertEqual(df.height, 2)

    def test_read_pylist(self) -> None:
        eng = Engine(sources={"t": _trades()})
        result = eng.execute("SELECT symbol FROM t LIMIT 2")
        self.assertEqual(len(result.read_pylist()), 2)


# ---------------------------------------------------------------------------
# Tabular sources — Delta as a SQL source
# ---------------------------------------------------------------------------


class TestTabularSources(ArrowTestCase):
    def test_arrow_tabular_source(self) -> None:
        from yggdrasil.io.tabular import ArrowTabular

        eng = Engine(sources={"t": ArrowTabular(_trades())})
        out = eng.execute("SELECT symbol FROM t WHERE qty > 5").read_arrow_table()
        self.assertEqual(out.num_rows, 2)


# ---------------------------------------------------------------------------
# Programmatic plan
# ---------------------------------------------------------------------------


class TestProgrammaticPlan(ArrowTestCase):
    def test_run_plan_directly(self) -> None:
        eng = Engine(sources={"t": _trades()})
        scan = Scan(name="t", projection=("symbol", "qty"))
        result = eng.run_plan(scan)
        self.assertEqual(result.read_arrow_table().num_rows, 5)

    def test_engine_plan_round_trip(self) -> None:
        eng = Engine(sources={"t": _trades()})
        ep = eng.prepare("SELECT symbol FROM t WHERE qty > 5")
        self.assertIsInstance(ep, EnginePlan)
        out = eng.execute(ep).read_arrow_table()
        self.assertEqual(out.num_rows, 2)
