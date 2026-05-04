"""End-to-end tests for :mod:`yggdrasil.sql`.

Coverage rules of the road:

- :class:`SqlPolarsTestCase` for anything beyond a bare
  ``SELECT cols FROM src [WHERE] [LIMIT]`` — joins, aggregations,
  ORDER BY, computed projections.
- :class:`SqlArrowTestCase` for the polars-free fallback path so
  base-install behavior is exercised independently.
- :class:`SqlTestCase` for parser / utility / context tests that
  don't actually execute SQL.
"""

from __future__ import annotations

from yggdrasil.sql.tests import (
    SqlArrowTestCase,
    SqlPolarsTestCase,
    SqlTestCase,
)


# ---------------------------------------------------------------------------
# Catalog / coercion
# ---------------------------------------------------------------------------


class TestCatalog(SqlTestCase):
    def test_register_arrow_table(self) -> None:
        table = self.table({"a": [1, 2, 3]})
        self.register("nums", table)
        self.assertIn("nums", self.ctx)
        io = self.ctx["nums"]
        self.assertEqual(io.read_arrow_table().num_rows, 3)

    def test_register_list_of_dicts(self) -> None:
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        self.register("rows", rows)
        out = self.ctx["rows"].read_arrow_table()
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(out.column_names, ["x", "y"])

    def test_unknown_source_raises_with_suggestion(self) -> None:
        self.register("trades", self.table({"x": [1]}))
        with self.assertRaises(KeyError) as cx:
            _ = self.ctx["trade"]
        msg = str(cx.exception)
        self.assertIn("trade", msg)
        self.assertIn("trades", msg)


# ---------------------------------------------------------------------------
# Parser & utilities
# ---------------------------------------------------------------------------


class TestParser(SqlTestCase):
    def test_extract_sources(self) -> None:
        from yggdrasil.sql import extract_sources, parse

        root = parse("SELECT * FROM trades JOIN users ON trades.uid = users.id")
        names = extract_sources(root)
        self.assertIn("trades", names)
        self.assertIn("users", names)

    def test_parse_predicate_to_python(self) -> None:
        from yggdrasil.sql import parse_predicate

        pred = parse_predicate("price >= 100 AND side = 'buy'")
        fn = pred.to_python()
        self.assertTrue(fn({"price": 150, "side": "buy"}))
        self.assertFalse(fn({"price": 50, "side": "buy"}))
        self.assertFalse(fn({"price": 150, "side": "sell"}))

    def test_empty_query_raises(self) -> None:
        from yggdrasil.sql import SqlParseError, parse

        with self.assertRaises(SqlParseError):
            parse("")

    def test_quote_ident_databricks_default(self) -> None:
        from yggdrasil.sql import quote_ident

        self.assertEqual(quote_ident("foo"), "`foo`")
        self.assertEqual(quote_ident("a`b"), "`a``b`")

    def test_quote_ident_postgres(self) -> None:
        from yggdrasil.sql import quote_ident

        self.assertEqual(quote_ident("foo", dialect="postgres"), '"foo"')

    def test_split_qualified_ident(self) -> None:
        from yggdrasil.sql import split_qualified_ident

        self.assertEqual(
            split_qualified_ident("catalog.schema.tbl"),
            ["catalog", "schema", "tbl"],
        )
        self.assertEqual(
            split_qualified_ident('"My DB"."public"."t"'),
            ["My DB", "public", "t"],
        )
        self.assertEqual(
            split_qualified_ident("`my db`.`public`.`t`"),
            ["my db", "public", "t"],
        )


# ---------------------------------------------------------------------------
# Arrow-only fallback path — uses ArrowSqlExecutor explicitly
# ---------------------------------------------------------------------------


class TestArrowFallback(SqlArrowTestCase):
    def test_select_star(self) -> None:
        self.register("t", self.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
        result = self.sql("SELECT * FROM t")
        out = result.read_arrow_table()
        self.assertEqual(out.num_rows, 3)
        self.assertEqual(out.column_names, ["a", "b"])

    def test_select_with_where(self) -> None:
        self.register("t", self.table({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]}))
        result = self.sql("SELECT a, b FROM t WHERE a > 2")
        out = result.read_arrow_table()
        self.assertEqual(out.num_rows, 2)
        self.assertEqual(out["a"].to_pylist(), [3, 4])

    def test_select_with_limit(self) -> None:
        self.register("t", self.table({"a": list(range(10))}))
        result = self.sql("SELECT * FROM t LIMIT 3")
        self.assertEqual(result.read_arrow_table().num_rows, 3)

    def test_select_with_alias(self) -> None:
        self.register("t", self.table({"a": [1, 2]}))
        out = self.sql("SELECT a AS renamed FROM t").read_arrow_table()
        self.assertEqual(out.column_names, ["renamed"])

    def test_join_raises_install_polars_hint(self) -> None:
        self.register("t", self.table({"a": [1]}))
        self.register("u", self.table({"a": [1]}))
        with self.assertRaises(NotImplementedError) as cx:
            self.sql("SELECT * FROM t JOIN u ON t.a = u.a")
        self.assertIn("polars", str(cx.exception).lower())

    def test_where_with_python_predicate_composes(self) -> None:
        from yggdrasil.data.expr import col

        self.register("t", self.table({"a": [1, 2, 3, 4]}))
        result = self.sql(
            "SELECT * FROM t WHERE a > 1",
            where=col("a") < 4,
        )
        out = result.read_arrow_table()
        self.assertEqual(out["a"].to_pylist(), [2, 3])


# ---------------------------------------------------------------------------
# Polars-backed path — full SQL surface
# ---------------------------------------------------------------------------


class TestPolarsBackend(SqlPolarsTestCase):
    def test_aggregation(self) -> None:
        self.register("trades", self.table({
            "symbol": ["AAPL", "GOOG", "AAPL", "MSFT"],
            "qty": [10, 5, 7, 3],
        }))
        result = self.sql(
            "SELECT symbol, SUM(qty) AS total FROM trades "
            "GROUP BY symbol ORDER BY total DESC"
        )
        out = result.read_polars_frame()
        self.assertEqual(out["symbol"].to_list(), ["AAPL", "GOOG", "MSFT"])
        self.assertEqual(out["total"].to_list(), [17, 5, 3])

    def test_join(self) -> None:
        self.register("orders", self.table({
            "id": [1, 2, 3],
            "user_id": [10, 10, 11],
            "amount": [100, 200, 50],
        }))
        self.register("users", self.table({
            "id": [10, 11],
            "name": ["alice", "bob"],
        }))
        result = self.sql(
            "SELECT u.name, SUM(o.amount) AS spent "
            "FROM orders o JOIN users u ON o.user_id = u.id "
            "GROUP BY u.name ORDER BY spent DESC"
        )
        out = result.read_polars_frame()
        self.assertEqual(out["name"].to_list(), ["alice", "bob"])
        self.assertEqual(out["spent"].to_list(), [300, 50])

    def test_persist_memory_caches_reads(self) -> None:
        self.register("t", self.table({"a": [1, 2, 3]}))
        result = self.sql("SELECT * FROM t")
        # Hits cache on second read — no re-execution.
        first = result.read_arrow_table()
        second = result.read_arrow_table()
        self.assertEqual(first.num_rows, second.num_rows)
        self.assertTrue(result.cached)

    def test_persist_path_writes_parquet(self) -> None:
        target = self.tmp_path / "out.parquet"
        self.register("t", self.table({"a": [1, 2, 3]}))
        result = self.sql(
            "SELECT * FROM t WHERE a > 1",
            persist="path",
            path=str(target),
        )
        self.assertTrue(target.exists())
        # Round-trip through the persisted holder.
        out = result.read_arrow_table()
        self.assertEqual(out["a"].to_pylist(), [2, 3])

    def test_select_kwarg_renames_after_sql(self) -> None:
        from yggdrasil.data.expr import select

        self.register("t", self.table({"a": [1, 2], "b": [3, 4]}))
        result = self.sql(
            "SELECT a, b FROM t",
            select=[select("a", output_name="aa"), "b"],
        )
        out = result.read_arrow_table()
        self.assertEqual(out.column_names, ["aa", "b"])

    def test_sources_kwarg_overrides_context(self) -> None:
        # Context-bound source is shadowed by per-call override.
        self.register("t", self.table({"a": [1]}))
        local = self.table({"a": [99]})
        result = self.sql("SELECT * FROM t", sources={"t": local})
        self.assertEqual(result.read_arrow_table()["a"].to_pylist(), [99])

    def test_string_where_is_parsed_to_predicate(self) -> None:
        self.register("t", self.table({"a": [1, 2, 3]}))
        result = self.sql("SELECT * FROM t", where="a > 1")
        self.assertEqual(
            result.read_arrow_table()["a"].to_pylist(), [2, 3],
        )


# ---------------------------------------------------------------------------
# StatementResult lifecycle
# ---------------------------------------------------------------------------


class TestStatementResult(SqlPolarsTestCase):
    def test_result_is_tabular_io(self) -> None:
        from yggdrasil.io.buffer.base import TabularIO

        self.register("t", self.table({"a": [1, 2]}))
        result = self.sql("SELECT * FROM t")
        self.assertIsInstance(result, TabularIO)

    def test_collect_schema(self) -> None:
        self.register("t", self.table({"a": [1, 2], "b": ["x", "y"]}))
        schema = self.sql("SELECT * FROM t").collect_schema()
        names = [f.name for f in schema.fields]
        self.assertEqual(names, ["a", "b"])

    def test_to_records(self) -> None:
        self.register("t", self.table({"a": [1, 2], "b": ["x", "y"]}))
        result = self.sql("SELECT * FROM t")
        records = list(result.to_records())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["a"], 1)

    def test_unpersist_then_read_works(self) -> None:
        self.register("t", self.table({"a": [1, 2]}))
        result = self.sql("SELECT * FROM t")
        result.read_arrow_table()
        self.assertTrue(result.cached)
        result.unpersist()
        # After unpersist we re-execute on next read; the result is
        # still in started state so .start() is idempotent.
        result.start(reset=True)
        self.assertEqual(result.read_arrow_table().num_rows, 2)
