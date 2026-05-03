"""Tests for :class:`ExecutionSchema` and :class:`Selector`.

Focus is on the fluent builder semantics, projection naming, and
``arrow_apply`` end-to-end behaviour against real
:class:`pyarrow.Table` inputs.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.expr import (
    Column,
    ExecutionSchema,
    Selector,
    col,
    select,
)


class TestSelector:
    def test_select_factory_builds_selector(self):
        s = select("price", alias="px")
        assert isinstance(s, Selector)
        assert isinstance(s, Column)  # Selector extends Column
        assert s.name == "price"
        assert s.alias == "px"
        assert s.projection_name == "px"

    def test_output_name_overrides_alias(self):
        s = select("price", alias="px").with_output_name("price_usd")
        assert s.projection_name == "price_usd"

    def test_projection_name_falls_back_to_name(self):
        # No alias, no output_name → bare column name.
        assert select("price").projection_name == "price"


class TestFluentBuilder:
    def test_select_appends_to_tuple(self):
        plan = ExecutionSchema().select(col("a")).select(col("b"), col("c"))
        names = plan.output_names
        assert names == ("a", "b", "c")

    def test_string_argument_is_coerced_to_column(self):
        plan = ExecutionSchema().select("a", "b")
        assert all(isinstance(s, Column) for s in plan.selects)
        assert plan.output_names == ("a", "b")

    def test_where_and_merges_repeated_calls(self):
        plan = ExecutionSchema().where(col("x") > 1).where(col("y") < 10)
        # Compiled SQL preserves the AND-merge.
        assert plan.where_predicate is not None
        assert plan.where_predicate.to_sql() == "`x` > 1 AND `y` < 10"

    def test_where_none_is_noop(self):
        plan = ExecutionSchema().where(None)
        assert plan.where_predicate is None

    def test_where_rejects_non_predicate(self):
        with pytest.raises(TypeError, match="Predicate"):
            ExecutionSchema().where(col("x") + 1)

    def test_immutability(self):
        base = ExecutionSchema().select(col("a"))
        with_b = base.select(col("b"))
        # ``base`` unchanged — every call returns a fresh schema.
        assert base.output_names == ("a",)
        assert with_b.output_names == ("a", "b")


class TestArrowApply:
    @staticmethod
    def _table():
        return pa.Table.from_pylist([
            {"price": 50, "side": "buy", "symbol": "AAPL"},
            {"price": 200, "side": "sell", "symbol": "MSFT"},
            {"price": 150, "side": "hold", "symbol": "AAPL"},
        ])

    def test_select_only_projects_named_columns(self):
        plan = ExecutionSchema().select(col("price"), col("symbol"))
        out = plan.arrow_apply(self._table())
        assert out.column_names == ["price", "symbol"]
        assert out.num_rows == 3

    def test_where_only_filters_without_projection(self):
        plan = ExecutionSchema().where(col("price") >= 100)
        out = plan.arrow_apply(self._table())
        # Empty selects → ``SELECT *`` semantics: every column kept.
        assert set(out.column_names) == {"price", "side", "symbol"}
        assert out.num_rows == 2

    def test_select_and_where_combine(self):
        plan = (
            ExecutionSchema()
            .select(col("price"), select("symbol", alias="ticker"))
            .where(col("price") >= 100)
        )
        out = plan.arrow_apply(self._table())
        assert out.column_names == ["price", "ticker"]
        assert out.to_pylist() == [
            {"price": 200, "ticker": "MSFT"},
            {"price": 150, "ticker": "AAPL"},
        ]

    def test_record_batch_input_is_lifted_to_table(self):
        rb = self._table().to_batches()[0]
        plan = ExecutionSchema().select(col("price")).where(col("price") >= 100)
        out = plan.arrow_apply(rb)
        assert isinstance(out, pa.Table)
        assert out.column_names == ["price"]

    def test_unsupported_input_type_raises(self):
        plan = ExecutionSchema().select(col("price"))
        with pytest.raises(TypeError, match="pa.Table or pa.RecordBatch"):
            plan.arrow_apply([{"price": 1}])
