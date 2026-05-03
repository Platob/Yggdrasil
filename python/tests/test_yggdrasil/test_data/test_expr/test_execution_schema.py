"""Tests for :class:`ExecutionSchema` and :class:`Selector`.

Covers fluent ``with_*`` builders, projection naming,
``arrow_apply`` end-to-end on :class:`pyarrow.Table` inputs,
and ``arrow_batch_apply`` for the streaming
:class:`pyarrow.RecordBatch` path. The bound-source mode (no-arg
``arrow_apply`` reading from ``ExecutionSchema.source``) is
exercised via an in-memory :class:`MemoryArrowIO`.
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


def _table():
    return pa.Table.from_pylist([
        {"price": 50, "side": "buy", "symbol": "AAPL"},
        {"price": 200, "side": "sell", "symbol": "MSFT"},
        {"price": 150, "side": "hold", "symbol": "AAPL"},
    ])


def _bound_source(table=None):
    """Build an in-memory TabularIO bound to ``table`` (or _table())."""
    from yggdrasil.io.buffer.memory import MemoryArrowIO

    table = _table() if table is None else table
    holder = MemoryArrowIO(schema=table.schema)
    holder.write_arrow_table(table)
    return holder


class TestSelector:
    def test_select_factory_builds_selector(self):
        s = select("price", alias="px")
        assert isinstance(s, Selector)
        assert isinstance(s, Column)
        assert s.name == "price"
        assert s.alias == "px"
        assert s.projection_name == "px"

    def test_output_name_overrides_alias(self):
        s = select("price", alias="px").with_output_name("price_usd")
        assert s.projection_name == "price_usd"

    def test_projection_name_falls_back_to_name(self):
        assert select("price").projection_name == "price"


class TestConstructor:
    def test_default_construction(self):
        plan = ExecutionSchema()
        assert plan.alias == ""
        assert plan.source is None
        assert plan.select == ()
        assert plan.where is None
        # Empty plan is falsy — handy for "did the caller
        # configure anything?" checks.
        assert not plan

    def test_keyword_construction(self):
        plan = ExecutionSchema(
            alias="t",
            source=_bound_source(),
            select=[col("price"), select("symbol", alias="ticker")],
            where=col("price") >= 100,
        )
        assert plan.alias == "t"
        assert plan.source is not None
        # Iterables get coerced to a tuple of Expression nodes.
        assert isinstance(plan.select, tuple)
        assert plan.select[0].name == "price"
        assert isinstance(plan.select[1], Selector)
        assert plan.where is not None
        assert bool(plan)

    def test_string_select_arg_is_coerced_to_column(self):
        plan = ExecutionSchema(select=["a", "b"])
        assert all(isinstance(s, Column) for s in plan.select)
        assert plan.output_names == ("a", "b")

    def test_non_predicate_where_rejected_at_construction(self):
        with pytest.raises(TypeError, match="Predicate"):
            ExecutionSchema(where=col("x") + 1)


class TestFluentBuilders:
    def test_with_select_appends_to_tuple(self):
        plan = (
            ExecutionSchema()
            .with_select(col("a"))
            .with_select(col("b"), col("c"))
        )
        assert plan.output_names == ("a", "b", "c")

    def test_with_where_and_merges_repeated_calls(self):
        plan = (
            ExecutionSchema()
            .with_where(col("x") > 1)
            .with_where(col("y") < 10)
        )
        assert plan.where is not None
        assert plan.where.to_sql() == "`x` > 1 AND `y` < 10"

    def test_with_where_none_is_noop(self):
        plan = ExecutionSchema().with_where(None)
        assert plan.where is None

    def test_with_alias_and_with_source_rebind_cleanly(self):
        src = _bound_source()
        plan = ExecutionSchema().with_alias("t").with_source(src)
        assert plan.alias == "t"
        assert plan.source is src

    def test_immutability(self):
        base = ExecutionSchema().with_select(col("a"))
        with_b = base.with_select(col("b"))
        # ``base`` unchanged — every call returns a fresh schema.
        assert base.output_names == ("a",)
        assert with_b.output_names == ("a", "b")


class TestArrowApply:
    def test_select_only_projects_named_columns(self):
        plan = ExecutionSchema().with_select(col("price"), col("symbol"))
        out = plan.arrow_apply(_table())
        assert out.column_names == ["price", "symbol"]
        assert out.num_rows == 3

    def test_where_only_filters_without_projection(self):
        plan = ExecutionSchema().with_where(col("price") >= 100)
        out = plan.arrow_apply(_table())
        # Empty select → ``SELECT *``: every column kept.
        assert set(out.column_names) == {"price", "side", "symbol"}
        assert out.num_rows == 2

    def test_select_and_where_combine(self):
        plan = (
            ExecutionSchema()
            .with_select(col("price"), select("symbol", alias="ticker"))
            .with_where(col("price") >= 100)
        )
        out = plan.arrow_apply(_table())
        assert out.column_names == ["price", "ticker"]
        assert out.to_pylist() == [
            {"price": 200, "ticker": "MSFT"},
            {"price": 150, "ticker": "AAPL"},
        ]

    def test_record_batch_input_is_lifted_to_table(self):
        rb = _table().to_batches()[0]
        plan = (
            ExecutionSchema()
            .with_select(col("price"))
            .with_where(col("price") >= 100)
        )
        out = plan.arrow_apply(rb)
        assert isinstance(out, pa.Table)
        assert out.column_names == ["price"]

    def test_unsupported_input_type_raises(self):
        plan = ExecutionSchema().with_select(col("price"))
        with pytest.raises(TypeError, match="pa.Table or pa.RecordBatch"):
            plan.arrow_apply([{"price": 1}])

    def test_bound_source_drives_no_arg_arrow_apply(self):
        # When the plan carries a source, arrow_apply() reads from
        # it directly — no caller-supplied input needed.
        plan = (
            ExecutionSchema(source=_bound_source())
            .with_select(col("price"))
            .with_where(col("price") >= 100)
        )
        out = plan.arrow_apply()
        assert out.column_names == ["price"]
        assert sorted(out.column("price").to_pylist()) == [150, 200]

    def test_no_source_and_no_arg_raises(self):
        with pytest.raises(TypeError, match="needs a source argument"):
            ExecutionSchema().with_select(col("price")).arrow_apply()


class TestArrowBatchApply:
    def test_filters_and_projects_a_record_batch(self):
        rb = _table().to_batches()[0]
        plan = (
            ExecutionSchema()
            .with_select(col("price"), select("symbol", alias="ticker"))
            .with_where(col("price") >= 100)
        )
        out = plan.arrow_batch_apply(rb)
        assert isinstance(out, pa.RecordBatch)
        assert out.schema.names == ["price", "ticker"]
        assert out.to_pylist() == [
            {"price": 200, "ticker": "MSFT"},
            {"price": 150, "ticker": "AAPL"},
        ]

    def test_empty_filter_result_returns_zero_row_batch(self):
        rb = _table().to_batches()[0]
        plan = (
            ExecutionSchema()
            .with_select(col("price"))
            .with_where(col("price") > 9999)
        )
        out = plan.arrow_batch_apply(rb)
        # The contract: still a RecordBatch matching the
        # projected schema, just zero rows — downstream
        # iterators don't have to special-case empty results.
        assert isinstance(out, pa.RecordBatch)
        assert out.schema.names == ["price"]
        assert out.num_rows == 0

    def test_bound_source_drives_no_arg_batch_apply(self):
        # No batch arg + bound source → consume the first batch
        # from the source.
        plan = (
            ExecutionSchema(source=_bound_source())
            .with_select(col("price"))
            .with_where(col("price") >= 100)
        )
        out = plan.arrow_batch_apply()
        assert isinstance(out, pa.RecordBatch)
        assert out.schema.names == ["price"]

    def test_unsupported_batch_type_raises(self):
        plan = ExecutionSchema().with_select(col("price"))
        with pytest.raises(TypeError, match="pa.RecordBatch"):
            plan.arrow_batch_apply(_table())  # Table, not RecordBatch

    def test_select_star_when_select_is_empty(self):
        # Empty select with a real input → all source columns
        # survive (SQL ``SELECT *`` semantics).
        rb = _table().to_batches()[0]
        plan = ExecutionSchema().with_where(col("price") >= 100)
        out = plan.arrow_batch_apply(rb)
        assert set(out.schema.names) == {"price", "side", "symbol"}
        assert out.num_rows == 2


class TestSelectStarFromBoundSource:
    def test_output_names_lift_from_bound_source(self):
        # Empty select + bound source → output_names follows
        # the source schema.
        plan = ExecutionSchema(source=_bound_source())
        # MemoryArrowIO's collect_schema gives back a yggdrasil
        # Schema whose field names match the underlying pa.Table.
        names = plan.output_names
        assert set(names) == {"price", "side", "symbol"}
