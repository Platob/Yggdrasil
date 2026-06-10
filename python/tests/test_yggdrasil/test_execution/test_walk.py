"""Tree-walk visitor tests — ``walk`` coverage and ``free_columns``."""

from __future__ import annotations

from yggdrasil.execution.expr import col, lit
from yggdrasil.execution.expr.nodes import (
    Alias,
    CaseWhen,
    Column,
    FunctionCall,
    Lambda,
    Literal,
    SortOrder,
    Subscript,
    WindowFunction,
    WindowSpec,
)
from yggdrasil.execution.expr.walk import free_columns, walk


def test_walk_yields_every_node_preorder():
    pred = (col("a") > 1) & col("b").is_in(["x"])
    nodes = list(walk(pred))
    assert nodes[0] is pred
    names = [n.name for n in nodes if isinstance(n, Column)]
    assert names == ["a", "b"]
    # Logical + Comparison + Column(a) + Literal(1) + InList + Column(b)
    assert len(nodes) == 6


def test_walk_covers_between_and_not():
    pred = ~col("x").between(col("lo"), col("hi"))
    assert free_columns(pred) == ("x", "lo", "hi")


def test_walk_covers_function_alias_sort_window():
    fn = FunctionCall("SUM", (col("amount"),))
    win = WindowFunction(
        fn,
        WindowSpec(
            partition_by=(col("region"),),
            order_by=(SortOrder(col("ts")),),
        ),
    )
    aliased = Alias(win, "total")
    assert free_columns(aliased) == ("amount", "region", "ts")


def test_walk_covers_case_when_and_subscript():
    case = CaseWhen(
        branches=((col("a") == 1, lit("one")),),
        else_expr=Subscript(col("tags"), lit(0)),
        operand=col("kind"),
    )
    assert free_columns(case) == ("kind", "a", "tags")


def test_free_columns_dedupes_in_first_seen_order():
    pred = (col("b") > 1) & (col("a") > 2) & (col("b") < 9)
    assert free_columns(pred) == ("b", "a")


def test_free_columns_excludes_lambda_params():
    # FILTER(items, x -> x > threshold): ``x`` is bound by the lambda,
    # ``items`` / ``threshold`` are free.
    body = col("x") > col("threshold")
    call = FunctionCall("FILTER", (col("items"), Lambda(("x",), body)))
    assert free_columns(call) == ("items", "threshold")


def test_walk_literal_is_terminal():
    only = list(walk(Literal(5)))
    assert len(only) == 1
