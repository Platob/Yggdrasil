"""Partition-pruning extractor tests — ``extract_partition_filters``.

The extractor is allowed to over-approximate (a surviving file may
still hold zero matching rows) but must never under-approximate: no
row the predicate accepts may fall outside the returned sets.
"""

from __future__ import annotations

from yggdrasil.execution.expr import col
from yggdrasil.execution.expr.nodes import InList, IsNull, Logical
from yggdrasil.execution.expr.operators import LogicalOp
from yggdrasil.execution.expr.partition import extract_partition_filters


def test_eq_constrains_to_single_value():
    out = extract_partition_filters(col("region") == "eu", ["region"])
    assert out == {"region": frozenset({"eu"})}


def test_eq_literal_on_left():
    from yggdrasil.execution.expr.nodes import Comparison, Literal
    from yggdrasil.execution.expr.operators import CompareOp

    flipped = Comparison(Literal("eu"), CompareOp.EQ, col("region"))
    assert extract_partition_filters(flipped, ["region"]) == {
        "region": frozenset({"eu"}),
    }


def test_is_in_constrains_to_value_set():
    out = extract_partition_filters(col("region").is_in(["eu", "us"]), ["region"])
    assert out == {"region": frozenset({"eu", "us"})}


def test_is_in_with_null_adds_none():
    out = extract_partition_filters(col("region").is_in(["eu", None]), ["region"])
    assert out == {"region": frozenset({"eu", None})}


def test_is_null_constrains_to_none():
    out = extract_partition_filters(col("region").is_null(), ["region"])
    assert out == {"region": frozenset({None})}


def test_unlisted_column_is_ignored():
    assert extract_partition_filters(col("other") == 1, ["region"]) == {}


def test_empty_column_list_returns_empty():
    assert extract_partition_filters(col("region") == "eu", []) == {}


def test_and_intersects_per_column():
    pred = col("region").is_in(["eu", "us"]) & (col("region") == "eu")
    assert extract_partition_filters(pred, ["region"]) == {
        "region": frozenset({"eu"}),
    }


def test_and_unions_keys_across_columns():
    pred = (col("region") == "eu") & (col("year") == 2026)
    assert extract_partition_filters(pred, ["region", "year"]) == {
        "region": frozenset({"eu"}),
        "year": frozenset({2026}),
    }


def test_and_can_prove_unsatisfiable():
    pred = (col("region") == "eu") & (col("region") == "us")
    assert extract_partition_filters(pred, ["region"]) == {
        "region": frozenset(),
    }


def test_or_unions_when_every_branch_constrains():
    # ``a == 1 OR a == 2`` collapses to InList at construction, so
    # build the Logical explicitly to exercise the OR path with
    # mixed shapes the collapse refuses (different columns per branch).
    pred = Logical(
        LogicalOp.OR,
        (
            (col("region") == "eu") & (col("year") == 2025),
            (col("region") == "us") & (col("year") == 2026),
        ),
    )
    assert extract_partition_filters(pred, ["region", "year"]) == {
        "region": frozenset({"eu", "us"}),
        "year": frozenset({2025, 2026}),
    }


def test_or_drops_column_with_unconstrained_branch():
    # The second branch doesn't constrain ``region`` — the OR could
    # accept any region via that branch, so no pruning is safe.
    pred = Logical(
        LogicalOp.OR,
        ((col("region") == "eu"), (col("year") == 2026)),
    )
    assert extract_partition_filters(pred, ["region", "year"]) == {}


def test_ranges_not_and_like_yield_no_constraint():
    assert extract_partition_filters(col("x") > 1, ["x"]) == {}
    assert extract_partition_filters(col("x").between(1, 5), ["x"]) == {}
    assert extract_partition_filters(~(col("x") == 1), ["x"]) == {}
    assert extract_partition_filters(col("x") != 1, ["x"]) == {}
    assert extract_partition_filters(col("x").like("a%"), ["x"]) == {}


def test_negated_shapes_yield_no_constraint():
    assert extract_partition_filters(
        InList(col("x"), (1, 2), negated=True), ["x"],
    ) == {}
    assert extract_partition_filters(
        IsNull(col("x"), negated=True), ["x"],
    ) == {}


def test_eq_null_yields_no_constraint():
    # ``x == NULL`` is UNKNOWN for every row — a {None} set would lie.
    assert extract_partition_filters(col("x") == None, ["x"]) == {}  # noqa: E711
