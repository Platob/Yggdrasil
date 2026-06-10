"""Python backend tests — ``to_python`` compiles to row callables with
SQL three-valued logic (missing / NULL → None, not False)."""

from __future__ import annotations

import pytest

from yggdrasil.execution.expr import col


def test_comparisons_evaluate():
    fn = (col("price") >= 100).to_python()
    assert fn({"price": 150}) is True
    assert fn({"price": 50}) is False


def test_missing_column_is_null_by_default():
    fn = (col("price") >= 100).to_python()
    assert fn({}) is None


def test_strict_mode_raises_on_missing_column():
    fn = (col("price") >= 100).to_python(strict=True)
    with pytest.raises(KeyError):
        fn({})


def test_logical_three_valued_semantics():
    pred = (col("a") > 1) & (col("b") > 1)
    fn = pred.to_python()
    assert fn({"a": 2, "b": 2}) is True
    assert fn({"a": 0, "b": 2}) is False
    # NULL AND TRUE → NULL; NULL AND FALSE → FALSE (Kleene logic).
    assert fn({"a": None, "b": 2}) is None
    assert fn({"a": None, "b": 0}) is False

    any_fn = ((col("a") > 1) | (col("b") > 1)).to_python()
    assert any_fn({"a": 0, "b": 2}) is True
    # NULL OR TRUE → TRUE; NULL OR FALSE → NULL.
    assert any_fn({"a": None, "b": 2}) is True
    assert any_fn({"a": None, "b": 0}) is None


def test_not_inverts_and_propagates_null():
    fn = (~(col("a") > 1)).to_python()
    assert fn({"a": 0}) is True
    assert fn({"a": 2}) is False
    assert fn({"a": None}) is None


def test_between_is_inclusive():
    fn = col("x").between(1, 5).to_python()
    assert fn({"x": 1}) is True
    assert fn({"x": 5}) is True
    assert fn({"x": 6}) is False
    assert col("x").not_between(1, 5).to_python()({"x": 6}) is True


def test_in_list_membership_and_null_flag():
    fn = col("side").is_in(["buy", "sell"]).to_python()
    assert fn({"side": "buy"}) is True
    assert fn({"side": "hold"}) is False

    with_null = col("side").is_in(["buy", None]).to_python()
    assert with_null({"side": None}) is True


def test_is_null_checks():
    assert col("x").is_null().to_python()({"x": None}) is True
    assert col("x").is_null().to_python()({"x": 1}) is False
    assert col("x").is_not_null().to_python()({"x": 1}) is True


def test_like_wildcards_and_case_flag():
    fn = col("name").like("a%_c").to_python()
    assert fn({"name": "abXc"}) is True
    assert fn({"name": "ab"}) is False

    ci = col("name").like("ABC%", case_insensitive=True).to_python()
    assert ci({"name": "abcdef"}) is True

    neg = col("name").not_like("a%").to_python()
    assert neg({"name": "zzz"}) is True


def test_arithmetic_chains():
    fn = ((col("a") + 1) * 2 == 8).to_python()
    assert fn({"a": 3}) is True
    assert fn({"a": 2}) is False


def test_eq_null_literal_is_unknown():
    # SQL 3VL: ``x = NULL`` is UNKNOWN whatever the row holds.
    fn = (col("x") == None).to_python()  # noqa: E711
    assert fn({"x": 1}) is None
    assert fn({"x": None}) is None


def test_filter_pylist_uses_predicate():
    rows = [{"k": 1}, {"k": 2}, {"k": 3}]
    assert col("k").is_in([1, 3]).filter_pylist(rows) == [{"k": 1}, {"k": 3}]
    assert (col("k") > 1).filter_pylist(rows) == [{"k": 2}, {"k": 3}]


def test_filter_iterable_with_key_projection():
    items = [("a", {"k": 1}), ("b", {"k": 2})]
    kept = list((col("k") == 2).filter_iterable(items, key=lambda t: t[1]))
    assert kept == [("b", {"k": 2})]
