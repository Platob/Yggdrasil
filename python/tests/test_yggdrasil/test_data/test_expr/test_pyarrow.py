"""Tests for the pyarrow backend.

Round-trips the AST through :class:`pyarrow.compute.Expression`
and verifies the result against actual table filtering — that's
the user-visible behaviour we care about.
"""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.expr import col


def _table():
    return pa.Table.from_pylist([
        {"price": 50, "side": "buy"},
        {"price": 200, "side": "sell"},
        {"price": 150, "side": "hold"},
        {"price": None, "side": "buy"},
    ])


class TestArrowFiltering:
    def test_basic_comparison_filters_correctly(self):
        t = _table()
        out = t.filter((col("price") >= 100).to_pyarrow())
        assert out.column("price").to_pylist() == [200, 150]

    def test_and_combination_works(self):
        t = _table()
        p = (col("price") >= 100) & col("side").is_in(["buy", "sell"])
        out = t.filter(p.to_pyarrow())
        assert out.to_pylist() == [{"price": 200, "side": "sell"}]

    def test_is_null_filters_to_null_rows_only(self):
        t = _table()
        out = t.filter(col("price").is_null().to_pyarrow())
        assert out.column("side").to_pylist() == ["buy"]

    def test_between_inclusive(self):
        t = _table()
        out = t.filter(col("price").between(100, 200).to_pyarrow())
        # NULL rows excluded by default arrow semantics.
        assert sorted(out.column("price").to_pylist()) == [150, 200]

    def test_negated_in_excludes_listed_values(self):
        t = _table()
        out = t.filter(col("side").not_in(["hold"]).to_pyarrow())
        # ``hold`` is dropped; the buy / sell rows survive, plus
        # the NULL-side row (Arrow's NOT IN keeps NULL rows by
        # default; SQL would drop them — choose the surface based
        # on the engine semantics).
        sides = sorted(out.column("side").to_pylist(), key=str)
        assert "hold" not in sides
        assert "buy" in sides and "sell" in sides


class TestNullInListExpansion:
    def test_includes_null_matches_null_rows(self):
        # ``IN (1, 2, None)`` — explicit NULL in the value list
        # should make NULL rows match. Arrow expands to
        # ``isin OR is_null``.
        t = pa.Table.from_pylist([{"x": 1}, {"x": 2}, {"x": None}, {"x": 5}])
        p = col("x").is_in([1, None])
        out = t.filter(p.to_pyarrow())
        assert sorted(
            v for v in out.column("x").to_pylist() if v is not None
        ) == [1]
        assert None in out.column("x").to_pylist()
