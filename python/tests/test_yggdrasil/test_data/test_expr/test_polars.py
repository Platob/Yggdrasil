"""Tests for the polars backend.

Same shape as :mod:`test_pyarrow` — verify the round-trip through
:class:`polars.Expr` against actual frame filtering.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.tabular.execution.expr import col

pl = pytest.importorskip("polars")


def _df():
    return pl.DataFrame(
        {
            "price": [50, 200, 150, None],
            "side": ["buy", "sell", "hold", "buy"],
        },
    )


class TestPolarsFiltering:
    def test_basic_comparison(self):
        out = _df().filter((col("price") >= 100).to_polars())
        assert out["price"].to_list() == [200, 150]

    def test_and_combination(self):
        out = _df().filter(
            ((col("price") >= 100) & col("side").is_in(["buy", "sell"])).to_polars()
        )
        assert out.to_dicts() == [{"price": 200, "side": "sell"}]

    def test_is_null_filters_to_null_rows(self):
        out = _df().filter(col("price").is_null().to_polars())
        assert out["side"].to_list() == ["buy"]

    def test_between_inclusive(self):
        out = _df().filter(col("price").between(100, 200).to_polars())
        assert sorted(out["price"].to_list()) == [150, 200]

    def test_like_translates_to_regex(self):
        df = pl.DataFrame({"s": ["foobar", "baz", "FOOBAR", None]})
        out = df.filter(col("s").like("foo%").to_polars())
        assert out["s"].to_list() == ["foobar"]

        out_ci = df.filter(
            col("s").like("foo%", case_insensitive=True).to_polars()
        )
        # Case-insensitive matches both ``foobar`` and ``FOOBAR``.
        assert sorted(out_ci["s"].to_list()) == ["FOOBAR", "foobar"]
