"""Tests for the polars backend.

Same shape as :mod:`test_pyarrow` — verify the round-trip through
:class:`polars.Expr` against actual frame filtering.
"""

from __future__ import annotations

import pytest

from yggdrasil.execution.expr import col

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


class TestPolarsTimezoneLiteral:
    """``filter_polars_frame`` against tz-aware columns.

    polars treats ``"UTC"`` and ``"Etc/UTC"`` as distinct zones for
    comparison and raises ``SchemaError`` when a literal's zone string
    differs from the column's. yggdrasil's Timezone canonicalises every
    UTC spelling to ``"Etc/UTC"``, so the emitted literal used to mismatch
    a column polars spells ``"UTC"``; the emitter now re-stamps the
    literal's tz from the (pushdown-aligned) value's own zone key.
    """

    def _utc_frame(self, tz: str):
        import datetime as dt
        import pyarrow as pa
        tbl = pa.table({
            "ts": pa.array(
                [dt.datetime(2026, 1, 1, h, tzinfo=dt.timezone.utc) for h in range(24)],
                type=pa.timestamp("us", tz=tz),
            ),
        })
        return pl.from_arrow(tbl)

    @pytest.mark.parametrize("col_tz", ["UTC", "Etc/UTC"])
    def test_tz_aware_literal_matches_column_spelling(self, col_tz):
        import datetime as dt
        import zoneinfo
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.types.primitive.temporal import TimestampType

        frame = self._utc_frame(col_tz)
        # 12:00 Europe/Paris == 11:00 UTC — the bare-tz pushdown converts the
        # literal to the column's zone; the filter must not raise on the
        # UTC-vs-Etc/UTC string mismatch.
        paris_field = Field(name="ts", dtype=TimestampType(tz="Europe/Paris"))
        paris = dt.datetime(2026, 1, 1, 12, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))
        out = (col("ts", field=paris_field) == paris).filter_polars_frame(frame)
        assert out.height == 1
        assert out["ts"].to_list()[0].hour == 11

    def test_non_utc_zone_unaffected(self):
        import datetime as dt
        import zoneinfo
        import pyarrow as pa
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.types.primitive.temporal import TimestampType

        paris = dt.datetime(2026, 1, 1, 12, tzinfo=zoneinfo.ZoneInfo("Europe/Paris"))
        frame = pl.from_arrow(
            pa.table({"ts": pa.array([paris], type=pa.timestamp("us", tz="Europe/Paris"))})
        )
        paris_field = Field(name="ts", dtype=TimestampType(tz="Europe/Paris"))
        out = (col("ts", field=paris_field) == paris).filter_polars_frame(frame)
        assert out.height == 1
