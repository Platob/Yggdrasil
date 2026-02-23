"""Unit tests for polars_cast module.

Each test class covers one public function. Tests are self-contained —
no external fixtures, no file I/O, no network. All CastOptions instances
are built inline via a minimal stub so the test file has zero imports from
the private application package.

Run with:
    pytest test_polars_cast.py -v
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from yggdrasil.polars.cast import *
from yggdrasil.polars.cast import _apply_tz, _resolve_source_field
from yggdrasil.types.cast.cast_options import CastOptions

_arrow_field_to_pl_field = arrow_field_to_polars_field

def _opts(
    src_type: pa.DataType,
    tgt_type: pa.DataType,
    src_name: str = "x",
    tgt_name: str = "x",
    **kwargs: Any,
) -> CastOptions:
    return CastOptions.check_arg(
        source_field=pa.field(src_name, src_type),
        target_field=pa.field(tgt_name, tgt_type),
        **kwargs,
    )


def _schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(n, t) for n, t in fields])


# ===========================================================================
# Helpers shared across tests
# ===========================================================================

def s(name: str, values: list, dtype: pl.DataType) -> pl.Series:
    return pl.Series(name, values, dtype=dtype)


def assert_series_equal(a: pl.Series, b: pl.Series, *, check_names: bool = True) -> None:
    assert a.equals(b, check_names=check_names)


# ===========================================================================
# arrow_type_to_polars_type
# ===========================================================================

class TestArrowTypeToPolarsType:

    @pytest.mark.parametrize("arrow_t,expected", [
        (pa.int8(),    pl.Int8()),
        (pa.int64(),   pl.Int64()),
        (pa.uint32(),  pl.UInt32()),
        (pa.float32(), pl.Float32()),
        (pa.float64(), pl.Float64()),
        (pa.bool_(),   pl.Boolean()),
        (pa.string(),  pl.Utf8()),
        (pa.binary(),  pl.Binary()),
        (pa.date32(),  pl.Date()),
        (pa.null(),    pl.Null()),
    ])
    def test_primitives(self, arrow_t, expected):
        assert arrow_type_to_polars_type(arrow_t) == expected

    def test_timestamp_us(self):
        result = arrow_type_to_polars_type(pa.timestamp("us"))
        assert result == pl.Datetime("us")

    def test_timestamp_ns_with_tz(self):
        result = arrow_type_to_polars_type(pa.timestamp("ns", tz="UTC"))
        assert result == pl.Datetime("ns", time_zone="UTC")

    def test_timestamp_s_upcasted_to_ms(self):
        result = arrow_type_to_polars_type(pa.timestamp("s"))
        assert result == pl.Datetime("ms")

    def test_date32(self):
        assert arrow_type_to_polars_type(pa.date32()) == pl.Date()

    def test_time64(self):
        assert arrow_type_to_polars_type(pa.time64("us")) == pl.Time()

    def test_duration_us(self):
        assert arrow_type_to_polars_type(pa.duration("us")) == pl.Duration("us")

    def test_duration_s_upcasted_to_ms(self):
        assert arrow_type_to_polars_type(pa.duration("s")) == pl.Duration("ms")

    def test_list(self):
        result = arrow_type_to_polars_type(pa.list_(pa.int32()))
        assert result == pl.List(pl.Int32())

    def test_nested_list(self):
        result = arrow_type_to_polars_type(pa.list_(pa.list_(pa.float64())))
        assert result == pl.List(pl.List(pl.Float64()))

    def test_struct(self):
        t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.utf8())])
        result = arrow_type_to_polars_type(t)
        assert result == pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Utf8())])

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="No Polars equivalent"):
            arrow_type_to_polars_type(pa.month_day_nano_interval())


# ===========================================================================
# polars_type_to_arrow_type
# ===========================================================================

class TestPolarsTypeToArrowType:

    @pytest.mark.parametrize("pl_t,expected", [
        (pl.Int32(),   pa.int32()),
        (pl.Int64(),   pa.int64()),
        (pl.Float64(), pa.float64()),
        (pl.Boolean(), pa.bool_()),
        (pl.Utf8(),    pa.string()),
        (pl.Binary(),  pa.binary()),
        (pl.Date(),    pa.date32()),
        (pl.Null(),    pa.null()),
    ])
    def test_primitives(self, pl_t, expected):
        assert polars_type_to_arrow_type(pl_t) == expected

    def test_datetime_us_no_tz(self):
        assert polars_type_to_arrow_type(pl.Datetime("us")) == pa.timestamp("us")

    def test_datetime_ns_utc(self):
        assert polars_type_to_arrow_type(pl.Datetime("ns", "UTC")) == pa.timestamp("ns", tz="UTC")

    def test_duration(self):
        assert polars_type_to_arrow_type(pl.Duration("ms")) == pa.duration("ms")

    def test_list(self):
        assert polars_type_to_arrow_type(pl.List(pl.Int64())) == pa.list_(pa.int64())

    def test_struct(self):
        result = polars_type_to_arrow_type(
            pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Utf8())])
        )
        assert result == pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.utf8())])

    def test_categorical(self):
        result = polars_type_to_arrow_type(pl.Categorical())
        assert pa.types.is_dictionary(result)

    def test_unknown_raises(self):
        class _Fake(pl.DataType):
            pass
        with pytest.raises(TypeError, match="No Arrow equivalent"):
            polars_type_to_arrow_type(_Fake())

    def test_roundtrip_datetime(self):
        orig = pl.Datetime("ms", "Europe/Paris")
        assert arrow_type_to_polars_type(polars_type_to_arrow_type(orig)) == orig


# ===========================================================================
# _apply_tz
# ===========================================================================

class TestApplyTz:
    """Tests for the internal timezone transition helper."""

    def _eval(self, expr: pl.Expr, dtype: pl.DataType) -> pl.Series:
        return pl.select(expr.cast(dtype)).to_series()

    def _naive_series(self) -> pl.Series:
        return pl.Series("ts", [dt.datetime(2024, 3, 10, 10, 0)], dtype=pl.Datetime("us"))

    def test_both_none_noop(self):
        s = self._naive_series()
        result = pl.select(_apply_tz(pl.lit(s), None, None)).to_series()
        assert result.dtype == pl.Datetime("us")
        assert result[0] == dt.datetime(2024, 3, 10, 10, 0)

    def test_none_to_utc_stamps_tz(self):
        s = self._naive_series()
        result = pl.select(_apply_tz(pl.lit(s), None, "UTC")).to_series()
        assert result.dtype == pl.Datetime("us", "UTC")

    def test_utc_to_paris_converts(self):
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us"))
        result = pl.select(_apply_tz(pl.lit(s), "UTC", "Europe/Paris")).to_series()
        # 12:00 UTC = 13:00 CET
        assert result[0].hour == 13

    def test_utc_to_none_strips_tz(self):
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us"))
        result = pl.select(_apply_tz(pl.lit(s), "UTC", None)).to_series()
        assert result.dtype == pl.Datetime("us")  # no tz

    def test_same_tz_noop(self):
        s = pl.Series("ts", [dt.datetime(2024, 1, 1, 9, 0)], dtype=pl.Datetime("us"))
        result = pl.select(_apply_tz(pl.lit(s), "UTC", "UTC")).to_series()
        assert result.dtype == pl.Datetime("us", "UTC")
        assert result[0].hour == 9


# ===========================================================================
# cast_polars_array_to_temporal — Datetime target
# ===========================================================================

class TestCastToDatetime:

    def _cast(self, series: pl.Series, target: pl.DataType, safe: bool = False, **kw) -> pl.Series:
        return cast_polars_array_to_temporal(series, series.dtype, target, safe=safe, **kw)

    # ── String → Datetime ────────────────────────────────────────────────────

    def test_iso_string_unsafe(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00", "2024-06-01 08:00:00"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.dtype == pl.Datetime("us")
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30, 0)

    def test_date_only_string_unsafe(self):
        s = pl.Series("ts", ["2024-03-15", "2024-12-31"])
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 3, 15, 0, 0)

    def test_dmy_slash_format_unsafe(self):
        s = pl.Series("ts", ["15/01/2024 10:30:00"])
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30, 0)

    def test_string_safe_mode_iso(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00"])
        result = self._cast(s, pl.Datetime("us"), safe=True)
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30, 0)

    def test_mixed_formats_unsafe_coalesces(self):
        s = pl.Series("ts", ["2024-01-15", "15/06/2024"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.null_count() == 0

    def test_unparseable_string_unsafe_yields_null(self):
        s = pl.Series("ts", ["not-a-date"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.null_count() == 1

    # ── Datetime → Datetime (unit conversion) ─────────────────────────────────

    def test_datetime_us_to_ns(self):
        s = pl.Series("ts", [dt.datetime(2024, 1, 1, 12, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("ns"))
        assert result.dtype == pl.Datetime("ns")
        assert result[0] == dt.datetime(2024, 1, 1, 12, 0)

    def test_datetime_us_to_ms(self):
        s = pl.Series("ts", [dt.datetime(2024, 6, 15, 9, 30)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("ms"))
        assert result[0] == dt.datetime(2024, 6, 15, 9, 30)

    # ── Datetime tz conversion ────────────────────────────────────────────────

    def test_utc_to_paris(self):
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us", "UTC"))
        result = self._cast(s, pl.Datetime("us", "Europe/Paris"))
        assert result[0].hour == 13  # UTC+1 in March

    def test_naive_gets_target_tz(self):
        s = pl.Series("ts", [dt.datetime(2024, 6, 1, 9, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("us", "UTC"))
        assert result.dtype == pl.Datetime("us", "UTC")

    def test_source_tz_override(self):
        # Naive series but logically UTC — explicit source_tz
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("us", "Europe/Paris"), source_tz="UTC")
        assert result[0].hour == 13

    # ── Date → Datetime ───────────────────────────────────────────────────────

    def test_date_to_datetime_midnight(self):
        s = pl.Series("d", [dt.date(2024, 1, 15)], dtype=pl.Date())
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 15, 0, 0)

    # ── Integer → Datetime ────────────────────────────────────────────────────

    def test_int64_epoch_us(self):
        epoch_us = 1_704_067_200_000_000  # 2024-01-01T00:00:00 UTC in microseconds
        s = pl.Series("ts", [epoch_us], dtype=pl.Int64())
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 1, 0, 0)

    # ── Duration → Datetime ───────────────────────────────────────────────────

    def test_duration_to_datetime(self):
        s = pl.Series("d", [1_000_000], dtype=pl.Duration("us"))  # 1 second in us
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(1970, 1, 1, 0, 0, 1)

    # ── Expr passthrough ──────────────────────────────────────────────────────

    def test_returns_expr_when_given_expr(self):
        df = pl.DataFrame({"ts": ["2024-01-15T10:30:00"]})
        expr = cast_polars_array_to_temporal(
            pl.col("ts"), pl.Utf8(), pl.Datetime("us"), safe=False
        )
        assert isinstance(expr, pl.Expr)
        result = df.select(expr).to_series()
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30)


# ===========================================================================
# cast_polars_array_to_temporal — Date target
# ===========================================================================

class TestCastToDate:

    def _cast(self, series: pl.Series, safe: bool = False) -> pl.Series:
        return cast_polars_array_to_temporal(series, series.dtype, pl.Date(), safe=safe)

    def test_string_iso(self):
        s = pl.Series("d", ["2024-03-15", "2024-12-31"])
        result = self._cast(s)
        assert result[0] == dt.date(2024, 3, 15)

    def test_string_dmy(self):
        s = pl.Series("d", ["15/03/2024"])
        result = self._cast(s)
        assert result[0] == dt.date(2024, 3, 15)

    def test_string_unparseable_yields_null(self):
        s = pl.Series("d", ["not-a-date"])
        result = self._cast(s)
        assert result.null_count() == 1

    def test_datetime_to_date(self):
        s = pl.Series("d", [dt.datetime(2024, 6, 15, 14, 30)], dtype=pl.Datetime("us"))
        result = self._cast(s)
        assert result[0] == dt.date(2024, 6, 15)

    def test_datetime_tz_aware_correct_calendar_day(self):
        # 2024-03-10 23:30 UTC = 2024-03-11 00:30 in Europe/Paris
        s = pl.Series("d", [dt.datetime(2024, 3, 10, 23, 30)], dtype=pl.Datetime("us"))
        result = cast_polars_array_to_temporal(
            s, s.dtype, pl.Date(), safe=False,
            source_tz="UTC", target_tz="Europe/Paris"
        )
        assert result[0] == dt.date(2024, 3, 11)

    def test_int32_days_since_epoch(self):
        # Day 19723 = 2024-01-01
        s = pl.Series("d", [19723], dtype=pl.Int32())
        result = self._cast(s)
        assert result[0] == dt.date(2024, 1, 1)

    def test_date_noop(self):
        s = pl.Series("d", [dt.date(2024, 6, 1)], dtype=pl.Date())
        result = self._cast(s)
        assert result[0] == dt.date(2024, 6, 1)


# ===========================================================================
# cast_polars_array_to_temporal — Time target
# ===========================================================================

class TestCastToTime:

    def _cast(self, series: pl.Series, safe: bool = False) -> pl.Series:
        return cast_polars_array_to_temporal(series, series.dtype, pl.Time(), safe=safe)

    def test_string_hhmmss(self):
        s = pl.Series("t", ["14:30:00"])
        result = self._cast(s)
        assert result[0] == dt.time(14, 30, 0)

    def test_string_hhmm(self):
        s = pl.Series("t", ["09:15"])
        result = self._cast(s)
        assert result[0] == dt.time(9, 15)

    def test_datetime_extract_time(self):
        s = pl.Series("t", [dt.datetime(2024, 6, 1, 14, 30, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s)
        assert result[0] == dt.time(14, 30, 0)

    def test_datetime_tz_aware_wall_clock(self):
        # 12:00 UTC → 13:00 Europe/Paris (summer)
        s = pl.Series("t", [dt.datetime(2024, 6, 1, 12, 0)], dtype=pl.Datetime("us"))
        result = cast_polars_array_to_temporal(
            s, s.dtype, pl.Time(), safe=False,
            source_tz="UTC", target_tz="Europe/Paris"
        )
        assert result[0].hour == 14

    def test_time_noop(self):
        s = pl.Series("t", [dt.time(10, 0, 0)], dtype=pl.Time())
        result = self._cast(s)
        assert result[0] == dt.time(10, 0, 0)


# ===========================================================================
# cast_polars_array_to_temporal — Duration target
# ===========================================================================

class TestCastToDuration:

    def _cast(self, series: pl.Series, tu: str = "us", safe: bool = False) -> pl.Series:
        return cast_polars_array_to_temporal(
            series, series.dtype, pl.Duration(tu), safe=safe
        )

    def test_int64_to_duration_us(self):
        s = pl.Series("d", [1_000_000], dtype=pl.Int64())  # 1 second
        result = self._cast(s, "us")
        assert result[0] == dt.timedelta(seconds=1)

    def test_duration_tu_conversion(self):
        s = pl.Series("d", [1_000], dtype=pl.Duration("ms"))  # 1 second in ms
        result = self._cast(s, "us")
        assert result[0] == dt.timedelta(seconds=1)

    def test_datetime_to_duration_from_epoch(self):
        # 2024-01-01 00:00:00 UTC = 1_704_067_200 seconds since epoch
        s = pl.Series("ts", [dt.datetime(2024, 1, 1, 0, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, "us")
        assert isinstance(result[0], dt.timedelta)
        assert result[0].days > 0

    def test_datetime_tz_normalised_to_utc_before_diff(self):
        # Two timestamps representing the same UTC instant in different tz
        utc_s = pl.Series("ts", [dt.datetime(2024, 1, 1, 12, 0)], dtype=pl.Datetime("us"))
        paris_s = pl.Series("ts", [dt.datetime(2024, 1, 1, 12, 0)], dtype=pl.Datetime("us"))
        utc_r = cast_polars_array_to_temporal(
            utc_s, utc_s.dtype, pl.Duration("us"), safe=False, source_tz="UTC"
        )
        paris_r = cast_polars_array_to_temporal(
            paris_s, paris_s.dtype, pl.Duration("us"), safe=False, source_tz="UTC"
        )
        assert utc_r[0] == paris_r[0]

    def test_string_unsafe_integer_string(self):
        s = pl.Series("d", ["1000000"], dtype=pl.Utf8())
        result = self._cast(s, "us")
        assert result.null_count() == 0

    def test_string_safe_raises(self):
        s = pl.Series("d", ["1000"], dtype=pl.Utf8())
        with pytest.raises(NotImplementedError):
            self._cast(s, "us", safe=True)

    def test_date_to_duration(self):
        s = pl.Series("d", [dt.date(1970, 1, 2)], dtype=pl.Date())
        result = self._cast(s, "us")
        assert result[0] == dt.timedelta(days=1)


# ===========================================================================
# _resolve_source_field
# ===========================================================================

class TestResolveSourceField:

    @pytest.fixture
    def fields(self) -> list[pl.Field]:
        return [
            pl.Field("Price", pl.Float64()),
            pl.Field("Volume", pl.Int64()),
            pl.Field("Symbol", pl.Utf8()),
        ]

    def test_exact_match(self, fields):
        result = _resolve_source_field("Price", fields, strict_match_names=True)
        assert result is not None
        assert result.name == "Price"

    def test_exact_miss_strict(self, fields):
        result = _resolve_source_field("price", fields, strict_match_names=True)
        assert result is None

    def test_case_insensitive_match(self, fields):
        result = _resolve_source_field("price", fields, strict_match_names=False)
        assert result is not None
        assert result.name == "Price"

    def test_mixed_case(self, fields):
        result = _resolve_source_field("SYMBOL", fields, strict_match_names=False)
        assert result is not None
        assert result.name == "Symbol"

    def test_no_match_returns_none(self, fields):
        result = _resolve_source_field("NonExistent", fields, strict_match_names=False)
        assert result is None


# ===========================================================================
# cast_polars_array_to_struct
# ===========================================================================

class TestCastToStruct:

    def _make_src(self) -> pl.Series:
        return pl.Series("row", [{"price": 100.0, "qty": 10, "sym": "TTF"}]).cast(
            pl.Struct([
                pl.Field("price", pl.Float64()),
                pl.Field("qty", pl.Int64()),
                pl.Field("sym", pl.Utf8()),
            ])
        )

    def _opts(self, **kwargs) -> CastOptions:
        return CastOptions.check_arg(safe=False, **kwargs)

    def test_basic_field_selection(self):
        src = self._make_src()
        target = pl.Struct([pl.Field("price", pl.Float64()), pl.Field("qty", pl.Int64())])
        opts = self._opts(strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        assert set(result.struct.fields) == {"price", "qty"}

    def test_field_type_cast(self):
        src = self._make_src()
        # qty Int64 → Float32
        target = pl.Struct([
            pl.Field("price", pl.Float32()),
            pl.Field("qty", pl.Float32()),
        ])
        opts = self._opts(strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        assert result.struct.field("qty").dtype == pl.Float32()

    def test_case_insensitive_match(self):
        src = self._make_src()
        target = pl.Struct([pl.Field("PRICE", pl.Float64())])
        opts = self._opts(strict_match_names=False, add_missing_columns=False)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        assert "PRICE" in result.struct.fields

    def test_missing_field_raises_without_flag(self):
        src = self._make_src()
        target = pl.Struct([pl.Field("nonexistent", pl.Utf8())])
        opts = self._opts(strict_match_names=True, add_missing_columns=False)
        with pytest.raises(ValueError, match="nonexistent"):
            cast_polars_array_to_struct(src, src.dtype, target, opts)

    def test_missing_field_filled_with_add_missing(self):
        src = self._make_src()
        target = pl.Struct([
            pl.Field("price", pl.Float64()),
            pl.Field("extra", pl.Utf8()),
        ])
        opts = self._opts(strict_match_names=True, add_missing_columns=True)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        assert "extra" in result.struct.fields

    def test_extra_src_fields_kept_with_allow_add(self):
        src = self._make_src()
        target = pl.Struct([pl.Field("price", pl.Float64())])
        opts = self._opts(strict_match_names=True, add_missing_columns=False, allow_add_columns=True)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        fields = result.struct.fields
        assert "qty" in fields
        assert "sym" in fields

    def test_extra_src_fields_dropped_without_allow_add(self):
        src = self._make_src()
        target = pl.Struct([pl.Field("price", pl.Float64())])
        opts = self._opts(strict_match_names=True, add_missing_columns=False, allow_add_columns=False)
        result = cast_polars_array_to_struct(src, src.dtype, target, opts)
        assert result.struct.fields == ["price"]

    def test_nested_struct(self):
        inner_type = pl.Struct([pl.Field("x", pl.Int32()), pl.Field("y", pl.Int32())])
        outer_type = pl.Struct([pl.Field("coord", inner_type), pl.Field("name", pl.Utf8())])
        src = pl.Series("row", [{"coord": {"x": 1, "y": 2}, "name": "A"}]).cast(outer_type)
        # Cast inner x: Int32 → Float64
        tgt_inner = pl.Struct([pl.Field("x", pl.Float64()), pl.Field("y", pl.Float64())])
        tgt = pl.Struct([pl.Field("coord", tgt_inner), pl.Field("name", pl.Utf8())])
        opts = self._opts(strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_struct(src, outer_type, tgt, opts)
        inner_result = result.struct.field("coord")
        assert inner_result.dtype == pl.Struct([
            pl.Field("x", pl.Float64()), pl.Field("y", pl.Float64())
        ])

    def test_returns_expr_for_expr_input(self):
        df = pl.DataFrame({
            "row": pl.Series([{"a": 1, "b": 2}]).cast(
                pl.Struct([pl.Field("a", pl.Int64()), pl.Field("b", pl.Int64())])
            )
        })
        target = pl.Struct([pl.Field("a", pl.Float64())])
        src_dtype = pl.Struct([pl.Field("a", pl.Int64()), pl.Field("b", pl.Int64())])
        opts = CastOptions.check_arg(strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_struct(pl.col("row"), src_dtype, target, opts)
        assert isinstance(result, pl.Expr)


# ===========================================================================
# cast_polars_array_to_list
# ===========================================================================

class TestCastToList:

    def _opts(self, **kwargs) -> CastOptions:
        return CastOptions.check_arg(safe=False, **kwargs)

    def test_list_int_to_list_float(self):
        src = pl.Series("v", [[1, 2, 3], [4, 5]], dtype=pl.List(pl.Int32()))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src, pl.List(pl.Int32()), pl.List(pl.Float64()), opts
        )
        assert result.dtype == pl.List(pl.Float64())
        assert result[0].to_list() == [1.0, 2.0, 3.0]

    def test_list_string_to_list_datetime(self):
        src = pl.Series("v", [["2024-01-15", "2024-06-01"]], dtype=pl.List(pl.Utf8()))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src, pl.List(pl.Utf8()), pl.List(pl.Date()), opts
        )
        assert result.dtype == pl.List(pl.Date())
        assert result[0][0] == dt.date(2024, 1, 15)

    def test_ragged_list_lengths_preserved(self):
        src = pl.Series("v", [[1], [2, 3], [4, 5, 6]], dtype=pl.List(pl.Int32()))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src, pl.List(pl.Int32()), pl.List(pl.Int64()), opts
        )
        assert [len(r) for r in result.to_list()] == [1, 2, 3]

    def test_null_rows_preserved(self):
        src = pl.Series("v", [[1, 2], None, [3]], dtype=pl.List(pl.Int32()))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src, pl.List(pl.Int32()), pl.List(pl.Float64()), opts
        )
        assert result.null_count() == 1
        assert result[0].to_list() == [1.0, 2.0]

    def test_list_inner_same_type_fast_path(self):
        src = pl.Series("v", [[1.0, 2.0]], dtype=pl.List(pl.Float64()))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src, pl.List(pl.Float64()), pl.List(pl.Float64()), opts
        )
        assert result.dtype == pl.List(pl.Float64())

    def test_nested_list_of_list(self):
        src = pl.Series("v", [[[1, 2], [3]]], dtype=pl.List(pl.List(pl.Int32())))
        opts = self._opts()
        result = cast_polars_array_to_list(
            src,
            pl.List(pl.List(pl.Int32())),
            pl.List(pl.List(pl.Float64())),
            opts,
        )
        assert result.dtype == pl.List(pl.List(pl.Float64()))

    def test_list_of_struct(self):
        struct_t = pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Utf8())])
        tgt_struct_t = pl.Struct([pl.Field("a", pl.Float64()), pl.Field("b", pl.Utf8())])
        src = pl.Series("v", [[{"a": 1, "b": "x"}]]).cast(pl.List(struct_t))
        opts = self._opts(strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_list(
            src, pl.List(struct_t), pl.List(tgt_struct_t), opts
        )
        assert result[0][0]["a"] == 1.0

    def test_returns_expr_for_expr_input(self):
        df = pl.DataFrame({"v": pl.Series([[1, 2]], dtype=pl.List(pl.Int32()))})
        opts = self._opts()
        result = cast_polars_array_to_list(
            pl.col("v"), pl.List(pl.Int32()), pl.List(pl.Float64()), opts
        )
        assert isinstance(result, pl.Expr)

    def test_list_of_struct_cast_and_missing_and_expr(self):
        # Source struct: a:int32, b:utf8, extra:int64
        src_struct = pl.Struct([
            pl.Field("a", pl.Int32()),
            pl.Field("b", pl.Utf8()),
            pl.Field("extra", pl.Int64()),
        ])

        # Target struct: a:float64 (cast), b:utf8 (same), c:int64 (missing -> filled)
        tgt_struct = pl.Struct([
            pl.Field("a", pl.Float64()),
            pl.Field("b", pl.Utf8()),
            pl.Field("c", pl.Int64()),
        ])

        src = pl.Series("v", [[{"a": 1, "b": "x", "extra": 9}, {"a": 2, "b": "y", "extra": 10}], None]).cast(
            pl.List(src_struct)
        )

        # Allow missing target fields and keep extra source fields
        opts = self._opts(strict_match_names=True, add_missing_columns=True, allow_add_columns=True)

        # ── Eager Series path ────────────────────────────────────────────────
        result = cast_polars_array_to_list(
            src,
            pl.List(src_struct),
            pl.List(tgt_struct),
            opts,
        )

        assert isinstance(result, pl.Series)
        assert result.dtype == pl.List(tgt_struct)

        # Row 0: inner structs cast + missing 'c' filled + extra preserved
        row0 = result[0]
        assert row0[0]["a"] == 1.0
        assert row0[1]["a"] == 2.0
        assert row0[0]["b"] == "x"
        assert row0[0]["c"] is None
        # extra is not in tgt_struct -> must be dropped by cast to target dtype
        with pytest.raises(KeyError):
            _ = row0[0]["extra"]

        # Row 1 remains null
        assert result.null_count() == 1

        # ── Lazy Expr path (must return Expr and evaluate correctly) ─────────
        df = pl.DataFrame({"v": src})
        expr = cast_polars_array_to_list(
            pl.col("v"),
            pl.List(src_struct),
            pl.List(tgt_struct),
            opts,
        )
        assert isinstance(expr, pl.Expr)

        out = df.select(expr.alias("out"))["out"]
        assert out.dtype == pl.List(tgt_struct)
        assert out.to_list() == result.to_list()

    def test_dataframe_nested_list_of_struct_cast_and_fill_missing(self):
        # Source: trades is List(Struct(trade_id:str, qty:int32))
        trade_src = pl.Struct([
            pl.Field("trade_id", pl.Utf8()),
            pl.Field("qty", pl.Int32()),
        ])
        df = pl.DataFrame({
            "asof": ["2024-01-15", "2024-01-16"],
            "trades": pl.Series(
                [
                    [{"trade_id": "T1", "qty": 10}, {"trade_id": "T2", "qty": 20}],
                    [{"trade_id": "T3", "qty": 5}],
                ],
                dtype=pl.List(trade_src),
            ),
        })

        # Target: add missing top-level "desk", and enrich trades with missing inner "venue"
        # Also cast qty -> float64
        schema = pa.schema([
            pa.field("asof", pa.date32()),
            pa.field("desk", pa.string()),  # missing top-level -> fill
            pa.field("trades", pa.list_(
                pa.struct([
                    pa.field("trade_id", pa.string()),
                    pa.field("qty", pa.float64()),  # cast
                    pa.field("venue", pa.string()),  # missing inner -> fill
                ])
            )),
        ])

        opts = CastOptions.check_arg(
            target_field=schema,
            safe=False,
            strict_match_names=True,
            add_missing_columns=True,
            allow_add_columns=False,
        )

        out = cast_polars_dataframe(df, opts)

        # --- Top-level checks ---
        assert out.columns == ["asof", "desk", "trades"]
        assert len(out) == len(df)
        assert out["asof"].dtype == pl.Date()
        assert out["desk"].null_count() == len(df)

        # --- Nested dtype check ---
        trade_tgt = pl.Struct([
            pl.Field("trade_id", pl.Utf8()),
            pl.Field("qty", pl.Float64()),
            pl.Field("venue", pl.Utf8()),
        ])
        assert out["trades"].dtype == pl.List(trade_tgt)

        # --- Nested value checks ---
        row0 = out["trades"][0]
        assert row0[0]["trade_id"] == "T1"
        assert row0[0]["qty"] == 10.0
        assert row0[0]["venue"] is None  # filled missing inner field

        row1 = out["trades"][1]
        assert row1[0]["trade_id"] == "T3"
        assert row1[0]["qty"] == 5.0
        assert row1[0]["venue"] is None

    def test_lazyframe_nested_list_of_struct_cast_and_fill_missing(self):
        trade_src = pl.Struct([pl.Field("trade_id", pl.Utf8()), pl.Field("qty", pl.Int32())])
        lf = pl.DataFrame({
            "asof": ["2024-01-15", "2024-01-16"],
            "trades": pl.Series(
                [
                    [{"trade_id": "T1", "qty": 10}, {"trade_id": "T2", "qty": 20}],
                    [{"trade_id": "T3", "qty": 5}],
                ],
                dtype=pl.List(trade_src),
            ),
        }).lazy()

        schema = pa.schema([
            pa.field("asof", pa.date32()),
            pa.field("desk", pa.string()),
            pa.field("trades", pa.list_(pa.struct([
                pa.field("trade_id", pa.string()),
                pa.field("qty", pa.float64()),
                pa.field("venue", pa.string()),
            ]))),
        ])

        opts = CastOptions.check_arg(
            target_field=schema,
            safe=False,
            strict_match_names=True,
            add_missing_columns=True,
            allow_add_columns=False,
        )

        out_lf = cast_polars_lazyframe(lf, opts)
        assert isinstance(out_lf, pl.LazyFrame)

        out = out_lf.collect()
        assert out["desk"].null_count() == len(out)
        assert out["trades"][0][0]["qty"] == 10.0
        assert out["trades"][0][0]["venue"] is None

# ===========================================================================
# cast_polars_dataframe
# ===========================================================================

class TestCastPolarsDataframe:

    def _df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "price":  pl.Series([100.5, 200.0, 150.75], dtype=pl.Float64()),
            "volume": pl.Series([1000, 2000, 500],      dtype=pl.Int32()),
            "symbol": pl.Series(["TTF", "NBP", "HH"],   dtype=pl.Utf8()),
        })

    def _cast(self, df: pl.DataFrame, schema: pa.Schema, **kw) -> pl.DataFrame:
        opts = CastOptions.check_arg(target_field=schema, **kw)
        return cast_polars_dataframe(df, opts)

    # ── No-op ─────────────────────────────────────────────────────────────────

    def test_no_schema_returns_unchanged(self):
        df = self._df()
        result = cast_polars_dataframe(df, CastOptions.check_arg())
        assert result.shape == df.shape

    # ── Basic column selection and cast ───────────────────────────────────────

    def test_selects_and_casts_columns(self):
        df = self._df()
        schema = _schema(("price", pa.float32()), ("symbol", pa.string()))
        result = self._cast(df, schema)
        assert result.columns == ["price", "symbol"]
        assert result["price"].dtype == pl.Float32()

    def test_column_order_follows_target_schema(self):
        df = self._df()
        schema = _schema(("symbol", pa.string()), ("price", pa.float64()))
        result = self._cast(df, schema)
        assert result.columns[0] == "symbol"
        assert result.columns[1] == "price"

    # ── Name matching ─────────────────────────────────────────────────────────

    def test_exact_match_strict(self):
        df = self._df()
        schema = _schema(("PRICE", pa.float64()))
        with pytest.raises(pa.ArrowInvalid, match="PRICE"):
            self._cast(df, schema, strict_match_names=True, add_missing_columns=False)

    def test_case_insensitive_match(self):
        df = self._df()
        schema = _schema(("PRICE", pa.float64()), ("VOLUME", pa.int32()))
        result = self._cast(df, schema, strict_match_names=False, add_missing_columns=False)
        assert set(result.columns) == {"PRICE", "VOLUME"}

    def test_positional_fallback(self):
        # Source: [price, volume, symbol], Target: [x, y] — positional
        df = self._df()
        schema = _schema(("x", pa.float64()), ("y", pa.int32()))
        result = self._cast(df, schema, strict_match_names=False, add_missing_columns=False)
        assert result.columns == ["x", "y"]
        # x should contain price values, y should contain volume values
        assert result["x"][0] == pytest.approx(100.5)

    # ── Missing columns ───────────────────────────────────────────────────────

    def test_missing_column_raises_by_default(self):
        df = self._df()
        schema = _schema(("nonexistent", pa.int64()))
        with pytest.raises(pa.ArrowInvalid, match="nonexistent"):
            self._cast(df, schema, strict_match_names=True, add_missing_columns=False)

    def test_missing_column_filled_nullable(self):
        df = self._df()
        schema = _schema(("price", pa.float64()), ("missing", pa.int64()))
        result = self._cast(df, schema, strict_match_names=True, add_missing_columns=True)
        assert "missing" in result.columns
        assert result["missing"].null_count() == len(df)

    def test_missing_column_filled_non_nullable(self):
        df = self._df()
        tgt_field = pa.field("missing", pa.int32(), nullable=False)
        schema = pa.schema([pa.field("price", pa.float64()), tgt_field])
        opts = CastOptions.check_arg(
            target_field=schema,
            strict_match_names=True,
            add_missing_columns=True,
        )
        result = cast_polars_dataframe(df, opts)
        assert result["missing"].null_count() == 0

    # ── Extra source columns ──────────────────────────────────────────────────

    def test_extra_src_dropped_by_default(self):
        df = self._df()
        schema = _schema(("price", pa.float64()))
        result = self._cast(df, schema)
        assert result.columns == ["price"]

    def test_extra_src_kept_with_allow_add(self):
        df = self._df()
        schema = _schema(("price", pa.float64()))
        result = self._cast(df, schema, allow_add_columns=True)
        assert set(result.columns) == {"price", "volume", "symbol"}

    def test_extra_cols_not_duplicated(self):
        # price is in target AND source, should appear once
        df = self._df()
        schema = _schema(("price", pa.float64()))
        result = self._cast(df, schema, allow_add_columns=True)
        assert result.columns.count("price") == 1

    # ── Temporal casting in DataFrame context ─────────────────────────────────

    def test_string_column_to_datetime(self):
        df = pl.DataFrame({"ts": ["2024-01-15T10:30:00", "2024-06-01T08:00:00"]})
        schema = _schema(("ts", pa.timestamp("us")))
        result = self._cast(df, schema, safe=False)
        assert result["ts"].dtype == pl.Datetime("us")

    def test_integer_column_to_date(self):
        df = pl.DataFrame({"d": pl.Series([19723, 19724], dtype=pl.Int32())})
        schema = _schema(("d", pa.date32()))
        result = self._cast(df, schema, safe=False)
        assert result["d"].dtype == pl.Date()

    # ── Row count preservation ────────────────────────────────────────────────

    def test_row_count_preserved(self):
        df = self._df()
        schema = _schema(("price", pa.float64()), ("symbol", pa.string()))
        result = self._cast(df, schema)
        assert len(result) == len(df)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32())})
        schema = _schema(("a", pa.int64()))
        result = self._cast(df, schema)
        assert len(result) == 0
        assert result["a"].dtype == pl.Int64()


# ===========================================================================
# cast_polars_lazyframe
# ===========================================================================

class TestCastPolarsLazyframe:

    def _lf(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "price":  pl.Series([100.5, 200.0], dtype=pl.Float64()),
            "volume": pl.Series([1000, 2000],   dtype=pl.Int32()),
            "symbol": pl.Series(["TTF", "NBP"], dtype=pl.Utf8()),
        }).lazy()

    def _cast(self, lf: pl.LazyFrame, schema: pa.Schema, **kw) -> pl.LazyFrame:
        opts = CastOptions.check_arg(target_field=schema, **kw)
        return cast_polars_lazyframe(lf, opts)

    # ── Returns a LazyFrame (does not collect) ────────────────────────────────

    def test_returns_lazyframe(self):
        lf = self._lf()
        schema = _schema(("price", pa.float32()))
        result = self._cast(lf, schema)
        assert isinstance(result, pl.LazyFrame)

    def test_no_schema_returns_unchanged(self):
        lf = self._lf()
        result = cast_polars_lazyframe(lf, CastOptions.check_arg())
        assert isinstance(result, pl.LazyFrame)

    # ── Correctness after collect ─────────────────────────────────────────────

    def test_selects_and_casts(self):
        lf = self._lf()
        schema = _schema(("price", pa.float32()), ("symbol", pa.string()))
        result = self._cast(lf, schema).collect()
        assert result.columns == ["price", "symbol"]
        assert result["price"].dtype == pl.Float32()

    def test_case_insensitive_match(self):
        lf = self._lf()
        schema = _schema(("PRICE", pa.float64()))
        result = self._cast(lf, schema, strict_match_names=False).collect()
        assert "PRICE" in result.columns

    def test_positional_fallback(self):
        lf = self._lf()
        schema = _schema(("x", pa.float64()), ("y", pa.int32()))
        result = self._cast(lf, schema, strict_match_names=False).collect()
        assert result.columns == ["x", "y"]

    def test_missing_column_raises_eagerly(self):
        # Error must surface before collect()
        lf = self._lf()
        schema = _schema(("ghost", pa.int64()))
        with pytest.raises(pa.ArrowInvalid, match="ghost"):
            self._cast(lf, schema, strict_match_names=True, add_missing_columns=False)

    def test_missing_column_filled_with_lit(self):
        lf = self._lf()
        schema = _schema(("price", pa.float64()), ("missing", pa.int64()))
        result = self._cast(lf, schema, strict_match_names=True, add_missing_columns=True).collect()
        assert "missing" in result.columns
        assert result["missing"].null_count() == len(result)

    def test_extra_src_kept_with_allow_add(self):
        lf = self._lf()
        schema = _schema(("price", pa.float64()))
        result = self._cast(lf, schema, allow_add_columns=True).collect()
        assert set(result.columns) == {"price", "volume", "symbol"}

    def test_string_to_datetime_stays_lazy(self):
        lf = pl.DataFrame({"ts": ["2024-01-15T10:30:00"]}).lazy()
        schema = _schema(("ts", pa.timestamp("us")))
        result_lf = self._cast(lf, schema, safe=False)
        assert isinstance(result_lf, pl.LazyFrame)
        result = result_lf.collect()
        assert result["ts"].dtype == pl.Datetime("us")
        assert result["ts"][0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_matches_dataframe_result(self):
        """LazyFrame and DataFrame cast must produce identical output."""
        df = pl.DataFrame({
            "ts":  pl.Series(["2024-01-15", "2024-06-01"], dtype=pl.Utf8()),
            "val": pl.Series([1, 2], dtype=pl.Int32()),
        })
        schema = _schema(("ts", pa.date32()), ("val", pa.int64()))
        opts = CastOptions.check_arg(target_field=schema, safe=False)
        df_result = cast_polars_dataframe(df, opts)
        lf_result = cast_polars_lazyframe(df.lazy(), opts).collect()

        assert df_result.equals(lf_result)

    def test_empty_lazyframe(self):
        lf = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32())}).lazy()
        schema = _schema(("a", pa.int64()))
        result = self._cast(lf, schema).collect()
        assert len(result) == 0
        assert result["a"].dtype == pl.Int64()


# ===========================================================================
# cast_polars_array — dispatch integration
# ===========================================================================

class TestCastPolarsArrayDispatch:

    def test_no_cast_needed_returns_as_is(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64())
        opts = CastOptions.check_arg(
            source_field=pa.field("x", pa.int64()),
            target_field=pa.field("x", pa.int64()),
        )
        result = cast_polars_array(s, opts)
        assert_series_equal(result, s)

    def test_scalar_cast(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int32())
        opts = CastOptions.check_arg(
            source_field=pa.field("x", pa.int32()),
            target_field=pa.field("x", pa.int64()),
        )
        result = cast_polars_array(s, opts)
        assert result.dtype == pl.Int64()

    def test_temporal_dispatch(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00"], dtype=pl.Utf8())
        opts = CastOptions.check_arg(
            source_field=pa.field("ts", pa.string()),
            target_field=pa.field("ts", pa.timestamp("us")),
            safe=False,
        )
        result = cast_polars_array(s, opts)
        assert result.dtype == pl.Datetime("us")

    def test_null_source_nullable_target(self):
        s = pl.Series("x", [None, None], dtype=pl.Null())
        opts = CastOptions.check_arg(
            source_field=pa.field("x", pa.null()),
            target_field=pa.field("x", pa.int64(), nullable=True),
        )
        result = cast_polars_array(s, opts)
        assert result.null_count() == 2

    def test_name_aliased_to_target(self):
        s = pl.Series("src_col", [1.0, 2.0], dtype=pl.Float64())
        opts = CastOptions.check_arg(
            source_field=pa.field("src_col", pa.float64()),
            target_field=pa.field("tgt_col", pa.float32()),
        )
        result = cast_polars_array(s, opts)
        assert result.name == "tgt_col"

    def test_expr_input_returns_expr(self):
        opts = CastOptions.check_arg(
            source_field=pa.field("x", pa.int32()),
            target_field=pa.field("x", pa.int64()),
        )
        result = cast_polars_array(pl.col("x"), opts)
        assert isinstance(result, pl.Expr)


# ===========================================================================
# arrow_field_to_polars_field / polars_field_to_arrow_field roundtrip
# ===========================================================================

class TestFieldRoundtrip:

    @pytest.mark.parametrize("arrow_t", [
        pa.int32(), pa.float64(), pa.string(), pa.bool_(),
        pa.timestamp("us"), pa.timestamp("ns", tz="UTC"),
        pa.date32(), pa.duration("ms"),
        pa.list_(pa.int64()),
        pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.float64())]),
    ])
    def test_arrow_to_polars_to_arrow_roundtrip(self, arrow_t):
        arrow_field = pa.field("col", arrow_t)
        pl_field = arrow_field_to_polars_field(arrow_field)
        back = polars_field_to_arrow_field(pl_field)
        assert back.type == arrow_field.type
        assert back.name == arrow_field.name

    def test_nullable_preserved(self):
        f = pa.field("x", pa.int32(), nullable=False)
        pl_f = arrow_field_to_polars_field(f)
        back = polars_field_to_arrow_field(pl_f)
        assert back.nullable == False  # noqa: E712

    def test_nullable_default_true(self):
        f = pa.field("x", pa.float64(), nullable=True)
        pl_f = arrow_field_to_polars_field(f)
        back = polars_field_to_arrow_field(pl_f)
        assert back.nullable == True  # noqa: E712


# ===========================================================================
# Commodity trading integration scenarios
# ===========================================================================

class TestCommodityScenarios:
    """End-to-end scenarios representative of commodity trading data."""

    def test_gas_tick_schema_cast(self):
        """Raw tick data (strings, ints) → typed schema."""
        raw = pl.DataFrame({
            "ts":     ["2024-03-10T14:30:00.123", "2024-03-10T14:30:01.456"],
            "price":  ["45.32", "45.35"],
            "volume": [100, 250],
            "hub":    ["TTF", "NBP"],
        })
        schema = pa.schema([
            pa.field("ts",     pa.timestamp("ms")),
            pa.field("price",  pa.float64()),
            pa.field("volume", pa.int64()),
            pa.field("hub",    pa.string()),
        ])
        opts = CastOptions.check_arg(target_field=schema, safe=False)
        result = cast_polars_dataframe(raw, opts)
        assert result["ts"].dtype == pl.Datetime("ms")
        assert result["price"].dtype == pl.Float64()
        assert result["price"][0] == pytest.approx(45.32)

    def test_forward_curve_date_pivot(self):
        """Delivery dates stored as YYYYMMDD integers."""
        raw = pl.DataFrame({
            "delivery": pl.Series(["20240401", "20240501", "20240601"], dtype=pl.String()),
            "price":    pl.Series([42.0, 43.5, 44.1], dtype=pl.Float64()),
        })
        schema = _schema(("delivery", pa.date32()), ("price", pa.float64()))
        opts = CastOptions.check_arg(target_field=schema, safe=False)
        result = cast_polars_dataframe(raw, opts)
        assert result["delivery"].to_list()[0] == dt.date(2024, 4, 1)

    def test_timezone_normalisation_utc_to_cet(self):
        """Exchange timestamps in UTC normalised to CET for settlement."""
        utc_ts = pl.Series(
            "ts",
            [dt.datetime(2024, 3, 10, 23, 0), dt.datetime(2024, 3, 11, 12, 0)],
            dtype=pl.Datetime("us"),
        )
        result = cast_polars_array_to_temporal(
            utc_ts, utc_ts.dtype,
            pl.Datetime("us", "Europe/Paris"),
            safe=False,
            source_tz="UTC",
        )
        # 23:00 UTC = 00:00 CET next day in winter
        assert result[0].day == 11

    def test_struct_trade_record_field_selection(self):
        """Select and recast fields from a nested trade struct."""
        trade_t = pl.Struct([
            pl.Field("trade_id",  pl.Utf8()),
            pl.Field("price",     pl.Float64()),
            pl.Field("quantity",  pl.Int64()),
            pl.Field("direction", pl.Utf8()),
        ])
        src = pl.Series("trade", [
            {"trade_id": "T001", "price": 45.5, "quantity": 100, "direction": "BUY"},
        ]).cast(trade_t)

        # Target: only price and quantity, quantity as Float64
        tgt_t = pl.Struct([
            pl.Field("price",    pl.Float64()),
            pl.Field("quantity", pl.Float64()),
        ])
        opts = CastOptions.check_arg(safe=False, strict_match_names=True, add_missing_columns=False)
        result = cast_polars_array_to_struct(src, pl.Field("", trade_t), pl.Field("", tgt_t), opts)
        assert result.struct.field("quantity")[0] == 100.0
        assert "direction" not in result.struct.fields

    def test_list_of_settlement_prices(self):
        """List of daily settlement prices cast from Int to Float."""
        src = pl.Series("settlements", [
            [4530, 4535, 4540],
            [4520, 4525],
        ], dtype=pl.List(pl.Int32()))
        opts = CastOptions.check_arg(safe=False)
        result = cast_polars_array_to_list(
            src, pl.List(pl.Int32()), pl.List(pl.Float64()), opts
        )
        assert result[0].to_list() == [4530.0, 4535.0, 4540.0]

    def test_lazy_pipeline_full_schema(self):
        """Full lazy pipeline: string dates + ints → typed schema, stays lazy."""
        raw = pl.DataFrame({
            "trade_date": ["2024-01-15", "2024-01-16"],
            "price":      ["45.5", "46.0"],
            "volume":     [500, 1000],
        }).lazy()

        schema = pa.schema([
            pa.field("trade_date", pa.date32()),
            pa.field("price",      pa.float64()),
            pa.field("volume",     pa.int64()),
        ])
        opts = CastOptions.check_arg(target_field=schema, safe=False)
        result_lf = cast_polars_lazyframe(raw, opts)
        assert isinstance(result_lf, pl.LazyFrame)
        result = result_lf.collect()
        assert result["trade_date"][0] == dt.date(2024, 1, 15)
        assert result["price"][0] == pytest.approx(45.5)
        assert result["volume"].dtype == pl.Int64()