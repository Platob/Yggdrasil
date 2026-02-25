"""Unit tests for polars_cast module.

Matches the current implementation signatures exactly:
  - CastOptions(source_field=..., target_field=...) constructor
  - cast_polars_array_to_struct(array, options)
  - cast_polars_array_to_list(array, options)
  - cast_polars_array_to_bool(array, options)
  - cast_polars_array(array, options)
  - cast_polars_dataframe(df, options)
  - cast_polars_lazyframe(lf, options)

Run with:
    pytest test_polars_cast.py -v
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from yggdrasil.polars.cast import (
    cast_polars_array,
    cast_polars_array_to_bool,
    cast_polars_array_to_list,
    cast_polars_array_to_struct,
    cast_polars_array_to_temporal,
    cast_polars_dataframe,
    cast_polars_lazyframe,
    arrow_field_to_polars_field,
    arrow_type_to_polars_type,
    polars_field_to_arrow_field,
    polars_type_to_arrow_type,
    _apply_tz,
    _resolve_source_field,
)
from yggdrasil.data.cast import CastOptions


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _opts(
    src_type: pa.DataType,
    tgt_type: pa.DataType,
    src_name: str = "x",
    tgt_name: str = "x",
    **kwargs: Any,
) -> CastOptions:
    return CastOptions(
        source_field=pa.field(src_name, src_type),
        target_field=pa.field(tgt_name, tgt_type),
        **kwargs,
    )


def _schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(n, t) for n, t in fields])


def _cast_df(df: pl.DataFrame, schema: pa.Schema, **kw) -> pl.DataFrame:
    return cast_polars_dataframe(df, CastOptions(target_field=schema, **kw))


def _cast_lf(lf: pl.LazyFrame, schema: pa.Schema, **kw) -> pl.DataFrame:
    return cast_polars_lazyframe(lf, CastOptions(target_field=schema, **kw)).collect()


# ===========================================================================
# arrow_type_to_polars_type
# ===========================================================================

class TestArrowTypeToPolarsType:

    @pytest.mark.parametrize("arrow_t,expected", [
        (pa.int8(),    pl.Int8()),
        (pa.int16(),   pl.Int16()),
        (pa.int32(),   pl.Int32()),
        (pa.int64(),   pl.Int64()),
        (pa.uint8(),   pl.UInt8()),
        (pa.uint16(),  pl.UInt16()),
        (pa.uint32(),  pl.UInt32()),
        (pa.uint64(),  pl.UInt64()),
        (pa.float32(), pl.Float32()),
        (pa.float64(), pl.Float64()),
        (pa.bool_(),   pl.Boolean()),
        (pa.string(),  pl.Utf8()),
        (pa.large_string(), pl.Utf8()),
        (pa.binary(),  pl.Binary()),
        (pa.large_binary(), pl.Binary()),
        (pa.date32(),  pl.Date()),
        (pa.null(),    pl.Null()),
    ])
    def test_primitives(self, arrow_t, expected):
        assert arrow_type_to_polars_type(arrow_t) == expected

    def test_timestamp_us(self):
        assert arrow_type_to_polars_type(pa.timestamp("us")) == pl.Datetime("us")

    def test_timestamp_ns_utc(self):
        assert arrow_type_to_polars_type(pa.timestamp("ns", tz="UTC")) == pl.Datetime("ns", "UTC")

    def test_timestamp_s_upcasted_to_ms(self):
        assert arrow_type_to_polars_type(pa.timestamp("s")) == pl.Datetime("ms")

    def test_time64_us(self):
        assert arrow_type_to_polars_type(pa.time64("us")) == pl.Time()

    def test_duration_us(self):
        assert arrow_type_to_polars_type(pa.duration("us")) == pl.Duration("us")

    def test_duration_s_upcasted_to_ms(self):
        assert arrow_type_to_polars_type(pa.duration("s")) == pl.Duration("ms")

    def test_list_int32(self):
        assert arrow_type_to_polars_type(pa.list_(pa.int32())) == pl.List(pl.Int32())

    def test_nested_list(self):
        assert arrow_type_to_polars_type(pa.list_(pa.list_(pa.float64()))) == pl.List(pl.List(pl.Float64()))

    def test_struct(self):
        t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.utf8())])
        assert arrow_type_to_polars_type(t) == pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Utf8())])

    def test_dictionary_to_categorical(self):
        assert arrow_type_to_polars_type(pa.dictionary(pa.int32(), pa.string())) == pl.Categorical()

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="No Polars equivalent"):
            arrow_type_to_polars_type(pa.month_day_nano_interval())


# ===========================================================================
# polars_type_to_arrow_type
# ===========================================================================

class TestPolarsTypeToArrowType:

    @pytest.mark.parametrize("pl_t,expected", [
        (pl.Int8(),    pa.int8()),
        (pl.Int32(),   pa.int32()),
        (pl.Int64(),   pa.int64()),
        (pl.UInt32(),  pa.uint32()),
        (pl.Float32(), pa.float32()),
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

    def test_duration_ms(self):
        assert polars_type_to_arrow_type(pl.Duration("ms")) == pa.duration("ms")

    def test_list(self):
        assert polars_type_to_arrow_type(pl.List(pl.Int64())) == pa.list_(pa.int64())

    def test_struct(self):
        result = polars_type_to_arrow_type(
            pl.Struct([pl.Field("a", pl.Int32()), pl.Field("b", pl.Utf8())])
        )
        assert result == pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.utf8())])

    def test_categorical(self):
        assert pa.types.is_dictionary(polars_type_to_arrow_type(pl.Categorical()))

    def test_unknown_raises(self):
        class _Fake(pl.DataType):
            pass
        with pytest.raises(TypeError, match="No Arrow equivalent"):
            polars_type_to_arrow_type(_Fake())

    def test_roundtrip_datetime_with_tz(self):
        orig = pl.Datetime("ms", "Europe/Paris")
        assert arrow_type_to_polars_type(polars_type_to_arrow_type(orig)) == orig


# ===========================================================================
# Field roundtrip
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
        af = pa.field("col", arrow_t)
        pf = arrow_field_to_polars_field(af)
        back = polars_field_to_arrow_field(pf)
        assert back.type == af.type
        assert back.name == af.name

    def test_nullable_false_preserved(self):
        f = pa.field("x", pa.int32(), nullable=False)
        back = polars_field_to_arrow_field(arrow_field_to_polars_field(f))
        assert back.nullable is False

    def test_nullable_true_preserved(self):
        f = pa.field("x", pa.float64(), nullable=True)
        back = polars_field_to_arrow_field(arrow_field_to_polars_field(f))
        assert back.nullable is True


# ===========================================================================
# _apply_tz
# ===========================================================================

class TestApplyTz:

    def _s(self, hour: int = 12, tz: str | None = None) -> pl.Series:
        return pl.Series("ts", [dt.datetime(2024, 3, 10, hour, 0)], dtype=pl.Datetime("us", tz))

    def test_both_none_noop(self):
        result = pl.select(_apply_tz(pl.lit(self._s()), None, None)).to_series()
        assert result.dtype == pl.Datetime("us")
        assert result[0] == dt.datetime(2024, 3, 10, 12, 0)

    def test_none_to_utc_stamps_tz(self):
        result = pl.select(_apply_tz(pl.lit(self._s()), None, "UTC")).to_series()
        assert result.dtype == pl.Datetime("us", "UTC")

    def test_utc_to_paris_converts(self):
        result = pl.select(_apply_tz(pl.lit(self._s()), "UTC", "Europe/Paris")).to_series()
        assert result[0].hour == 13  # UTC+1 in March

    def test_utc_to_none_strips_tz(self):
        result = pl.select(_apply_tz(pl.lit(self._s()), "UTC", None)).to_series()
        assert result.dtype == pl.Datetime("us")

    def test_same_tz_stamps_without_conversion(self):
        result = pl.select(_apply_tz(pl.lit(self._s()), "UTC", "UTC")).to_series()
        assert result.dtype == pl.Datetime("us", "UTC")
        assert result[0].hour == 12


# ===========================================================================
# _resolve_source_field
# ===========================================================================

class TestResolveSourceField:

    @pytest.fixture
    def fields(self) -> list[pl.Field]:
        return [
            pl.Field("Price",  pl.Float64()),
            pl.Field("Volume", pl.Int64()),
            pl.Field("Symbol", pl.Utf8()),
        ]

    def test_exact_match(self, fields):
        result = _resolve_source_field("Price", fields, strict_match_names=True)
        assert result.name == "Price"

    def test_exact_miss_strict_returns_none(self, fields):
        assert _resolve_source_field("price", fields, strict_match_names=True) is None

    def test_case_insensitive(self, fields):
        result = _resolve_source_field("price", fields, strict_match_names=False)
        assert result.name == "Price"

    def test_mixed_case(self, fields):
        result = _resolve_source_field("SYMBOL", fields, strict_match_names=False)
        assert result.name == "Symbol"

    def test_no_match_returns_none(self, fields):
        assert _resolve_source_field("Ghost", fields, strict_match_names=False) is None


# ===========================================================================
# cast_polars_array_to_temporal — Datetime
# ===========================================================================

class TestCastToDatetime:

    def _cast(self, s: pl.Series, target: pl.DataType, safe: bool = False, **kw) -> pl.Series:
        return cast_polars_array_to_temporal(s, s.dtype, target, safe=safe, **kw)

    def test_iso_string_unsafe(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00", "2024-06-01 08:00:00"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.dtype == pl.Datetime("us")
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_date_only_string(self):
        s = pl.Series("ts", ["2024-03-15"])
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 3, 15, 0, 0)

    def test_dmy_slash_format(self):
        s = pl.Series("ts", ["15/01/2024 10:30:00"])
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_unparseable_yields_null_unsafe(self):
        s = pl.Series("ts", ["not-a-date"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.null_count() == 1

    def test_mixed_formats_no_nulls(self):
        s = pl.Series("ts", ["2024-01-15", "15/06/2024"])
        result = self._cast(s, pl.Datetime("us"))
        assert result.null_count() == 0

    def test_string_safe_iso_only(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00"])
        result = self._cast(s, pl.Datetime("us"), safe=True)
        assert result[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_datetime_unit_conversion_us_to_ns(self):
        s = pl.Series("ts", [dt.datetime(2024, 1, 1, 12, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("ns"))
        assert result.dtype == pl.Datetime("ns")

    def test_datetime_tz_conversion_utc_to_paris(self):
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us", "UTC"))
        result = self._cast(s, pl.Datetime("us", "Europe/Paris"))
        assert result[0].hour == 13

    def test_naive_gets_target_tz(self):
        s = pl.Series("ts", [dt.datetime(2024, 6, 1, 9, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("us", "UTC"))
        assert result.dtype == pl.Datetime("us", "UTC")

    def test_source_tz_override(self):
        s = pl.Series("ts", [dt.datetime(2024, 3, 10, 12, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, pl.Datetime("us", "Europe/Paris"), source_tz="UTC")
        assert result[0].hour == 13

    def test_date_to_datetime_midnight(self):
        s = pl.Series("d", [dt.date(2024, 1, 15)], dtype=pl.Date())
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 15, 0, 0)

    def test_int64_epoch_us(self):
        s = pl.Series("ts", [1_704_067_200_000_000], dtype=pl.Int64())
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(2024, 1, 1, 0, 0)

    def test_duration_to_datetime(self):
        s = pl.Series("d", [1_000_000], dtype=pl.Duration("us"))
        result = self._cast(s, pl.Datetime("us"))
        assert result[0] == dt.datetime(1970, 1, 1, 0, 0, 1)

    def test_returns_expr_when_given_expr(self):
        df = pl.DataFrame({"ts": ["2024-01-15T10:30:00"]})
        expr = cast_polars_array_to_temporal(
            pl.col("ts"), pl.Utf8(), pl.Datetime("us"), safe=False
        )
        assert isinstance(expr, pl.Expr)
        assert df.select(expr).to_series()[0] == dt.datetime(2024, 1, 15, 10, 30)


# ===========================================================================
# cast_polars_array_to_temporal — Date
# ===========================================================================

class TestCastToDate:

    def _cast(self, s: pl.Series, safe: bool = False, **kw) -> pl.Series:
        return cast_polars_array_to_temporal(s, s.dtype, pl.Date(), safe=safe, **kw)

    def test_string_iso(self):
        s = pl.Series("d", ["2024-03-15", "2024-12-31"])
        result = self._cast(s)
        assert result[0] == dt.date(2024, 3, 15)

    def test_string_dmy_slash(self):
        s = pl.Series("d", ["15/03/2024"])
        result = self._cast(s)
        assert result[0] == dt.date(2024, 3, 15)

    def test_string_yyyymmdd(self):
        s = pl.Series("d", ["20240401"])
        result = self._cast(s)
        assert result[0] == dt.date(2024, 4, 1)

    def test_unparseable_yields_null(self):
        s = pl.Series("d", ["not-a-date"])
        result = self._cast(s)
        assert result.null_count() == 1

    def test_datetime_to_date(self):
        s = pl.Series("d", [dt.datetime(2024, 6, 15, 14, 30)], dtype=pl.Datetime("us"))
        result = self._cast(s)
        assert result[0] == dt.date(2024, 6, 15)

    def test_datetime_tz_correct_calendar_day(self):
        # 2024-03-10 23:30 UTC = 2024-03-11 00:30 Europe/Paris
        s = pl.Series("d", [dt.datetime(2024, 3, 10, 23, 30)], dtype=pl.Datetime("us"))
        result = cast_polars_array_to_temporal(
            s, s.dtype, pl.Date(), safe=False,
            source_tz="UTC", target_tz="Europe/Paris",
        )
        assert result[0] == dt.date(2024, 3, 11)

    def test_int32_days_since_epoch(self):
        s = pl.Series("d", [19723], dtype=pl.Int32())  # 2024-01-01
        result = self._cast(s)
        assert result[0] == dt.date(2024, 1, 1)

    def test_date_noop(self):
        s = pl.Series("d", [dt.date(2024, 6, 1)], dtype=pl.Date())
        assert self._cast(s)[0] == dt.date(2024, 6, 1)


# ===========================================================================
# cast_polars_array_to_temporal — Time
# ===========================================================================

class TestCastToTime:

    def _cast(self, s: pl.Series, safe: bool = False, **kw) -> pl.Series:
        return cast_polars_array_to_temporal(s, s.dtype, pl.Time(), safe=safe, **kw)

    def test_string_hhmmss(self):
        s = pl.Series("t", ["14:30:00"])
        assert self._cast(s)[0] == dt.time(14, 30, 0)

    def test_string_hhmm(self):
        s = pl.Series("t", ["09:15"])
        assert self._cast(s)[0] == dt.time(9, 15)

    def test_datetime_extract_time(self):
        s = pl.Series("t", [dt.datetime(2024, 6, 1, 14, 30, 0)], dtype=pl.Datetime("us"))
        assert self._cast(s)[0] == dt.time(14, 30, 0)

    def test_datetime_tz_wall_clock(self):
        # 12:00 UTC → 14:00 Europe/Paris in summer (UTC+2)
        s = pl.Series("t", [dt.datetime(2024, 6, 1, 12, 0)], dtype=pl.Datetime("us"))
        result = cast_polars_array_to_temporal(
            s, s.dtype, pl.Time(), safe=False,
            source_tz="UTC", target_tz="Europe/Paris",
        )
        assert result[0].hour == 14

    def test_time_noop(self):
        s = pl.Series("t", [dt.time(10, 0, 0)], dtype=pl.Time())
        assert self._cast(s)[0] == dt.time(10, 0, 0)


# ===========================================================================
# cast_polars_array_to_temporal — Duration
# ===========================================================================

class TestCastToDuration:

    def _cast(self, s: pl.Series, tu: str = "us", safe: bool = False) -> pl.Series:
        return cast_polars_array_to_temporal(s, s.dtype, pl.Duration(tu), safe=safe)

    def test_int64_to_duration_us(self):
        s = pl.Series("d", [1_000_000], dtype=pl.Int64())
        assert self._cast(s, "us")[0] == dt.timedelta(seconds=1)

    def test_duration_unit_conversion_ms_to_us(self):
        s = pl.Series("d", [1_000], dtype=pl.Duration("ms"))
        assert self._cast(s, "us")[0] == dt.timedelta(seconds=1)

    def test_datetime_to_duration_from_epoch(self):
        s = pl.Series("ts", [dt.datetime(2024, 1, 1, 0, 0)], dtype=pl.Datetime("us"))
        result = self._cast(s, "us")
        assert isinstance(result[0], dt.timedelta)
        assert result[0].days > 0

    def test_string_integer_unsafe(self):
        s = pl.Series("d", ["1000000"], dtype=pl.Utf8())
        assert self._cast(s, "us").null_count() == 0

    def test_string_safe_raises(self):
        s = pl.Series("d", ["1000"], dtype=pl.Utf8())
        with pytest.raises(NotImplementedError):
            self._cast(s, "us", safe=True)

    def test_date_to_duration_one_day(self):
        s = pl.Series("d", [dt.date(1970, 1, 2)], dtype=pl.Date())
        assert self._cast(s, "us")[0] == dt.timedelta(days=1)


# ===========================================================================
# cast_polars_array_to_bool
# ===========================================================================

class TestCastToBool:

    def _cast(self, s: pl.Series, safe: bool = False) -> pl.Series:
        opts = _opts(
            polars_type_to_arrow_type(s.dtype),
            pa.bool_(),
            safe=safe,
        )
        return cast_polars_array_to_bool(s, opts)

    def test_bool_noop(self):
        s = pl.Series("x", [True, False, True], dtype=pl.Boolean())
        result = self._cast(s)
        assert result.to_list() == [True, False, True]

    def test_int_nonzero_is_true(self):
        s = pl.Series("x", [0, 1, -1, 42], dtype=pl.Int64())
        result = self._cast(s)
        assert result.to_list() == [False, True, True, True]

    def test_uint_nonzero_is_true(self):
        s = pl.Series("x", [0, 1, 255], dtype=pl.UInt8())
        result = self._cast(s)
        assert result.to_list() == [False, True, True]

    def test_float_nonzero_is_true(self):
        s = pl.Series("x", [0.0, 1.5, -0.1], dtype=pl.Float64())
        result = self._cast(s)
        assert result.to_list() == [False, True, False]

    @pytest.mark.parametrize("value,expected", [
        ("true",  True),
        ("True",  True),
        ("TRUE",  True),
        ("1",     True),
        ("yes",   True),
        ("on",    True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0",     False),
        ("no",    False),
        ("off",   False),
    ])
    def test_string_keywords(self, value, expected):
        s = pl.Series("x", [value], dtype=pl.Utf8())
        result = self._cast(s)
        assert result[0] == expected

    def test_string_unrecognised_yields_null_unsafe(self):
        s = pl.Series("x", ["maybe", "unknown"], dtype=pl.Utf8())
        result = self._cast(s, safe=False)
        assert result.null_count() == 2

    def test_string_unrecognised_raises_safe(self):
        s = pl.Series("x", ["maybe"], dtype=pl.Utf8())
        with pytest.raises(Exception):
            self._cast(s, safe=True)

    def test_null_source_yields_null(self):
        s = pl.Series("x", [None, None], dtype=pl.Null())
        opts = _opts(pa.null(), pa.bool_())
        result = cast_polars_array_to_bool(s, opts)
        assert result.null_count() == 2

    def test_returns_expr_for_expr_input(self):
        opts = _opts(pa.string(), pa.bool_())
        result = cast_polars_array_to_bool(pl.col("x"), opts)
        assert isinstance(result, pl.Expr)

    def test_string_null_yields_null(self):
        s = pl.Series("x", [None, "true"], dtype=pl.Utf8())
        result = self._cast(s)
        assert result[0] is None
        assert result[1] is True


# ===========================================================================
# cast_polars_array_to_struct
# ===========================================================================

class TestCastPolarsArrayToStruct:

    def _opts(
        self,
        src_type: pa.DataType,
        tgt_type: pa.DataType,
        **kwargs,
    ) -> CastOptions:
        return CastOptions(
            source_field=pa.field("s", src_type),
            target_field=pa.field("s", tgt_type),
            **kwargs,
        )

    def test_flat_struct_round_trip(self):
        src_t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        tgt_t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        s = pl.Series("s", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        result = cast_polars_array_to_struct(s, self._opts(src_t, tgt_t))
        rows = result.struct.unnest()
        assert rows["a"].to_list() == [1, 2]
        assert rows["b"].to_list() == ["x", "y"]

    def test_field_type_upcast(self):
        src_t = pa.struct([pa.field("v", pa.int32())])
        tgt_t = pa.struct([pa.field("v", pa.int64())])
        s = pl.Series("s", [{"v": 1}, {"v": 2}])
        result = cast_polars_array_to_struct(s, self._opts(src_t, tgt_t))
        assert result.struct.field("v").dtype == pl.Int64()

    def test_nested_struct(self):
        src_t = pa.struct([pa.field("outer", pa.struct([pa.field("inner", pa.int32())]))])
        tgt_t = pa.struct([pa.field("outer", pa.struct([pa.field("inner", pa.int64())]))])
        s = pl.Series("s", [{"outer": {"inner": 42}}])
        result = cast_polars_array_to_struct(s, self._opts(src_t, tgt_t))
        inner = result.struct.field("outer").struct.field("inner")
        assert inner[0] == 42
        assert inner.dtype == pl.Int64()

    def test_missing_field_filled_nullable(self):
        src_t = pa.struct([pa.field("a", pa.int32())])
        tgt_t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string(), nullable=True)])
        s = pl.Series("s", [{"a": 1}, {"a": 2}])
        result = cast_polars_array_to_struct(s, self._opts(src_t, tgt_t, add_missing_columns=True))
        assert result.struct.field("b").null_count() == 2

    def test_json_string_to_struct(self):
        src_t = pa.string()
        tgt_t = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        s = pl.Series("s", ['{"a": 1, "b": "hello"}', '{"a": 2, "b": "world"}'])
        result = cast_polars_array_to_struct(s, self._opts(src_t, tgt_t))
        assert result.struct.field("a").to_list() == [1, 2]

    def test_returns_expr_for_expr_input(self):
        src_t = pa.struct([pa.field("a", pa.int32())])
        tgt_t = pa.struct([pa.field("a", pa.int64())])
        opts = self._opts(src_t, tgt_t)
        result = cast_polars_array_to_struct(pl.col("s"), opts)
        assert isinstance(result, pl.Expr)

    def test_case_insensitive_field_match(self):
        src_t = pa.struct([pa.field("PRICE", pa.float64())])
        tgt_t = pa.struct([pa.field("price", pa.float64())])
        s = pl.Series("s", [{"PRICE": 42.0}])
        result = cast_polars_array_to_struct(
            s, self._opts(src_t, tgt_t, strict_match_names=False)
        )
        assert result.struct.field("price")[0] == pytest.approx(42.0)


# ===========================================================================
# cast_polars_array_to_list
# ===========================================================================

class TestCastPolarsArrayToList:

    def _opts(
        self,
        src_type: pa.DataType,
        tgt_type: pa.DataType,
        **kwargs,
    ) -> CastOptions:
        return CastOptions(
            source_field=pa.field("lst", src_type),
            target_field=pa.field("lst", tgt_type),
            **kwargs,
        )

    def test_list_int_round_trip(self):
        s = pl.Series("lst", [[1, 2], [3, 4, 5]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(s, self._opts(pa.list_(pa.int32()), pa.list_(pa.int32())))
        assert result.to_list() == [[1, 2], [3, 4, 5]]

    def test_list_inner_type_upcast(self):
        s = pl.Series("lst", [[1, 2], [3]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(s, self._opts(pa.list_(pa.int32()), pa.list_(pa.int64())))
        assert result.dtype == pl.List(pl.Int64())

    def test_list_string(self):
        s = pl.Series("lst", [["a", "b"], ["c"]], dtype=pl.List(pl.Utf8()))
        result = cast_polars_array_to_list(s, self._opts(pa.list_(pa.string()), pa.list_(pa.string())))
        assert result.to_list() == [["a", "b"], ["c"]]

    def test_large_list(self):
        s = pl.Series("lst", [[1, 2], [3]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.int32()), pa.large_list(pa.int64()))
        )
        assert result.shape == (2,)

    def test_fixed_size_list_inner_cast(self):
        s = pl.Series("lst", [[1, 2, 3], [4, 5, 6]], dtype=pl.Array(pl.Int32(), 3))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.int32(), 3), pa.list_(pa.int64()))
        )
        assert result[0].to_list() == [1, 2, 3]

    def test_fixed_size_list_same_inner_type_fast_path(self):
        s = pl.Series("lst", [[1, 2], [3, 4]], dtype=pl.Array(pl.Int32(), 2))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.int32(), 2), pa.list_(pa.int32()))
        )
        assert result.shape == (2,)

    def test_scalar_wrapped_into_list(self):
        s = pl.Series("lst", [1, 2, 3], dtype=pl.Int32())
        result = cast_polars_array_to_list(
            s, self._opts(pa.int32(), pa.list_(pa.int64()))
        )
        assert result.to_list() == [[1], [2], [3]]

    def test_list_of_structs_inner_cast(self):
        src_t = pa.list_(pa.struct([pa.field("v", pa.int32())]))
        tgt_t = pa.list_(pa.struct([pa.field("v", pa.int64())]))
        s = pl.Series("lst", [[{"v": 1}, {"v": 2}], [{"v": 3}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        assert result[0].to_list() == [{"v": 1}, {"v": 2}]

    def test_empty_inner_list(self):
        s = pl.Series("lst", [[], [1, 2]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.int32()), pa.list_(pa.int64()))
        )
        assert result[0].to_list() == []

    def test_null_outer_row_preserved(self):
        s = pl.Series("lst", [[1], None, [3]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.int32()), pa.list_(pa.int64()))
        )
        assert result[1] is None

    def test_returns_expr_for_expr_input(self):
        opts = self._opts(pa.list_(pa.int32()), pa.list_(pa.int64()))
        result = cast_polars_array_to_list(pl.col("lst"), opts)
        assert isinstance(result, pl.Expr)

    def test_deeply_nested_list(self):
        s = pl.Series("lst", [[[1, 2], [3]], [[4]]], dtype=pl.List(pl.List(pl.Int32())))
        result = cast_polars_array_to_list(
            s, self._opts(pa.list_(pa.list_(pa.int32())), pa.list_(pa.list_(pa.int64())))
        )
        assert result.shape == (2,)


# ===========================================================================
# cast_polars_array — dispatch
# ===========================================================================

class TestCastPolarsArrayDispatch:

    def test_noop_when_types_match(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64())
        opts = _opts(pa.int64(), pa.int64())
        result = cast_polars_array(s, opts)
        assert result.dtype == pl.Int64()
        assert result.to_list() == [1, 2, 3]

    def test_scalar_cast_int_to_float(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int32())
        result = cast_polars_array(s, _opts(pa.int32(), pa.float64()))
        assert result.dtype == pl.Float64()

    def test_temporal_dispatch(self):
        s = pl.Series("ts", ["2024-01-15T10:30:00"], dtype=pl.Utf8())
        result = cast_polars_array(s, _opts(pa.string(), pa.timestamp("us"), safe=False))
        assert result.dtype == pl.Datetime("us")

    def test_bool_dispatch(self):
        s = pl.Series("x", [0, 1, 2], dtype=pl.Int32())
        result = cast_polars_array(s, _opts(pa.int32(), pa.bool_()))
        assert result.to_list() == [False, True, True]

    def test_struct_dispatch(self):
        src_t = pa.struct([pa.field("a", pa.int32())])
        tgt_t = pa.struct([pa.field("a", pa.int64())])
        s = pl.Series("x", [{"a": 1}, {"a": 2}])
        result = cast_polars_array(s, _opts(src_t, tgt_t))
        assert result.struct.field("a").dtype == pl.Int64()

    def test_list_dispatch(self):
        s = pl.Series("x", [[1, 2], [3]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array(s, _opts(pa.list_(pa.int32()), pa.list_(pa.int64())))
        assert result.dtype == pl.List(pl.Int64())

    def test_null_source_nullable_target_fills_null(self):
        s = pl.Series("x", [None, None], dtype=pl.Null())
        result = cast_polars_array(
            s, CastOptions(
                source_field=pa.field("x", pa.null()),
                target_field=pa.field("x", pa.int64(), nullable=True),
            )
        )
        assert result.null_count() == 2

    def test_null_source_non_nullable_fills_default(self):
        s = pl.Series("x", [None, None], dtype=pl.Null())
        result = cast_polars_array(
            s, CastOptions(
                source_field=pa.field("x", pa.null()),
                target_field=pa.field("x", pa.int32(), nullable=False),
            )
        )
        assert result.null_count() == 0
        assert result.to_list() == [0, 0]

    def test_nullability_fill_applied_after_cast(self):
        s = pl.Series("x", [1, None, 3], dtype=pl.Int32())
        result = cast_polars_array(
            s, CastOptions(
                source_field=pa.field("x", pa.int32(), nullable=True),
                target_field=pa.field("x", pa.int64(), nullable=False),
            )
        )
        assert result.null_count() == 0

    def test_name_aliased_to_target(self):
        s = pl.Series("src_col", [1.0], dtype=pl.Float64())
        result = cast_polars_array(
            s, CastOptions(
                source_field=pa.field("src_col", pa.float64()),
                target_field=pa.field("tgt_col", pa.float32()),
            )
        )
        assert result.name == "tgt_col"

    def test_expr_input_returns_expr(self):
        result = cast_polars_array(pl.col("x"), _opts(pa.int32(), pa.int64()))
        assert isinstance(result, pl.Expr)


# ===========================================================================
# cast_polars_dataframe
# ===========================================================================

class TestCastPolarsDataframe:

    @pytest.fixture
    def market_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "price":  pl.Series([100.5, 200.0, 150.75], dtype=pl.Float64()),
            "volume": pl.Series([1000, 2000, 500],       dtype=pl.Int32()),
            "symbol": pl.Series(["TTF", "NBP", "HH"],    dtype=pl.Utf8()),
        })

    def test_no_schema_passthrough(self, market_df):
        result = cast_polars_dataframe(market_df, CastOptions())
        assert result.equals(market_df)

    def test_selects_and_casts_columns(self, market_df):
        schema = _schema(("price", pa.float32()), ("symbol", pa.string()))
        result = _cast_df(market_df, schema)
        assert result.columns == ["price", "symbol"]
        assert result["price"].dtype == pl.Float32()

    def test_column_order_follows_target_schema(self, market_df):
        schema = _schema(("symbol", pa.string()), ("price", pa.float64()))
        result = _cast_df(market_df, schema)
        assert result.columns == ["symbol", "price"]

    def test_exact_name_strict_raises_on_mismatch(self, market_df):
        schema = _schema(("PRICE", pa.float64()))
        with pytest.raises(pa.ArrowInvalid, match="PRICE"):
            _cast_df(market_df, schema, strict_match_names=True, add_missing_columns=False)

    def test_case_insensitive_match(self, market_df):
        schema = _schema(("PRICE", pa.float64()), ("VOLUME", pa.int32()))
        result = _cast_df(market_df, schema, strict_match_names=False, add_missing_columns=False)
        assert set(result.columns) == {"PRICE", "VOLUME"}

    def test_positional_fallback(self, market_df):
        schema = _schema(("x", pa.float64()), ("y", pa.int64()))
        result = _cast_df(market_df, schema, strict_match_names=False, add_missing_columns=False)
        assert result.columns == ["x", "y"]
        assert result["x"][0] == pytest.approx(100.5)

    def test_missing_column_raises_by_default(self, market_df):
        schema = _schema(("ghost", pa.int64()))
        with pytest.raises(pa.ArrowInvalid, match="ghost"):
            _cast_df(market_df, schema, strict_match_names=True, add_missing_columns=False)

    def test_missing_column_filled_nullable(self, market_df):
        schema = _schema(("price", pa.float64()), ("missing", pa.int64()))
        result = _cast_df(market_df, schema, strict_match_names=True, add_missing_columns=True)
        assert result["missing"].null_count() == len(market_df)

    def test_missing_column_filled_non_nullable(self, market_df):
        schema = pa.schema([
            pa.field("price", pa.float64()),
            pa.field("fill",  pa.int32(), nullable=False),
        ])
        result = _cast_df(market_df, schema, strict_match_names=True, add_missing_columns=True)
        assert result["fill"].null_count() == 0
        assert result["fill"].to_list() == [0, 0, 0]

    def test_extra_columns_dropped_by_default(self, market_df):
        schema = _schema(("price", pa.float64()))
        result = _cast_df(market_df, schema)
        assert result.columns == ["price"]

    def test_extra_columns_preserved_allow_add(self, market_df):
        schema = _schema(("price", pa.float64()))
        result = _cast_df(market_df, schema, allow_add_columns=True)
        assert set(result.columns) == {"price", "volume", "symbol"}

    def test_no_duplicate_column_with_allow_add(self, market_df):
        schema = _schema(("price", pa.float64()))
        result = _cast_df(market_df, schema, allow_add_columns=True)
        assert result.columns.count("price") == 1

    def test_string_to_datetime(self):
        df = pl.DataFrame({"ts": ["2024-01-15T10:30:00", "2024-06-01T08:00:00"]})
        result = _cast_df(df, _schema(("ts", pa.timestamp("us"))), safe=False)
        assert result["ts"].dtype == pl.Datetime("us")

    def test_string_tz_to_datetime(self):
        df = pl.DataFrame({"ts": ["2026-02-17 03:00+01:00", "2026-02-18 01:00+01:00"]})
        result = _cast_df(df, _schema(("ts", pa.timestamp("us", "UTC"))), safe=False)
        assert result["ts"].dtype == pl.Datetime("us", "UTC")
        assert result["ts"][0] == dt.datetime(2026, 2, 17, 2, 0, 0, tzinfo=dt.timezone.utc)
        assert result["ts"][1] == dt.datetime(2026, 2, 18, 0, 0, 0, tzinfo=dt.timezone.utc)

    def test_list_string_tz_to_datetime(self):
        df = pl.DataFrame({
            "ts": [
                [
                    {"data": "2026-02-17 03:00+01:00", "value": "123"},
                    {"data": "2026-02-18 01:00+01:00", "value": "456"},  # → UTC 2026-02-18 00:00
                ]
            ]
        })

        result = _cast_df(df, _schema(("ts", pa.list_(
            pa.struct([
                pa.field("data", pa.timestamp("us", "UTC")),  # case matches source keys
                pa.field("value", pa.float64()),
            ])))), safe=False
        )

        # The column is a List[Struct], so drill in:
        ts_structs = result["ts"].explode()  # Series of Struct rows
        data_col = ts_structs.struct.field("data")  # Series of Datetime[us, UTC]
        value_col = ts_structs.struct.field("value")  # Series of Float64

        assert result["ts"].dtype == pl.List(
            pl.Struct({"data": pl.Datetime("us", "UTC"), "value": pl.Float64})
        )

        assert data_col[0] == dt.datetime(2026, 2, 17, 2, 0, 0, tzinfo=dt.timezone.utc)
        assert data_col[1] == dt.datetime(2026, 2, 18, 0, 0, 0, tzinfo=dt.timezone.utc)

        assert value_col[0] == 123.0
        assert value_col[1] == 456.0

    def test_int_to_date(self):
        df = pl.DataFrame({"d": pl.Series([19723, 19724], dtype=pl.Int32())})
        result = _cast_df(df, _schema(("d", pa.date32())), safe=False)
        assert result["d"].dtype == pl.Date()

    def test_row_count_preserved(self, market_df):
        schema = _schema(("price", pa.float64()))
        assert len(_cast_df(market_df, schema)) == len(market_df)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32())})
        result = _cast_df(df, _schema(("a", pa.int64())))
        assert len(result) == 0
        assert result["a"].dtype == pl.Int64()


# ===========================================================================
# cast_polars_lazyframe
# ===========================================================================

class TestCastPolarsLazyframe:

    @pytest.fixture
    def market_lf(self) -> pl.LazyFrame:
        return pl.DataFrame({
            "price":  pl.Series([100.5, 200.0], dtype=pl.Float64()),
            "volume": pl.Series([1000, 2000],   dtype=pl.Int32()),
            "symbol": pl.Series(["TTF", "NBP"], dtype=pl.Utf8()),
        }).lazy()

    def test_returns_lazyframe_before_collect(self, market_lf):
        result = cast_polars_lazyframe(
            market_lf, CastOptions(target_field=_schema(("price", pa.float32())))
        )
        assert isinstance(result, pl.LazyFrame)

    def test_no_schema_passthrough(self, market_lf):
        result = cast_polars_lazyframe(market_lf, CastOptions())
        assert isinstance(result, pl.LazyFrame)

    def test_selects_and_casts(self, market_lf):
        result = _cast_lf(market_lf, _schema(("price", pa.float32()), ("symbol", pa.string())))
        assert result.columns == ["price", "symbol"]
        assert result["price"].dtype == pl.Float32()

    def test_case_insensitive_match(self, market_lf):
        result = _cast_lf(market_lf, _schema(("PRICE", pa.float64())), strict_match_names=False)
        assert "PRICE" in result.columns

    def test_positional_fallback(self, market_lf):
        result = _cast_lf(
            market_lf, _schema(("x", pa.float64()), ("y", pa.int32())),
            strict_match_names=False,
        )
        assert result.columns == ["x", "y"]

    def test_missing_column_raises_before_collect(self, market_lf):
        with pytest.raises(pa.ArrowInvalid, match="ghost"):
            cast_polars_lazyframe(
                market_lf,
                CastOptions(
                    target_field=_schema(("ghost", pa.int64())),
                    strict_match_names=True,
                    add_missing_columns=False,
                ),
            )

    def test_missing_column_filled_with_lit(self, market_lf):
        result = _cast_lf(
            market_lf,
            _schema(("price", pa.float64()), ("missing", pa.int64())),
            strict_match_names=True,
            add_missing_columns=True,
        )
        assert result["missing"].null_count() == len(result)

    def test_extra_columns_preserved(self, market_lf):
        result = _cast_lf(
            market_lf, _schema(("price", pa.float64())), allow_add_columns=True
        )
        assert set(result.columns) == {"price", "volume", "symbol"}

    def test_string_to_datetime_stays_lazy(self, market_lf):
        lf = pl.DataFrame({"ts": ["2024-01-15T10:30:00"]}).lazy()
        result_lf = cast_polars_lazyframe(
            lf, CastOptions(target_field=_schema(("ts", pa.timestamp("us"))), safe=False)
        )
        assert isinstance(result_lf, pl.LazyFrame)
        result = result_lf.collect()
        assert result["ts"][0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_matches_dataframe_output(self):
        df = pl.DataFrame({
            "ts":  ["2024-01-15", "2024-06-01"],
            "val": pl.Series([1, 2], dtype=pl.Int32()),
        })
        schema = _schema(("ts", pa.date32()), ("val", pa.int64()))
        opts = CastOptions(target_field=schema, safe=False)
        df_result = cast_polars_dataframe(df, opts)
        lf_result = cast_polars_lazyframe(df.lazy(), opts).collect()
        assert df_result.equals(lf_result)

    def test_empty_lazyframe(self):
        lf = pl.DataFrame({"a": pl.Series([], dtype=pl.Int32())}).lazy()
        result = _cast_lf(lf, _schema(("a", pa.int64())))
        assert len(result) == 0
        assert result["a"].dtype == pl.Int64()

    def test_plan_inspectable_without_collect(self, market_lf):
        result_lf = cast_polars_lazyframe(
            market_lf, CastOptions(target_field=_schema(("price", pa.float32())))
        )
        assert isinstance(result_lf.explain(), str)


# ===========================================================================
# List of structs
# ===========================================================================

class TestListOfStructs:

    def _opts(self, src_t: pa.DataType, tgt_t: pa.DataType, **kwargs) -> CastOptions:
        return CastOptions(
            source_field=pa.field("lst", src_t),
            target_field=pa.field("lst", tgt_t),
            **kwargs,
        )

    def test_round_trip(self):
        t = pa.list_(pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())]))
        s = pl.Series("lst", [[{"a": 1, "b": "x"}], [{"a": 2, "b": "y"}, {"a": 3, "b": "z"}]])
        result = cast_polars_array_to_list(s, self._opts(t, t))
        rows = result.to_list()
        assert rows[0] == [{"a": 1, "b": "x"}]
        assert rows[1] == [{"a": 2, "b": "y"}, {"a": 3, "b": "z"}]

    def test_inner_struct_field_upcast(self):
        src_t = pa.list_(pa.struct([pa.field("v", pa.int32())]))
        tgt_t = pa.list_(pa.struct([pa.field("v", pa.int64())]))
        s = pl.Series("lst", [[{"v": 1}, {"v": 2}], [{"v": 3}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        inner_dtype = result.dtype.inner
        assert inner_dtype == pl.Struct([pl.Field("v", pl.Int64())])

    def test_inner_struct_missing_field_filled(self):
        src_t = pa.list_(pa.struct([pa.field("a", pa.int32())]))
        tgt_t = pa.list_(pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string(), nullable=True)]))
        s = pl.Series("lst", [[{"a": 1}], [{"a": 2}, {"a": 3}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t, add_missing_columns=True))
        rows = result.to_list()
        assert all("b" in row for sublist in rows for row in sublist)
        assert all(row["b"] is None for sublist in rows for row in sublist)

    def test_nested_struct_in_list(self):
        src_t = pa.list_(pa.struct([pa.field("outer", pa.struct([pa.field("inner", pa.int32())]))]))
        tgt_t = pa.list_(pa.struct([pa.field("outer", pa.struct([pa.field("inner", pa.int64())]))]))
        s = pl.Series("lst", [[{"outer": {"inner": 1}}], [{"outer": {"inner": 2}}, {"outer": {"inner": 3}}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        rows = result.to_list()
        assert rows[0][0]["outer"]["inner"] == 1
        assert rows[1][1]["outer"]["inner"] == 3

    def test_large_list_of_structs(self):
        src_t = pa.large_list(pa.struct([pa.field("x", pa.int32())]))
        tgt_t = pa.large_list(pa.struct([pa.field("x", pa.int64())]))
        s = pl.Series("lst", [[{"x": 1}], [{"x": 2}, {"x": 3}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        assert result.to_list()[1] == [{"x": 2}, {"x": 3}]

    def test_fixed_size_list_of_structs(self):
        src_t = pa.list_(pa.struct([pa.field("v", pa.int32())]), 2)
        tgt_t = pa.list_(pa.struct([pa.field("v", pa.int64())]))
        s = pl.Series("lst", [[{"v": 1}, {"v": 2}], [{"v": 3}, {"v": 4}]], dtype=pl.Array(pl.Struct([pl.Field("v", pl.Int32())]), 2))
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        assert result.to_list()[0] == [{"v": 1}, {"v": 2}]

    def test_empty_inner_lists(self):
        src_t = pa.list_(pa.struct([pa.field("a", pa.int32())]))
        tgt_t = pa.list_(pa.struct([pa.field("a", pa.int64())]))
        s = pl.Series("lst", [[], [{"a": 1}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        assert result.to_list()[0] == []
        assert result.to_list()[1] == [{"a": 1}]

    def test_null_outer_rows_preserved(self):
        src_t = pa.list_(pa.struct([pa.field("a", pa.int32())]))
        tgt_t = pa.list_(pa.struct([pa.field("a", pa.int64())]))
        s = pl.Series("lst", [[{"a": 1}], None, [{"a": 3}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t))
        rows = result.to_list()
        assert rows[0] == [{"a": 1}]
        assert rows[1] is None
        assert rows[2] == [{"a": 3}]

    def test_struct_with_temporal_field(self):
        src_t = pa.list_(pa.struct([pa.field("name", pa.string()), pa.field("dt", pa.string())]))
        tgt_t = pa.list_(pa.struct([pa.field("name", pa.string()), pa.field("dt", pa.date32())]))
        s = pl.Series("lst", [[{"name": "foo", "dt": "2024-01-01"}], [{"name": "bar", "dt": "2024-06-15"}]])
        result = cast_polars_array_to_list(s, self._opts(src_t, tgt_t, safe=False))
        rows = result.to_list()
        assert rows[0][0]["dt"] == dt.date(2024, 1, 1)
        assert rows[1][0]["dt"] == dt.date(2024, 6, 15)


# ===========================================================================
# Commodity trading integration
# ===========================================================================

class TestCommodityIntegration:

    def test_gas_tick_schema_cast(self):
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
        result = cast_polars_dataframe(raw, CastOptions(target_field=schema, safe=False))
        assert result["ts"].dtype == pl.Datetime("ms")
        assert result["price"].dtype == pl.Float64()
        assert result["price"][0] == pytest.approx(45.32)

    def test_forward_curve_yyyymmdd_string_to_date(self):
        raw = pl.DataFrame({
            "delivery": ["20240401", "20240501", "20240601"],
            "price":    pl.Series([42.0, 43.5, 44.1], dtype=pl.Float64()),
        })
        schema = _schema(("delivery", pa.date32()), ("price", pa.float64()))
        result = cast_polars_dataframe(raw, CastOptions(target_field=schema, safe=False))
        assert result["delivery"][0] == dt.date(2024, 4, 1)

    def test_utc_to_cet_settlement(self):
        s = pl.Series(
            "ts",
            [dt.datetime(2024, 3, 10, 23, 0), dt.datetime(2024, 3, 11, 12, 0)],
            dtype=pl.Datetime("us"),
        )
        result = cast_polars_array_to_temporal(
            s, s.dtype, pl.Datetime("us", "Europe/Paris"),
            safe=False, source_tz="UTC",
        )
        assert result[0].day == 11  # 23:00 UTC = 00:00 CET next day

    def test_trade_struct_field_cast(self):
        src_t = pa.struct([
            pa.field("trade_id",  pa.string()),
            pa.field("price",     pa.float64()),
            pa.field("quantity",  pa.int64()),
            pa.field("direction", pa.string()),
        ])
        tgt_t = pa.struct([
            pa.field("price",    pa.float64()),
            pa.field("quantity", pa.float64()),
        ])
        s = pl.Series("trade", [{"trade_id": "T001", "price": 45.5, "quantity": 100, "direction": "BUY"}])
        result = cast_polars_array_to_struct(
            s, CastOptions(source_field=pa.field("trade", src_t), target_field=pa.field("trade", tgt_t))
        )
        assert result.struct.field("quantity")[0] == 100.0
        assert "direction" not in [f.name for f in result.dtype.fields]

    def test_settlement_price_list_int_to_float(self):
        s = pl.Series("settlements", [[4530, 4535, 4540], [4520, 4525]], dtype=pl.List(pl.Int32()))
        result = cast_polars_array_to_list(
            s, CastOptions(
                source_field=pa.field("settlements", pa.list_(pa.int32())),
                target_field=pa.field("settlements", pa.list_(pa.float64())),
            )
        )
        assert result[0].to_list() == [4530.0, 4535.0, 4540.0]

    def test_lazy_pipeline_full_schema(self):
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
        result_lf = cast_polars_lazyframe(raw, CastOptions(target_field=schema, safe=False))
        assert isinstance(result_lf, pl.LazyFrame)
        result = result_lf.collect()
        assert result["trade_date"][0] == dt.date(2024, 1, 15)
        assert result["price"][0] == pytest.approx(45.5)
        assert result["volume"].dtype == pl.Int64()