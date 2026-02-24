"""Unit tests for cast_polars_dataframe and cast_polars_lazyframe.

Coverage
────────
- Every primitive Polars dtype (int, uint, float, bool, string, binary, date, null)
- Temporal types: Date, Time, Datetime (all time units × tz-aware/naive), Duration
- Nested types: List, Array (fixed-size), Struct (flat + nested)
- Categorical / Enum
- Decimal
- Column matching: exact, case-insensitive, positional
- Missing columns: add_missing_columns=True/False
- Extra columns: allow_add_columns=True/False
- Nullability fill
- LazyFrame mirrors DataFrame semantics (lazy, no collect until tested)
"""
from __future__ import annotations

import datetime
from decimal import Decimal

import polars as pl
import pyarrow as pa
import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal CastOptions pointing at a target Arrow schema
# ---------------------------------------------------------------------------

def make_options(
    schema: pa.Schema,
    *,
    safe: bool = True,
    strict_match_names: bool = True,
    add_missing_columns: bool = False,
    allow_add_columns: bool = False,
):
    """Construct CastOptions with a target schema and sensible defaults."""
    from yggdrasil.data.cast import CastOptions

    return CastOptions(
        target_field=pa.field("__frame__", pa.struct(list(schema))),
        safe=safe,
        strict_match_names=strict_match_names,
        add_missing_columns=add_missing_columns,
        allow_add_columns=allow_add_columns,
    )


def cast_df(df: pl.DataFrame, schema: pa.Schema, **kw) -> pl.DataFrame:
    from yggdrasil.polars.cast import cast_polars_dataframe
    return cast_polars_dataframe(df, make_options(schema, **kw))


def cast_lf(lf: pl.LazyFrame, schema: pa.Schema, **kw) -> pl.DataFrame:
    from yggdrasil.polars.cast import cast_polars_lazyframe
    casted = cast_polars_lazyframe(lf, make_options(schema, **kw))

    if isinstance(lf, pl.LazyFrame):
        return casted.collect()
    return casted


# Run each DataFrame test against both the eager and lazy path
@pytest.fixture(params=["df", "lf"], ids=["DataFrame", "LazyFrame"])
def cast_fn(request):
    if request.param == "df":
        return cast_df
    return cast_lf


# ---------------------------------------------------------------------------
# 1. Primitive scalar types
# ---------------------------------------------------------------------------

class TestPrimitiveScalars:
    """Round-trip and cross-cast for every primitive Arrow ↔ Polars type."""

    @pytest.mark.parametrize("src_dtype,tgt_arrow,values", [
        # integers
        (pl.Int8,   pa.int8(),   [-1, 0, 1]),
        (pl.Int16,  pa.int16(),  [-1000, 0, 1000]),
        (pl.Int32,  pa.int32(),  [-1, 0, 1]),
        (pl.Int64,  pa.int64(),  [-1, 0, 1]),
        (pl.UInt8,  pa.uint8(),  [0, 1, 255]),
        (pl.UInt16, pa.uint16(), [0, 1, 65535]),
        (pl.UInt32, pa.uint32(), [0, 1, 2**32 - 1]),
        (pl.UInt64, pa.uint64(), [0, 1, 2**63]),
        # floats
        (pl.Float32, pa.float32(), [0.0, 1.5, -1.5]),
        (pl.Float64, pa.float64(), [0.0, 1.5, -1.5]),
        # bool
        (pl.Boolean, pa.bool_(), [True, False, True]),
        # utf8
        (pl.Utf8,   pa.string(),       ["a", "b", "c"]),
        (pl.Utf8, pa.string_view(), ["a", "b", "c"]),
        (pl.Utf8,   pa.large_string(), ["a", "b", "c"]),
        (pl.Utf8, pa.large_utf8(), ["a", "b", "c"]),
        # string
        (pl.String, pa.string(), ["a", "b", "c"]),
        (pl.String, pa.string_view(), ["a", "b", "c"]),
        (pl.String, pa.large_string(), ["a", "b", "cf"]),
        (pl.String, pa.large_utf8(), ["a", "b", "c"]),
        # binary
        (pl.Binary, pa.binary(),       [b"a", b"b", b"c"]),
        (pl.Binary, pa.binary(4), [b"abcd", b"defg", b"hijk"]),
        (pl.Binary, pa.binary_view(), [b"a", b"b", b"c"]),
        (pl.Binary, pa.large_binary(), [b"a", b"b", b"c"]),
    ])
    def test_round_trip(self, cast_fn, src_dtype, tgt_arrow, values):
        df = pl.DataFrame({"x": pl.Series("x", values, dtype=src_dtype)})
        schema = pa.schema([pa.field("x", tgt_arrow)])
        result = cast_fn(df, schema)
        assert result.shape == (3, 1)
        assert result["x"].null_count() == 0

    def test_int_to_float(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [1, 2, 3], dtype=pl.Int32)})
        schema = pa.schema([pa.field("x", pa.float64())])
        result = cast_fn(df, schema)
        assert result["x"].dtype == pl.Float64
        assert result["x"].to_list() == [1.0, 2.0, 3.0]

    def test_float_to_int_safe(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [1.0, 2.0, 3.0], dtype=pl.Float64)})
        schema = pa.schema([pa.field("x", pa.int64())])
        result = cast_fn(df, schema)
        assert result["x"].dtype == pl.Int64

    def test_bool_to_int(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [True, False, True], dtype=pl.Boolean)})
        schema = pa.schema([pa.field("x", pa.int8())])
        result = cast_fn(df, schema)
        assert result["x"].to_list() == [1, 0, 1]


# ---------------------------------------------------------------------------
# 2. Temporal types
# ---------------------------------------------------------------------------

class TestTemporalTypes:

    def test_date_round_trip(self, cast_fn):
        dates = [datetime.date(2024, 1, 1), datetime.date(2024, 6, 15)]
        df = pl.DataFrame({"d": pl.Series("d", dates, dtype=pl.Date)})
        schema = pa.schema([pa.field("d", pa.date32())])
        result = cast_fn(df, schema)
        assert result["d"].dtype == pl.Date
        assert result["d"].to_list() == dates

    def test_datetime_us_naive(self, cast_fn):
        dts = [datetime.datetime(2024, 1, 1, 12, 0, 0)]
        df = pl.DataFrame({"ts": pl.Series("ts", dts, dtype=pl.Datetime("us"))})
        schema = pa.schema([pa.field("ts", pa.timestamp("us"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("us")

    def test_datetime_ns_naive(self, cast_fn):
        dts = [datetime.datetime(2024, 1, 1, 12, 0, 0)]
        df = pl.DataFrame({"ts": pl.Series("ts", dts, dtype=pl.Datetime("ns"))})
        schema = pa.schema([pa.field("ts", pa.timestamp("ns"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("ns")

    def test_datetime_ms_naive(self, cast_fn):
        dts = [datetime.datetime(2024, 1, 1, 12, 0, 0)]
        df = pl.DataFrame({"ts": pl.Series("ts", dts, dtype=pl.Datetime("ms"))})
        schema = pa.schema([pa.field("ts", pa.timestamp("ms"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("ms")

    def test_datetime_tz_aware(self, cast_fn):
        dts = [datetime.datetime(2024, 1, 1, 12, 0, 0)]
        df = pl.DataFrame({"ts": pl.Series("ts", dts, dtype=pl.Datetime("us", "UTC"))})
        schema = pa.schema([pa.field("ts", pa.timestamp("us", tz="UTC"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("us", "UTC")

    def test_datetime_tz_conversion(self, cast_fn):
        dts = [datetime.datetime(2024, 1, 1, 12, 0, 0)]
        df = pl.DataFrame({"ts": pl.Series("ts", dts, dtype=pl.Datetime("us", "UTC"))})
        schema = pa.schema([pa.field("ts", pa.timestamp("us", tz="Europe/Paris"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("us", "Europe/Paris")

    def test_string_to_datetime(self, cast_fn):
        df = pl.DataFrame({"ts": ["2024-01-01T12:00:00", "2024-06-15T08:30:00"]})
        schema = pa.schema([pa.field("ts", pa.timestamp("us"))])
        result = cast_fn(df, schema, safe=False)
        assert result["ts"].dtype == pl.Datetime("us")
        assert result["ts"].null_count() == 0

    def test_string_to_date(self, cast_fn):
        df = pl.DataFrame({"d": ["2024-01-01", "2024-06-15"]})
        schema = pa.schema([pa.field("d", pa.date32())])
        result = cast_fn(df, schema, safe=False)
        assert result["d"].dtype == pl.Date

    def test_time_round_trip(self, cast_fn):
        times = [datetime.time(12, 0, 0), datetime.time(8, 30, 0)]
        df = pl.DataFrame({"t": pl.Series("t", times, dtype=pl.Time)})
        schema = pa.schema([pa.field("t", pa.time64("us"))])
        result = cast_fn(df, schema)
        assert result["t"].dtype == pl.Time

    def test_duration_us(self, cast_fn):
        durs = [datetime.timedelta(seconds=3600), datetime.timedelta(days=1)]
        df = pl.DataFrame({"dur": pl.Series("dur", durs, dtype=pl.Duration("us"))})
        schema = pa.schema([pa.field("dur", pa.duration("us"))])
        result = cast_fn(df, schema)
        assert result["dur"].dtype == pl.Duration("us")

    def test_duration_ms(self, cast_fn):
        durs = [datetime.timedelta(milliseconds=500)]
        df = pl.DataFrame({"dur": pl.Series("dur", durs, dtype=pl.Duration("ms"))})
        schema = pa.schema([pa.field("dur", pa.duration("ms"))])
        result = cast_fn(df, schema)
        assert result["dur"].dtype == pl.Duration("ms")

    def test_int_epoch_to_datetime(self, cast_fn):
        # Integer epoch microseconds → Datetime
        df = pl.DataFrame({"ts": pl.Series("ts", [0, 1_000_000], dtype=pl.Int64)})
        schema = pa.schema([pa.field("ts", pa.timestamp("us"))])
        result = cast_fn(df, schema)
        assert result["ts"].dtype == pl.Datetime("us")

    def test_date_to_datetime(self, cast_fn):
        dates = [datetime.date(2024, 1, 1)]
        df = pl.DataFrame({"d": pl.Series("d", dates, dtype=pl.Date)})
        schema = pa.schema([pa.field("d", pa.timestamp("us"))])
        result = cast_fn(df, schema)
        assert result["d"].dtype == pl.Datetime("us")


# ---------------------------------------------------------------------------
# 3. Decimal
# ---------------------------------------------------------------------------

class TestDecimal:

    def test_decimal_round_trip(self, cast_fn):
        values = [Decimal("1.23"), Decimal("4.56"), Decimal("7.89")]
        df = pl.DataFrame({"v": pl.Series("v", values, dtype=pl.Decimal(precision=10, scale=2))})
        schema = pa.schema([pa.field("v", pa.decimal128(10, 2))])
        result = cast_fn(df, schema)
        assert result["v"].dtype == pl.Decimal(precision=10, scale=2)

    def test_float_to_decimal(self, cast_fn):
        df = pl.DataFrame({"v": pl.Series("v", [1.23, 4.56], dtype=pl.Float64)})
        schema = pa.schema([pa.field("v", pa.decimal128(10, 2))])
        result = cast_fn(df, schema, safe=False)
        assert result.shape == (2, 1)


# ---------------------------------------------------------------------------
# 4. Categorical / Enum
# ---------------------------------------------------------------------------

class TestCategorical:

    def test_string_to_categorical(self, cast_fn):
        df = pl.DataFrame({"cat": ["a", "b", "a", "c"]})
        schema = pa.schema([pa.field("cat", pa.dictionary(pa.int32(), pa.string()))])
        result = cast_fn(df, schema)
        assert result["cat"].dtype in (pl.Categorical, pl.Utf8)  # Polars may keep as Utf8

    def test_categorical_round_trip(self, cast_fn):
        df = pl.DataFrame({"cat": pl.Series("cat", ["x", "y", "x"], dtype=pl.Categorical)})
        schema = pa.schema([pa.field("cat", pa.dictionary(pa.int32(), pa.string()))])
        result = cast_fn(df, schema)
        assert result.shape == (3, 1)


# ---------------------------------------------------------------------------
# 5. List types
# ---------------------------------------------------------------------------

class TestListTypes:

    def test_list_int_round_trip(self, cast_fn):
        df = pl.DataFrame({"lst": pl.Series("lst", [[1, 2], [3, 4, 5]], dtype=pl.List(pl.Int32))})
        schema = pa.schema([pa.field("lst", pa.list_(pa.int32()))])
        result = cast_fn(df, schema)
        assert result["lst"].dtype == pl.List(pl.Int32)
        assert result["lst"].to_list() == [[1, 2], [3, 4, 5]]

    def test_list_int_upcast(self, cast_fn):
        df = pl.DataFrame({"lst": pl.Series("lst", [[1, 2], [3]], dtype=pl.List(pl.Int32))})
        schema = pa.schema([pa.field("lst", pa.list_(pa.int64()))])
        result = cast_fn(df, schema)
        assert result["lst"].dtype == pl.List(pl.Int64)

    def test_list_string(self, cast_fn):
        df = pl.DataFrame({"lst": pl.Series("lst", [["a", "b"], ["c"]], dtype=pl.List(pl.Utf8))})
        schema = pa.schema([pa.field("lst", pa.list_(pa.string()))])
        result = cast_fn(df, schema)
        assert result["lst"].dtype == pl.List(pl.Utf8)

    def test_large_list(self, cast_fn):
        df = pl.DataFrame({"lst": pl.Series("lst", [[1, 2], [3]], dtype=pl.List(pl.Int32))})
        schema = pa.schema([pa.field("lst", pa.large_list(pa.int64()))])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)

    def test_list_of_structs(self, cast_fn):
        inner_schema = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"a": 1, "b": "x"}],
                [{"a": 2, "b": "y"}, {"a": 3, "b": "z"}],
            ])
        })
        schema = pa.schema([pa.field("lst", pa.list_(inner_schema))])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)

    def test_fixed_size_list(self, cast_fn):
        df = pl.DataFrame({
            "arr": pl.Series("arr", [[1, 2, 3], [4, 5, 6]], dtype=pl.Array(pl.Int32, 3))
        })
        schema = pa.schema([pa.field("arr", pa.list_(pa.int64()))])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)

    def test_scalar_to_list(self, cast_fn):
        # A scalar int column → list of int (wrap each value)
        df = pl.DataFrame({"x": pl.Series("x", [1, 2, 3], dtype=pl.Int32)})
        schema = pa.schema([pa.field("x", pa.list_(pa.int64()))])
        result = cast_fn(df, schema)
        assert result["x"].to_list() == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# 6. Struct types
# ---------------------------------------------------------------------------

class TestStructTypes:

    def test_flat_struct_round_trip(self, cast_fn):
        df = pl.DataFrame({
            "s": pl.Series("s", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        })
        schema = pa.schema([
            pa.field("s", pa.struct([
                pa.field("a", pa.int32()),
                pa.field("b", pa.string()),
            ]))
        ])
        result = cast_fn(df, schema)
        assert result["s"].dtype == pl.Struct([pl.Field("a", pl.Int32), pl.Field("b", pl.Utf8)])

    def test_struct_field_type_cast(self, cast_fn):
        df = pl.DataFrame({
            "s": pl.Series("s", [{"a": 1, "b": 2}])
        })
        schema = pa.schema([
            pa.field("s", pa.struct([
                pa.field("a", pa.int64()),
                pa.field("b", pa.float64()),
            ]))
        ])
        result = cast_fn(df, schema)
        inner = result["s"].struct.unnest()
        assert inner["a"].dtype == pl.Int64
        assert inner["b"].dtype == pl.Float64

    def test_nested_struct(self, cast_fn):
        df = pl.DataFrame({
            "s": pl.Series("s", [{"outer": {"inner": 42}}])
        })
        schema = pa.schema([
            pa.field("s", pa.struct([
                pa.field("outer", pa.struct([
                    pa.field("inner", pa.int64())
                ]))
            ]))
        ])
        result = cast_fn(df, schema)
        assert result.shape == (1, 1)

    def test_struct_missing_field_filled(self, cast_fn):
        df = pl.DataFrame({
            "s": pl.Series("s", [{"a": 1}])
        })
        schema = pa.schema([
            pa.field("s", pa.struct([
                pa.field("a", pa.int32()),
                pa.field("b", pa.string(), nullable=True),
            ]))
        ])
        result = cast_fn(df, schema, add_missing_columns=True)
        inner = result["s"].struct.unnest()
        assert "b" in inner.columns

    def test_json_string_to_struct(self, cast_fn):
        df = pl.DataFrame({"s": ['{"a": 1, "b": "hello"}', '{"a": 2, "b": "world"}']})
        schema = pa.schema([
            pa.field("s", pa.struct([
                pa.field("a", pa.int32()),
                pa.field("b", pa.string()),
            ]))
        ])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)


# ---------------------------------------------------------------------------
# 7. Column matching strategy
# ---------------------------------------------------------------------------

class TestColumnMatching:

    def test_exact_match(self, cast_fn):
        df = pl.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        schema = pa.schema([pa.field("foo", pa.int64()), pa.field("bar", pa.int64())])
        result = cast_fn(df, schema)
        assert result.columns == ["foo", "bar"]

    def test_case_insensitive_match(self, cast_fn):
        df = pl.DataFrame({"FOO": [1, 2], "BAR": [3, 4]})
        schema = pa.schema([pa.field("foo", pa.int64()), pa.field("bar", pa.int64())])
        result = cast_fn(df, schema, strict_match_names=False)
        assert result.columns == ["foo", "bar"]
        assert result["foo"].to_list() == [1, 2]

    def test_positional_fallback(self, cast_fn):
        df = pl.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.string())])
        result = cast_fn(df, schema, strict_match_names=False)
        assert result.columns == ["x", "y"]
        assert result["x"].to_list() == [1, 2]

    def test_strict_match_raises_on_mismatch(self, cast_fn):
        df = pl.DataFrame({"FOO": [1, 2]})
        schema = pa.schema([pa.field("foo", pa.int64())])
        with pytest.raises(Exception):
            cast_fn(df, schema, strict_match_names=True)

    def test_column_reorder(self, cast_fn):
        df = pl.DataFrame({"b": [1, 2], "a": [3, 4]})
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.int64())])
        result = cast_fn(df, schema)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [3, 4]


# ---------------------------------------------------------------------------
# 8. Missing and extra columns
# ---------------------------------------------------------------------------

class TestMissingExtraColumns:

    def test_missing_column_raises_by_default(self, cast_fn):
        df = pl.DataFrame({"a": [1, 2]})
        schema = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
        with pytest.raises(Exception):
            cast_fn(df, schema)

    def test_missing_column_filled_nullable(self, cast_fn):
        df = pl.DataFrame({"a": [1, 2]})
        schema = pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.string(), nullable=True),
        ])
        result = cast_fn(df, schema, add_missing_columns=True)
        assert "b" in result.columns
        assert result["b"].null_count() == 2

    def test_missing_column_filled_non_nullable(self, cast_fn):
        df = pl.DataFrame({"a": [1, 2]})
        schema = pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.int32(), nullable=False),
        ])
        result = cast_fn(df, schema, add_missing_columns=True)
        assert result["b"].null_count() == 0
        assert result["b"].to_list() == [0, 0]

    def test_extra_column_dropped_by_default(self, cast_fn):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        schema = pa.schema([pa.field("a", pa.int64())])
        result = cast_fn(df, schema)
        assert result.columns == ["a"]

    def test_extra_column_preserved(self, cast_fn):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        schema = pa.schema([pa.field("a", pa.int64())])
        result = cast_fn(df, schema, allow_add_columns=True)
        assert "b" in result.columns
        assert "c" in result.columns


# ---------------------------------------------------------------------------
# 9. Nullability fill
# ---------------------------------------------------------------------------

class TestNullabilityFill:

    def test_nullable_column_with_nulls_passes(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [1, None, 3], dtype=pl.Int32)})
        schema = pa.schema([pa.field("x", pa.int32(), nullable=True)])
        result = cast_fn(df, schema)
        assert result["x"].null_count() == 1

    def test_non_nullable_column_fills_nulls(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [1, None, 3], dtype=pl.Int32)})
        schema = pa.schema([pa.field("x", pa.int32(), nullable=False)])
        result = cast_fn(df, schema)
        assert result["x"].null_count() == 0
        assert result["x"][1] == 0  # default int fill

    def test_non_nullable_string_fills_nulls(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", ["a", None, "c"], dtype=pl.Utf8)})
        schema = pa.schema([pa.field("x", pa.string(), nullable=False)])
        result = cast_fn(df, schema)
        assert result["x"].null_count() == 0


# ---------------------------------------------------------------------------
# 10. Multi-column schemas (integration)
# ---------------------------------------------------------------------------

class TestMultiColumnIntegration:

    def test_mixed_types_dataframe(self, cast_fn):
        df = pl.DataFrame({
            "id":    pl.Series("id",    [1, 2, 3],                    dtype=pl.Int32),
            "name":  pl.Series("name",  ["a", "b", "c"],              dtype=pl.Utf8),
            "score": pl.Series("score", [1.1, 2.2, 3.3],             dtype=pl.Float32),
            "flag":  pl.Series("flag",  [True, False, True],          dtype=pl.Boolean),
            "dt":    pl.Series("dt",    [datetime.date(2024, 1, i+1) for i in range(3)], dtype=pl.Date),
        })
        schema = pa.schema([
            pa.field("id",    pa.int64()),
            pa.field("name",  pa.large_string()),
            pa.field("score", pa.float64()),
            pa.field("flag",  pa.bool_()),
            pa.field("dt",    pa.date32()),
        ])
        result = cast_fn(df, schema)
        assert result.shape == (3, 5)
        assert result["id"].dtype    == pl.Int64
        assert result["score"].dtype == pl.Float64

    def test_empty_dataframe(self, cast_fn):
        df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int32)})
        schema = pa.schema([pa.field("a", pa.int64())])
        result = cast_fn(df, schema)
        assert result.shape == (0, 1)
        assert result["a"].dtype == pl.Int64

    def test_single_row(self, cast_fn):
        df = pl.DataFrame({"x": [42]})
        schema = pa.schema([pa.field("x", pa.float32())])
        result = cast_fn(df, schema)
        assert result["x"].to_list() == [42.0]

    def test_no_target_schema_passthrough(self):
        """When no target schema is set, cast functions return the frame unchanged."""
        from yggdrasil.polars.cast import cast_polars_dataframe, cast_polars_lazyframe
        from yggdrasil.data.cast import CastOptions

        df = pl.DataFrame({"x": [1, 2, 3]})
        opts = CastOptions()  # no target_field

        assert cast_polars_dataframe(df, opts).equals(df)
        assert cast_polars_lazyframe(df.lazy(), opts).collect().equals(df)


# ---------------------------------------------------------------------------
# 11. LazyFrame-specific: stays lazy until collect
# ---------------------------------------------------------------------------

class TestLazyFrame:

    def test_returns_lazyframe_before_collect(self):
        from yggdrasil.polars.cast import cast_polars_lazyframe

        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        schema = pa.schema([pa.field("x", pa.int64())])
        opts = make_options(schema)
        result = cast_polars_lazyframe(lf, opts)
        assert isinstance(result, pl.LazyFrame)

    def test_lazyframe_plan_does_not_collect(self):
        """Verify the lazy plan can be inspected without triggering execution."""
        from yggdrasil.polars.cast import cast_polars_lazyframe

        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        schema = pa.schema([pa.field("x", pa.int64())])
        result = cast_polars_lazyframe(lf, make_options(schema))
        # .explain() must not raise
        plan = result.explain()
        assert isinstance(plan, str)

    def test_lazyframe_matches_dataframe_output(self):
        from yggdrasil.polars.cast import cast_polars_dataframe, cast_polars_lazyframe

        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.1, 2.2, 3.3],
        })
        schema = pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.large_string()),
            pa.field("c", pa.float32()),
        ])
        opts = make_options(schema)
        eager  = cast_polars_dataframe(df, opts)
        lazy   = cast_polars_lazyframe(df.lazy(), opts).collect()
        assert eager.equals(lazy)

    def test_lazyframe_missing_column_filled(self):
        from yggdrasil.polars.cast import cast_polars_lazyframe

        lf = pl.DataFrame({"a": [1, 2]}).lazy()
        schema = pa.schema([
            pa.field("a", pa.int64()),
            pa.field("b", pa.string(), nullable=True),
        ])
        result = cast_polars_lazyframe(lf, make_options(schema, add_missing_columns=True)).collect()
        assert "b" in result.columns
        assert result["b"].null_count() == 2

    def test_lazyframe_extra_column_preserved(self):
        from yggdrasil.polars.cast import cast_polars_lazyframe

        lf = pl.DataFrame({"a": [1], "extra": [99]}).lazy()
        schema = pa.schema([pa.field("a", pa.int64())])
        result = cast_polars_lazyframe(
            lf, make_options(schema, allow_add_columns=True)
        ).collect()
        assert "extra" in result.columns


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_all_nulls_nullable(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [None, None, None], dtype=pl.Int32)})
        schema = pa.schema([pa.field("x", pa.int64(), nullable=True)])
        result = cast_fn(df, schema)
        assert result["x"].null_count() == 3

    def test_large_integer_values(self, cast_fn):
        df = pl.DataFrame({"x": pl.Series("x", [2**62, 2**63 - 1], dtype=pl.UInt64)})
        schema = pa.schema([pa.field("x", pa.uint64())])
        result = cast_fn(df, schema)
        assert result["x"].null_count() == 0

    def test_unicode_strings(self, cast_fn):
        df = pl.DataFrame({"s": ["こんにちは", "世界", "🎉"]})
        schema = pa.schema([pa.field("s", pa.string())])
        result = cast_fn(df, schema)
        assert result["s"].to_list() == ["こんにちは", "世界", "🎉"]

    def test_binary_data(self, cast_fn):
        df = pl.DataFrame({"b": pl.Series("b", [b"\x00\xff", b"\x01\x02"], dtype=pl.Binary)})
        schema = pa.schema([pa.field("b", pa.binary())])
        result = cast_fn(df, schema)
        assert result["b"].to_list() == [b"\x00\xff", b"\x01\x02"]

    def test_deeply_nested_list(self, cast_fn):
        # List[List[Int32]] → List[List[Int64]]
        df = pl.DataFrame({
            "x": pl.Series("x", [[[1, 2], [3]], [[4, 5, 6]]], dtype=pl.List(pl.List(pl.Int32)))
        })
        schema = pa.schema([pa.field("x", pa.list_(pa.list_(pa.int64())))])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)

    def test_string_to_bool(self, cast_fn):
        # Safe cast: string → bool via cast() which Polars handles
        df = pl.DataFrame({"x": ["true", "false", "true"]})
        schema = pa.schema([pa.field("x", pa.bool_())])
        # safe=False since string→bool isn't guaranteed strict
        result = cast_fn(df, schema, safe=False)
        assert result.shape == (3, 1)

class TestListOfStructs:

    def test_list_of_flat_structs_round_trip(self, cast_fn):
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"a": 1, "b": "x"}],
                [{"a": 2, "b": "y"}, {"a": 3, "b": "z"}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("a", pa.int32()),
                pa.field("b", pa.string()),
            ])))
        ])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)
        rows = result["lst"].to_list()
        assert rows[0] == [{"a": 1, "b": "x"}]
        assert rows[1] == [{"a": 2, "b": "y"}, {"a": 3, "b": "z"}]

    def test_list_of_structs_field_type_upcast(self, cast_fn):
        """Inner struct field int32 → int64."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"a": 1, "b": 10}],
                [{"a": 2, "b": 20}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("a", pa.int64()),
                pa.field("b", pa.int64()),
            ])))
        ])
        result = cast_fn(df, schema)
        inner_dtype = result["lst"].dtype.inner
        assert inner_dtype == pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Int64)])

    def test_list_of_structs_missing_field_filled(self, cast_fn):
        """Target struct has an extra nullable field absent from source."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"a": 1}],
                [{"a": 2}, {"a": 3}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("a", pa.int32()),
                pa.field("b", pa.string(), nullable=True),
            ])))
        ])
        result = cast_fn(df, schema, add_missing_columns=True)
        rows = result["lst"].to_list()
        assert all("b" in row for sublist in rows for row in sublist)
        assert all(row["b"] is None for sublist in rows for row in sublist)

    def test_list_of_structs_with_temporal_field(self, cast_fn):
        """Struct inside list contains a date field cast from string."""
        import datetime
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"name": "foo", "dt": "2024-01-01"}],
                [{"name": "bar", "dt": "2024-06-15"}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("name", pa.string()),
                pa.field("dt", pa.date32()),
            ])))
        ])
        result = cast_fn(df, schema, safe=False)
        rows = result["lst"].to_list()
        assert rows[0][0]["dt"] == datetime.date(2024, 1, 1)
        assert rows[1][0]["dt"] == datetime.date(2024, 6, 15)

    def test_list_of_nested_structs(self, cast_fn):
        """Struct inside list contains another struct."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"outer": {"inner": 1}}],
                [{"outer": {"inner": 2}}, {"outer": {"inner": 3}}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("outer", pa.struct([
                    pa.field("inner", pa.int64()),
                ]))
            ])))
        ])
        result = cast_fn(df, schema)
        rows = result["lst"].to_list()
        assert rows[0][0]["outer"]["inner"] == 1
        assert rows[1][1]["outer"]["inner"] == 3

    def test_large_list_of_structs(self, cast_fn):
        """large_list wrapper around a struct."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"x": 1}],
                [{"x": 2}, {"x": 3}],
            ])
        })
        schema = pa.schema([
            pa.field("lst", pa.large_list(pa.struct([
                pa.field("x", pa.int64()),
            ])))
        ])
        result = cast_fn(df, schema)
        assert result.shape == (2, 1)
        rows = result["lst"].to_list()
        assert rows[1] == [{"x": 2}, {"x": 3}]

    def test_fixed_size_list_of_structs(self, cast_fn):
        """Fixed-size array of structs with inner type cast."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"v": 1}, {"v": 2}],
                [{"v": 3}, {"v": 4}],
            ], dtype=pl.Array(pl.Struct([pl.Field("v", pl.Int32)]), 2))
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("v", pa.int64()),
            ])))
        ])
        result = cast_fn(df, schema)
        rows = result["lst"].to_list()
        assert rows[0] == [{"v": 1}, {"v": 2}]
        assert rows[1] == [{"v": 3}, {"v": 4}]

    def test_empty_lists_of_structs(self, cast_fn):
        """Empty inner lists should remain empty after cast."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [[], [{"a": 1}]], dtype=pl.List(
                pl.Struct([pl.Field("a", pl.Int32)])
            ))
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("a", pa.int64()),
            ])))
        ])
        result = cast_fn(df, schema)
        rows = result["lst"].to_list()
        assert rows[0] == []
        assert rows[1] == [{"a": 1}]

    def test_list_of_structs_null_entries(self, cast_fn):
        """Null rows in the outer list survive the cast."""
        df = pl.DataFrame({
            "lst": pl.Series("lst", [
                [{"a": 1}],
                None,
                [{"a": 3}],
            ], dtype=pl.List(pl.Struct([pl.Field("a", pl.Int32)])))
        })
        schema = pa.schema([
            pa.field("lst", pa.list_(pa.struct([
                pa.field("a", pa.int64()),
            ])), nullable=True)
        ])
        result = cast_fn(df, schema)
        rows = result["lst"].to_list()
        assert rows[0] == [{"a": 1}]
        assert rows[1] is None
        assert rows[2] == [{"a": 3}]