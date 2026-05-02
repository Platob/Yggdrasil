"""Integration tests for ``yggdrasil.data.types.primitive.temporal``.

We exercise the real :class:`TemporalType` subclasses through
:class:`CastOptions` and :class:`Field`, not the engine dispatchers in
isolation — that way these tests double as acceptance tests for the
whole cast pipeline.

Layout:

* **arrow** — ``_cast_arrow_array`` via CastOptions, including the
  ISO / locale-format catalogue for CSV-shaped strings.
* **polars** — Series and Expr paths.
* **pandas** — the Arrow bridge.
* **spark** — type-system mappings only (no SparkSession spin-up).
* **scalar** — ``_convert_pyobj`` per subclass.
* **merge** — unit and timezone reconciliation.
* **serde** — to_dict / from_dict / autotag / arrow round-trip.
* **dispatchers** — direct ``arrow_cast`` / ``cast_polars_array_to_temporal``
  / ``spark_cast`` calls for the cases that are awkward to reach
  through CastOptions.
"""
from __future__ import annotations

import datetime as dt
import decimal

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.primitive.temporal import (
    DateType,
    DurationType,
    TemporalType,
    TimeType,
    TimestampType,
    arrow_cast,
    cast_polars_array_to_temporal,
    spark_cast,
)


# ---------------------------------------------------------------------------
# Optional-dependency gates
# ---------------------------------------------------------------------------

pl = pytest.importorskip("polars", reason="polars not installed")

try:
    import pandas as pd  # noqa: F401

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False

try:
    import pyspark.sql  # noqa: F401
    import pyspark.sql.types as pst

    HAS_SPARK = True
except ImportError:  # pragma: no cover
    HAS_SPARK = False

pandas_only = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
spark_only = pytest.mark.skipif(not HAS_SPARK, reason="pyspark not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _options(target: TemporalType, *, safe: bool = False) -> CastOptions:
    """Build :class:`CastOptions` with *target* as the target field.

    Source binding is deliberately deferred to call time so we exercise
    the same ``check_source`` / ``need_cast`` path production casts use.
    """
    return CastOptions(target_field=Field("col", target), safe=safe)


def _cast_arrow(
    array: pa.Array, target: TemporalType, *, safe: bool = False
) -> pa.Array:
    return target._cast_arrow_array(array, _options(target, safe=safe))


def _cast_polars(
    series: "pl.Series", target: TemporalType, *, safe: bool = False
) -> "pl.Series":
    return target._cast_polars_series(series, _options(target, safe=safe))


def _polars_source_field(col: str, df: "pl.DataFrame") -> Field:
    """Build a :class:`Field` from a polars column dtype.

    The Expr-path dispatcher needs ``options.source_field.dtype.to_polars()``
    to re-derive the source dtype. Using ``DataType.from_polars_type``
    keeps the test agnostic to which primitive subclass owns the dtype.
    """
    return Field(col, DataType.from_polars_type(df.schema[col]))


# ===========================================================================
# Arrow
# ===========================================================================


class TestArrowTimestamp:

    @pytest.mark.parametrize(
        "raw,unit,tz,expected",
        [
            (["2024-01-15T10:30:00"], "us", None, [dt.datetime(2024, 1, 15, 10, 30)]),
            (
                ["2024-01-15T10:30:00.123456"],
                "us",
                None,
                [dt.datetime(2024, 1, 15, 10, 30, 0, 123456)],
            ),
            (
                ["2024-01-15T10:30:00.123"],
                "ms",
                None,
                [dt.datetime(2024, 1, 15, 10, 30, 0, 123000)],
            ),
            (["garbage"], "us", None, [None]),
            ([None], "us", None, [None]),
        ],
    )
    def test_string_to_naive_timestamp(self, raw, unit, tz, expected) -> None:
        arr = pa.array(raw, type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit=unit, tz=tz))

        assert out.type == pa.timestamp(unit, tz)
        assert out.to_pylist() == expected

    def test_naive_to_aware_keeps_wall_clock(self) -> None:
        arr = pa.array(["2024-01-15T10:30:00"], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us", tz="Europe/Paris"))

        assert out.type.tz == "Europe/Paris"
        v = out.to_pylist()[0]
        # 10:30 Paris (wall-clock reinterpret), not 10:30 UTC.
        assert v.hour == 10 and v.minute == 30
        assert str(v.tzinfo) in {"Europe/Paris", "UTC+01:00", "CET"}

    def test_unit_conversion_us_to_ms(self) -> None:
        arr = pa.array([1_700_000_000_000_000], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimestampType(unit="ms", tz=None))

        assert out.type == pa.timestamp("ms")
        assert out.to_pylist()[0] == dt.datetime.fromtimestamp(
            1_700_000_000, tz=dt.timezone.utc
        ).replace(tzinfo=None)

    def test_naive_existing_timestamp_to_aware_keeps_wall_clock(self) -> None:
        arr = pa.array([dt.datetime(2024, 6, 15, 14, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimestampType(unit="us", tz="America/New_York"))

        assert out.type.tz == "America/New_York"
        v = out.to_pylist()[0]
        assert v.hour == 14 and v.minute == 0

    def test_aware_to_aware_same_instant(self) -> None:
        arr = pa.array(
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            type=pa.timestamp("us", tz="UTC"),
        )
        out = _cast_arrow(arr, TimestampType(unit="us", tz="America/New_York"))

        # 14:00 UTC == 10:00 EDT in June.
        assert out.to_pylist()[0].hour == 10

    def test_aware_to_naive_drops_zone(self) -> None:
        arr = pa.array(
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            type=pa.timestamp("us", tz="UTC"),
        )
        out = _cast_arrow(arr, TimestampType(unit="us", tz=None))

        assert out.type.tz is None

    def test_second_precision_falls_back_to_pc_cast(self) -> None:
        arr = pa.array([1_700_000_000], type=pa.int64())
        out = _cast_arrow(arr, TimestampType(unit="s", tz=None))

        assert out.type == pa.timestamp("s")

    def test_date_source_widens_to_midnight_timestamp(self) -> None:
        arr = pa.array([dt.date(2024, 3, 1)], type=pa.date32())
        out = _cast_arrow(arr, TimestampType(unit="us", tz=None))

        assert out.type == pa.timestamp("us")
        assert out.to_pylist()[0] == dt.datetime(2024, 3, 1)

    def test_chunked_array_keeps_chunked_shape(self) -> None:
        chunked = pa.chunked_array(
            [
                pa.array(["2024-01-15T10:00:00"], type=pa.string()),
                pa.array(["2024-02-15T12:00:00"], type=pa.string()),
            ]
        )
        out = _cast_arrow(chunked, TimestampType(unit="us", tz=None))

        assert isinstance(out, pa.ChunkedArray)
        assert out.num_chunks == 2
        assert out.to_pylist() == [
            dt.datetime(2024, 1, 15, 10, 0),
            dt.datetime(2024, 2, 15, 12, 0),
        ]


class TestArrowDate:

    def test_iso_strings_parse(self) -> None:
        arr = pa.array(["2024-01-15", "2024-12-31"], type=pa.string())
        out = _cast_arrow(arr, DateType())

        assert out.type == pa.date32()
        assert out.to_pylist() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]

    def test_invalid_strings_become_null(self) -> None:
        arr = pa.array(["2024-01-15", "not-a-date"], type=pa.string())
        out = _cast_arrow(arr, DateType())

        assert out.to_pylist() == [dt.date(2024, 1, 15), None]

    def test_timestamp_source_truncates_to_date(self) -> None:
        arr = pa.array([dt.datetime(2024, 5, 1, 12, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, DateType())

        assert out.type == pa.date32()
        assert out.to_pylist() == [dt.date(2024, 5, 1)]

    def test_date64_target_widens(self) -> None:
        arr = pa.array([dt.date(2024, 1, 15)], type=pa.date32())
        out = _cast_arrow(arr, DateType(byte_size=8, unit="ms"))

        assert out.type == pa.date64()


class TestArrowTime:

    def test_iso_strings_parse(self) -> None:
        arr = pa.array(["10:30:45", "23:59:59.999999"], type=pa.string())
        out = _cast_arrow(arr, TimeType(unit="us"))

        assert pa.types.is_time(out.type)
        assert out.type.unit == "us"
        assert out.to_pylist() == [dt.time(10, 30, 45), dt.time(23, 59, 59, 999999)]

    def test_time32_ms_target(self) -> None:
        arr = pa.array(["10:30:45.123"], type=pa.string())
        out = _cast_arrow(arr, TimeType(byte_size=4, unit="ms"))

        assert out.type == pa.time32("ms")

    def test_timestamp_source_keeps_time_component(self) -> None:
        arr = pa.array([dt.datetime(2024, 1, 1, 14, 30, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimeType(unit="us"))

        assert out.to_pylist() == [dt.time(14, 30, 0)]


class TestArrowDuration:

    def test_int_source_to_us_duration(self) -> None:
        arr = pa.array([1_000_000, 2_500_000], type=pa.int64())
        out = _cast_arrow(arr, DurationType(unit="us"))

        assert out.type == pa.duration("us")
        assert out.to_pylist() == [
            dt.timedelta(seconds=1),
            dt.timedelta(seconds=2, microseconds=500_000),
        ]

    def test_ms_to_us_unit_conversion(self) -> None:
        arr = pa.array([1_000], type=pa.duration("ms"))
        out = _cast_arrow(arr, DurationType(unit="us"))

        assert out.type == pa.duration("us")
        assert out.to_pylist() == [dt.timedelta(seconds=1)]

    def test_second_precision_uses_pc_cast(self) -> None:
        arr = pa.array([60], type=pa.int64())
        out = _cast_arrow(arr, DurationType(unit="s"))

        assert out.type == pa.duration("s")


class TestArrowCsvFormatCatalogue:
    """ISO + 3 CSV/Excel shapes: locked to keep the catalogue stable."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-01-15", dt.date(2024, 1, 15)),
            ("15/01/2024", dt.date(2024, 1, 15)),
            ("01/15/2024", dt.date(2024, 1, 15)),
            ("2024/01/15", dt.date(2024, 1, 15)),
        ],
    )
    def test_date_shapes(self, raw: str, expected: dt.date) -> None:
        arr = pa.array([raw], type=pa.string())
        out = _cast_arrow(arr, DateType())

        assert out.to_pylist() == [expected]

    def test_dayfirst_wins_on_ambiguous_dates(self) -> None:
        """``01/02/2024`` is 1 Feb (day-first), not 2 Jan."""
        arr = pa.array(["01/02/2024"], type=pa.string())
        out = _cast_arrow(arr, DateType())

        assert out.to_pylist() == [dt.date(2024, 2, 1)]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-01-15T10:30:00", dt.datetime(2024, 1, 15, 10, 30)),
            ("2024-01-15 10:30:00", dt.datetime(2024, 1, 15, 10, 30)),
            ("15/01/2024 10:30:00", dt.datetime(2024, 1, 15, 10, 30)),
            ("01/15/2024 10:30:00", dt.datetime(2024, 1, 15, 10, 30)),
            ("2024/01/15 10:30:00", dt.datetime(2024, 1, 15, 10, 30)),
        ],
    )
    def test_datetime_shapes(self, raw: str, expected: dt.datetime) -> None:
        arr = pa.array([raw], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us"))

        assert out.to_pylist() == [expected]

    def test_mixed_format_column_via_coalesce(self) -> None:
        arr = pa.array(
            ["2024-01-15", "15/02/2024", "03/16/2024", "2024/04/17"],
            type=pa.string(),
        )
        out = _cast_arrow(arr, DateType())

        assert out.to_pylist() == [
            dt.date(2024, 1, 15),
            dt.date(2024, 2, 15),
            dt.date(2024, 3, 16),
            dt.date(2024, 4, 17),
        ]

    def test_all_garbage_yields_all_null_no_raise(self) -> None:
        arr = pa.array(["garbage", "more garbage"], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us"))

        assert out.to_pylist() == [None, None]


# ===========================================================================
# Polars
# ===========================================================================


class TestPolarsSeries:

    def test_string_to_naive_timestamp(self) -> None:
        s = pl.Series(
            "col",
            ["2024-01-15T10:30:00", "2024-06-22T08:15:45", "bad", None],
        )
        out = _cast_polars(s, TimestampType(unit="us"))

        assert out.dtype == pl.Datetime("us", None)
        vals = out.to_list()
        assert vals[0] == dt.datetime(2024, 1, 15, 10, 30)
        assert vals[2] is None
        assert vals[3] is None

    def test_string_to_date(self) -> None:
        s = pl.Series("col", ["2024-01-15", "2024-12-31"])
        out = _cast_polars(s, DateType())

        assert out.dtype == pl.Date
        assert out.to_list() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]

    def test_string_to_time(self) -> None:
        s = pl.Series("col", ["10:30:00", "23:59:59"])
        out = _cast_polars(s, TimeType(unit="us"))

        assert out.dtype == pl.Time
        assert out.to_list() == [dt.time(10, 30), dt.time(23, 59, 59)]

    def test_naive_to_aware_keeps_wall_clock(self) -> None:
        s = pl.Series(
            "col",
            [dt.datetime(2024, 6, 15, 14, 0)],
            dtype=pl.Datetime("us"),
        )
        out = _cast_polars(s, TimestampType(unit="us", tz="Europe/Paris"))

        assert isinstance(out.dtype, pl.Datetime)
        assert out.dtype.time_zone == "Europe/Paris"
        v = out.to_list()[0]
        assert v.hour == 14 and v.minute == 0

    def test_aware_to_aware_converts_zone(self) -> None:
        s = pl.Series(
            "col",
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            dtype=pl.Datetime("us", "UTC"),
        )
        out = _cast_polars(s, TimestampType(unit="us", tz="America/New_York"))

        assert out.dtype.time_zone == "America/New_York"
        # 14:00 UTC → 10:00 NY (EDT) in June.
        assert out.to_list()[0].hour == 10

    def test_second_precision_routes_through_arrow_bridge(self) -> None:
        s = pl.Series("col", [1_700_000_000], dtype=pl.Int64)
        out = _cast_polars(s, TimestampType(unit="s"))

        assert isinstance(out.dtype, pl.Datetime)

    def test_unit_widening(self) -> None:
        s = pl.Series("col", [dt.datetime(2024, 1, 1)], dtype=pl.Datetime("ms"))
        out = _cast_polars(s, TimestampType(unit="us"))

        assert out.dtype == pl.Datetime("us", None)

    def test_int_to_duration(self) -> None:
        s = pl.Series("col", [1_000_000, 2_000_000], dtype=pl.Int64)
        out = _cast_polars(s, DurationType(unit="us"))

        assert out.dtype == pl.Duration("us")

    def test_name_is_preserved(self) -> None:
        s = pl.Series("my_ts", ["2024-01-01T00:00:00"])
        out = _cast_polars(s, TimestampType(unit="us"))

        assert out.name == "my_ts"


class TestPolarsExpr:

    def test_expr_path_to_timestamp(self) -> None:
        df = pl.DataFrame(
            {"col": ["2024-01-15T10:30:00", "2024-06-22T08:15:45"]}
        )
        target = TimestampType(unit="us")
        options = CastOptions(
            source_field=_polars_source_field("col", df),
            target_field=Field("col", target),
            safe=False,
        )

        out = df.select(target._cast_polars_expr(pl.col("col"), options)).to_series()

        assert out.dtype == pl.Datetime("us", None)
        assert out.to_list()[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_expr_path_to_date(self) -> None:
        df = pl.DataFrame({"col": ["2024-01-15", "2024-12-31"]})
        target = DateType()
        options = CastOptions(
            source_field=_polars_source_field("col", df),
            target_field=Field("col", target),
            safe=False,
        )

        out = df.select(target._cast_polars_expr(pl.col("col"), options)).to_series()

        assert out.dtype == pl.Date
        assert out.to_list() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]


# ===========================================================================
# Pandas — Arrow bridge
# ===========================================================================


@pandas_only
class TestPandasBridge:

    def test_string_series_to_timestamp(self) -> None:
        import pandas as pd

        s = pd.Series(["2024-01-15T10:30:00", "2024-06-22T08:15:45"])
        target = TimestampType(unit="us")

        out = target._cast_pandas_series(s, _options(target))

        assert out.iloc[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_date_series_round_trip(self) -> None:
        import pandas as pd

        s = pd.Series(["2024-01-15", "2024-12-31"])
        target = DateType()

        out = target._cast_pandas_series(s, _options(target))

        assert list(out) == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]


# ===========================================================================
# Spark — type-system mappings
# ===========================================================================


@spark_only
class TestSparkTypes:
    """Type-system level only; SparkSession spin-up belongs to integration."""

    def test_aware_timestamp_maps_to_spark_timestamp(self) -> None:
        assert isinstance(
            TimestampType(unit="us", tz="UTC").to_spark(), pst.TimestampType
        )

    def test_naive_timestamp_maps_to_ntz_when_available(self) -> None:
        # Older Spark lacks TimestampNTZType — silently widens to TimestampType.
        expected = getattr(pst, "TimestampNTZType", pst.TimestampType)
        assert isinstance(TimestampType(unit="us", tz=None).to_spark(), expected)

    def test_date_maps_to_spark_date(self) -> None:
        assert isinstance(DateType().to_spark(), pst.DateType)

    def test_time_drops_to_string(self) -> None:
        # Spark has no time-of-day type; carry as string.
        assert isinstance(TimeType().to_spark(), pst.StringType)

    def test_duration_widens_to_long(self) -> None:
        assert isinstance(DurationType(unit="us").to_spark(), pst.LongType)

    def test_databricks_ddl_smoke(self) -> None:
        assert TimestampType(tz="UTC").to_databricks_ddl() == "TIMESTAMP"
        assert TimestampType(tz=None).to_databricks_ddl() == "TIMESTAMP_NTZ"
        assert DateType().to_databricks_ddl() == "DATE"
        assert TimeType().to_databricks_ddl() == "STRING"
        assert DurationType().to_databricks_ddl() == "BIGINT"

    def test_from_spark_round_trip_preserves_tz(self) -> None:
        original = TimestampType(unit="us", tz="UTC")
        round_tripped = TimestampType.from_spark_type(original.to_spark())
        assert round_tripped.tz == "UTC"


# ===========================================================================
# Scalar conversion (_convert_pyobj)
# ===========================================================================


class TestScalarDate:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2024-01-15", dt.date(2024, 1, 15)),
            (dt.date(2024, 3, 1), dt.date(2024, 3, 1)),
            (dt.datetime(2024, 3, 1, 14, 30), dt.date(2024, 3, 1)),
            (None, None),
            (0, dt.date(1970, 1, 1)),  # epoch days
            (1, dt.date(1970, 1, 2)),
        ],
    )
    def test_accepts_known_shapes(self, value, expected) -> None:
        assert DateType()._convert_pyobj(value) == expected

    def test_invalid_returns_none_in_best_effort_mode(self) -> None:
        assert DateType()._convert_pyobj("garbage") is None

    def test_invalid_safe_raises(self) -> None:
        with pytest.raises(ValueError):
            DateType()._convert_pyobj("garbage", safe=True)

    def test_empty_string_safe_raises(self) -> None:
        with pytest.raises(ValueError):
            DateType()._convert_pyobj("", safe=True)

    def test_empty_string_best_effort_returns_none(self) -> None:
        assert DateType()._convert_pyobj("") is None


class TestScalarTime:

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("10:30:45", dt.time(10, 30, 45)),
            ("23:59:59.123456", dt.time(23, 59, 59, 123456)),
            (dt.time(5, 0), dt.time(5, 0)),
            (dt.datetime(2024, 1, 1, 14, 30), dt.time(14, 30)),
            (None, None),
        ],
    )
    def test_accepts_known_shapes(self, value, expected) -> None:
        assert TimeType()._convert_pyobj(value) == expected

    def test_invalid_safe_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeType()._convert_pyobj("not-a-time", safe=True)


class TestScalarTimestamp:

    def test_iso_naive(self) -> None:
        assert TimestampType()._convert_pyobj(
            "2024-01-15T10:30:00"
        ) == dt.datetime(2024, 1, 15, 10, 30)

    def test_iso_with_z_suffix_normalized(self) -> None:
        out = TimestampType(tz="UTC")._convert_pyobj("2024-01-15T10:30:00Z")
        assert out == dt.datetime(2024, 1, 15, 10, 30, tzinfo=dt.timezone.utc)

    def test_naive_target_strips_tz_via_utc(self) -> None:
        aware = dt.datetime(2024, 1, 15, 10, 30, tzinfo=dt.timezone.utc)
        out = TimestampType(tz=None)._convert_pyobj(aware)

        assert out.tzinfo is None
        assert out == dt.datetime(2024, 1, 15, 10, 30)

    def test_aware_target_attaches_utc_to_naive(self) -> None:
        out = TimestampType(tz="UTC")._convert_pyobj(dt.datetime(2024, 1, 15, 10, 30))

        assert out.tzinfo is dt.timezone.utc

    def test_date_widens_to_midnight(self) -> None:
        assert TimestampType(tz=None)._convert_pyobj(
            dt.date(2024, 5, 1)
        ) == dt.datetime(2024, 5, 1)

    def test_numeric_treated_as_epoch_in_target_unit(self) -> None:
        out = TimestampType(unit="s", tz="UTC")._convert_pyobj(1_700_000_000)
        assert out == dt.datetime.fromtimestamp(1_700_000_000, tz=dt.timezone.utc)

    def test_invalid_safe_raises(self) -> None:
        with pytest.raises(ValueError):
            TimestampType()._convert_pyobj("nonsense", safe=True)


class TestScalarDuration:

    @pytest.mark.parametrize(
        "value,unit,expected",
        [
            (dt.timedelta(seconds=5), "us", dt.timedelta(seconds=5)),
            (1_000_000, "us", dt.timedelta(seconds=1)),
            (1000, "ms", dt.timedelta(seconds=1)),
            (1, "s", dt.timedelta(seconds=1)),
            ("1000000", "us", dt.timedelta(seconds=1)),
            (None, "us", None),
        ],
    )
    def test_numeric_and_string_inputs(self, value, unit, expected) -> None:
        assert DurationType(unit=unit)._convert_pyobj(value) == expected

    def test_decimal_input(self) -> None:
        out = DurationType(unit="us")._convert_pyobj(decimal.Decimal("500000"))
        assert out == dt.timedelta(milliseconds=500)

    def test_bool_treated_as_one_unit(self) -> None:
        assert DurationType(unit="us")._convert_pyobj(True) == dt.timedelta(microseconds=1)

    def test_garbage_string_best_effort_returns_none(self) -> None:
        assert DurationType()._convert_pyobj("not-a-number") is None

    def test_garbage_string_safe_raises(self) -> None:
        with pytest.raises(ValueError):
            DurationType()._convert_pyobj("not-a-number", safe=True)

    def test_iso_8601_duration_at_scalar_level(self) -> None:
        assert DurationType()._convert_pyobj("PT15M") == dt.timedelta(minutes=15)
        assert DurationType()._convert_pyobj("PT1H30M") == dt.timedelta(
            hours=1, minutes=30
        )
        assert DurationType()._convert_pyobj("-PT1H") == dt.timedelta(hours=-1)
        # Clock-style.
        assert DurationType()._convert_pyobj("01:30:00") == dt.timedelta(
            hours=1, minutes=30
        )


# ===========================================================================
# Merge
# ===========================================================================


class TestMerge:

    @pytest.mark.parametrize(
        "left_unit,right_unit,downcast,expected_unit",
        [
            # Default upcast — widest unit wins.
            ("s", "us", False, "us"),
            ("ms", "ns", False, "ns"),
            ("us", "us", False, "us"),
            # Downcast — narrowest wins.
            ("us", "ns", True, "us"),
            ("s", "ms", True, "s"),
        ],
    )
    def test_timestamp_unit_reconciliation(
        self,
        left_unit: str,
        right_unit: str,
        downcast: bool,
        expected_unit: str,
    ) -> None:
        merged = TimestampType(unit=left_unit)._merge_with_same_id(
            TimestampType(unit=right_unit), downcast=downcast
        )
        assert merged.unit == expected_unit

    def test_tz_unification_same(self) -> None:
        merged = TimestampType(tz="UTC")._merge_with_same_id(
            TimestampType(tz="UTC")
        )
        assert merged.tz == "UTC"

    def test_tz_unification_conflict_upcast_picks_first_non_none(self) -> None:
        merged = TimestampType(tz="UTC")._merge_with_same_id(
            TimestampType(tz="Europe/Paris")
        )
        assert merged.tz in {"UTC", "Europe/Paris"}

    def test_tz_unification_conflict_downcast_drops(self) -> None:
        merged = TimestampType(tz="UTC")._merge_with_same_id(
            TimestampType(tz="Europe/Paris"), downcast=True
        )
        assert merged.tz is None

    def test_cross_class_merge_raises(self) -> None:
        with pytest.raises(TypeError):
            DateType()._merge_with_same_id(TimestampType())


# ===========================================================================
# Serde
# ===========================================================================


class TestSerde:

    @pytest.mark.parametrize(
        "target",
        [
            # Round-trip identity needs explicit ``byte_size`` on both sides
            # because ``from_dict`` injects a default the constructor doesn't.
            DateType(byte_size=4),
            DateType(byte_size=8, unit="ms"),
            TimeType(byte_size=8, unit="us"),
            TimeType(byte_size=4, unit="ms"),
            TimestampType(byte_size=8, unit="us", tz="UTC"),
            TimestampType(byte_size=8, unit="ns", tz="Europe/Paris"),
            TimestampType(byte_size=8, unit="us", tz=None),
            DurationType(byte_size=8, unit="us"),
            DurationType(byte_size=8, unit="ns"),
        ],
    )
    def test_dict_round_trip(self, target: TemporalType) -> None:
        restored = type(target).from_dict(target.to_dict())
        assert restored == target

    def test_autotag_carries_unit_and_tz(self) -> None:
        tags = TimestampType(unit="ns", tz="Europe/Paris").autotag()

        assert tags[b"unit"] == b"ns"
        assert tags[b"tz"] == b"Europe/Paris"

    def test_autotag_omits_none_tz(self) -> None:
        assert b"tz" not in TimestampType(tz=None).autotag()

    @pytest.mark.parametrize(
        "target,expected_arrow",
        [
            (DateType(), pa.date32()),
            (DateType(byte_size=8, unit="ms"), pa.date64()),
            (TimeType(unit="us"), pa.time64("us")),
            (TimeType(unit="ms"), pa.time32("ms")),
            (TimestampType(unit="us", tz="UTC"), pa.timestamp("us", "UTC")),
            (TimestampType(unit="ns"), pa.timestamp("ns")),
            (DurationType(unit="us"), pa.duration("us")),
        ],
    )
    def test_to_arrow_matrix(
        self, target: TemporalType, expected_arrow: pa.DataType
    ) -> None:
        assert target.to_arrow() == expected_arrow

    @pytest.mark.parametrize(
        "target,arrow_dtype",
        [
            (DateType(), pa.date32()),
            (TimeType(unit="ns"), pa.time64("ns")),
            (TimestampType(unit="us", tz="UTC"), pa.timestamp("us", "UTC")),
            (DurationType(unit="ms"), pa.duration("ms")),
        ],
    )
    def test_arrow_round_trip(
        self, target: TemporalType, arrow_dtype: pa.DataType
    ) -> None:
        cls = type(target)
        assert cls.handles_arrow_type(arrow_dtype)

        round_tripped = cls.from_arrow_type(arrow_dtype)
        assert round_tripped.unit == target.unit
        if isinstance(target, TimestampType):
            assert round_tripped.tz == target.tz

    def test_polars_dispatch_handles(self) -> None:
        assert TimestampType.handles_polars_type(pl.Datetime("us", "UTC"))
        assert DateType.handles_polars_type(pl.Date)
        assert TimeType.handles_polars_type(pl.Time)
        assert DurationType.handles_polars_type(pl.Duration("us"))
        # Cross-class rejection.
        assert not DateType.handles_polars_type(pl.Datetime("us"))


# ===========================================================================
# Engine dispatchers (direct calls)
# ===========================================================================


class TestArrowDispatcher:

    def test_non_temporal_target_passes_through(self) -> None:
        arr = pa.array([1, 2, 3], type=pa.int64())
        out = arrow_cast(arr, pa.float64())

        assert out.type == pa.float64()

    def test_null_source_widens_to_temporal(self) -> None:
        arr = pa.nulls(3, type=pa.null())
        out = arrow_cast(arr, pa.timestamp("us"))

        assert out.type == pa.timestamp("us")
        assert out.to_pylist() == [None, None, None]

    def test_empty_chunked_array_round_trips(self) -> None:
        chunked = pa.chunked_array([], type=pa.string())
        out = arrow_cast(chunked, pa.timestamp("us"))

        assert isinstance(out, pa.ChunkedArray)
        assert len(out) == 0
        assert out.type == pa.timestamp("us")

    def test_safe_true_succeeds_on_valid_input(self) -> None:
        arr = pa.array(["2024-01-15T10:30:00"], type=pa.string())
        out = arrow_cast(arr, pa.timestamp("us"), safe=True)

        assert out.to_pylist() == [dt.datetime(2024, 1, 15, 10, 30)]


class TestPolarsDispatcher:

    def test_series_keeps_name(self) -> None:
        s = pl.Series("x", ["2024-01-01T00:00:00"])
        out = cast_polars_array_to_temporal(
            s,
            source=pl.String,
            target=pl.Datetime("us"),
            safe=False,
        )

        assert out.name == "x"

    def test_to_expr_returns_expression(self) -> None:
        s = pl.Series("x", ["2024-01-01T00:00:00"])
        out = cast_polars_array_to_temporal(
            s,
            source=pl.String,
            target=pl.Datetime("us"),
            safe=False,
            to_expr=True,
        )

        assert isinstance(out, pl.Expr)

    def test_unsupported_target_raises(self) -> None:
        s = pl.Series("x", [1, 2, 3])

        with pytest.raises(TypeError, match="Unsupported temporal target"):
            cast_polars_array_to_temporal(
                s,
                source=pl.Int64,
                target=pl.Float64,  # not a temporal target
                safe=False,
            )


class TestSparkDispatcher:

    def test_delegates_to_column_cast(self) -> None:
        class _FakeCol:
            def __init__(self) -> None:
                self.cast_arg = None

            def cast(self, target):
                self.cast_arg = target
                return self

        col = _FakeCol()
        target = "<fake-target>"
        out = spark_cast(col, target)

        assert out is col
        assert col.cast_arg is target

    def test_extra_kwargs_silently_ignored(self) -> None:
        class _FakeCol:
            def cast(self, target):
                return ("cast", target)

        # Signature accepts unit/tz/safe but discards them.
        result = spark_cast(
            _FakeCol(), "fake_target", safe=True, unit="ns", tz="Europe/Paris"
        )
        assert result == ("cast", "fake_target")
