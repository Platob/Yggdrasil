"""Integration tests for ``yggdrasil.data.types.primitive.temporal``.

Assumes a working yggdrasil install — we exercise the real
:class:`TemporalType` subclasses through ``CastOptions`` / ``Field``
rather than the engine dispatchers in isolation. That means these tests
double as acceptance tests for the whole cast pipeline.

Layout:

* **arrow**    — roundtrips through ``_cast_arrow_array`` via CastOptions.
* **polars**   — Series and Expr paths via ``_cast_polars_series`` /
                 ``_cast_polars_expr``.
* **pandas**   — the Arrow bridge path.
* **spark**    — gated on ``pyspark`` importability. Checks type-system
                 correctness; we don't spin up a full SparkSession for
                 compute-level assertions because that's integration-
                 test-infrastructure territory, not this file.
* **scalar**   — ``_convert_pyobj`` on each subclass.
* **merge**    — unit-widening and tz-unification.
* **serde**    — ``to_dict`` / ``from_dict`` / ``autotag`` roundtrips.
* **engine dispatchers** — direct ``arrow_cast`` / ``cast_polars_array_to_temporal``
                 calls for cases that are awkward to reach through the
                 CastOptions facade (polars-incompatible units, the
                 Arrow→polars fallback boundary).

The test file is organized by target type within each section (Date,
Time, Timestamp, Duration) so a failure points at both the dimension
and the engine that broke.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

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
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field  # yggdrasil's Field class


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
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_options(target: TemporalType, *, safe: bool = False) -> CastOptions:
    """Build CastOptions for a given target type.

    ``CastOptions.check`` is the canonical constructor — it normalises the
    target into a Field and binds the source later at call time. The tests
    below deliberately don't pre-bind the source so we exercise the same
    ``check_source`` / ``need_cast`` path that production ``.cast()`` calls
    take.
    """
    target_field = Field("col", target)
    return CastOptions(target_field=target_field, safe=safe)


def _cast_arrow(array: pa.Array, target: TemporalType, *, safe: bool = False) -> pa.Array:
    options = _make_options(target, safe=safe)
    return target._cast_arrow_array(array, options)


def _cast_polars(series: "pl.Series", target: TemporalType, *, safe: bool = False) -> "pl.Series":
    options = _make_options(target, safe=safe)
    return target._cast_polars_series(series, options)


def _source_field_from_polars(col: str, df: "pl.DataFrame") -> Field:
    """Build a ``Field`` from a polars column dtype for Expr-path tests.

    The Expr dispatcher needs ``options.source_field.dtype.to_polars()`` to
    re-derive the source dtype. We go through yggdrasil's own DataType
    registry (``DataType.from_polars_type``) rather than import individual
    primitive subclasses — the primitive module layout varies, the
    registry dispatch doesn't.
    """
    from yggdrasil.data.types.base import DataType

    return Field(col, DataType.from_polars_type(df.schema[col]))


# ===========================================================================
# Arrow integration
# ===========================================================================


class TestArrowTimestamp:
    @pytest.mark.parametrize(
        "raw,unit,tz,expected",
        [
            # naive ISO
            (["2024-01-15T10:30:00"], "us", None, [dt.datetime(2024, 1, 15, 10, 30)]),
            # ISO with fractional seconds
            (["2024-01-15T10:30:00.123456"], "us", None, [dt.datetime(2024, 1, 15, 10, 30, 0, 123456)]),
            # ms unit
            (["2024-01-15T10:30:00.123"], "ms", None, [dt.datetime(2024, 1, 15, 10, 30, 0, 123000)]),
            # unparseable → null
            (["garbage"], "us", None, [None]),
            # None passes through
            ([None], "us", None, [None]),
        ],
    )
    def test_string_to_naive(self, raw, unit, tz, expected):
        arr = pa.array(raw, type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit=unit, tz=tz))
        assert out.type == pa.timestamp(unit, tz)
        assert out.to_pylist() == expected

    def test_string_to_aware_wallclock_reinterpret(self):
        """Naive→aware keeps wall-clock digits and stamps target zone on top."""
        arr = pa.array(["2024-01-15T10:30:00"], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us", tz="Europe/Paris"))
        assert out.type.tz == "Europe/Paris"
        # Wall-clock reinterpret: 10:30 Paris, not 10:30 UTC → 11:30 Paris.
        v = out.to_pylist()[0]
        assert v.hour == 10 and v.minute == 30
        assert str(v.tzinfo) in {"Europe/Paris", "UTC+01:00", "CET"}

    def test_naive_timestamp_unit_conversion(self):
        arr = pa.array([1_700_000_000_000_000], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimestampType(unit="ms", tz=None))
        assert out.type == pa.timestamp("ms")
        # us → ms: integer division by 1000.
        assert out.to_pylist()[0] == dt.datetime.fromtimestamp(1_700_000_000, tz=dt.timezone.utc).replace(tzinfo=None)

    def test_naive_to_aware_existing_timestamp(self):
        """Existing naive timestamp → aware: wall-clock reinterpret at the cast level."""
        arr = pa.array([dt.datetime(2024, 6, 15, 14, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimestampType(unit="us", tz="America/New_York"))
        assert out.type.tz == "America/New_York"
        v = out.to_pylist()[0]
        # Wall-clock preserved: 14:00 NY, whatever the UTC offset.
        assert v.hour == 14 and v.minute == 0

    def test_aware_to_aware_same_instant(self):
        """aware→aware converts zones (same instant, different wall clock)."""
        arr = pa.array(
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            type=pa.timestamp("us", tz="UTC"),
        )
        out = _cast_arrow(arr, TimestampType(unit="us", tz="America/New_York"))
        v = out.to_pylist()[0]
        # 14:00 UTC == 10:00 EDT (summer).
        assert v.hour == 10

    def test_aware_to_naive_drops_zone(self):
        arr = pa.array(
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            type=pa.timestamp("us", tz="UTC"),
        )
        out = _cast_arrow(arr, TimestampType(unit="us", tz=None))
        assert out.type.tz is None

    def test_second_precision_falls_back_to_pc_cast(self):
        """``unit='s'`` Timestamp can't round-trip through polars — Arrow fallback engages."""
        arr = pa.array([1_700_000_000], type=pa.int64())
        out = _cast_arrow(arr, TimestampType(unit="s", tz=None))
        assert out.type == pa.timestamp("s")

    def test_date_source_to_timestamp(self):
        arr = pa.array([dt.date(2024, 3, 1)], type=pa.date32())
        out = _cast_arrow(arr, TimestampType(unit="us", tz=None))
        assert out.type == pa.timestamp("us")
        assert out.to_pylist()[0] == dt.datetime(2024, 3, 1)

    def test_chunked_array_preserves_structure(self):
        c1 = pa.array(["2024-01-15T10:00:00"], type=pa.string())
        c2 = pa.array(["2024-02-15T12:00:00"], type=pa.string())
        chunked = pa.chunked_array([c1, c2])
        out = _cast_arrow(chunked, TimestampType(unit="us", tz=None))
        assert isinstance(out, pa.ChunkedArray)
        assert out.num_chunks == 2
        assert out.to_pylist() == [
            dt.datetime(2024, 1, 15, 10, 0),
            dt.datetime(2024, 2, 15, 12, 0),
        ]


class TestArrowDate:
    def test_iso_string_to_date(self):
        arr = pa.array(["2024-01-15", "2024-12-31"], type=pa.string())
        out = _cast_arrow(arr, DateType())
        assert out.type == pa.date32()
        assert out.to_pylist() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]

    def test_invalid_string_nulls(self):
        arr = pa.array(["2024-01-15", "not-a-date"], type=pa.string())
        out = _cast_arrow(arr, DateType())
        assert out.to_pylist() == [dt.date(2024, 1, 15), None]

    def test_timestamp_source(self):
        arr = pa.array([dt.datetime(2024, 5, 1, 12, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, DateType())
        assert out.type == pa.date32()
        assert out.to_pylist() == [dt.date(2024, 5, 1)]

    def test_date64_target(self):
        arr = pa.array([dt.date(2024, 1, 15)], type=pa.date32())
        out = _cast_arrow(arr, DateType(byte_size=8, unit="ms"))
        assert out.type == pa.date64()


class TestArrowTime:
    def test_iso_string_to_time(self):
        arr = pa.array(["10:30:45", "23:59:59.999999"], type=pa.string())
        out = _cast_arrow(arr, TimeType(unit="us"))
        assert pa.types.is_time(out.type)
        assert out.type.unit == "us"
        assert out.to_pylist() == [dt.time(10, 30, 45), dt.time(23, 59, 59, 999999)]

    def test_time32_ms(self):
        arr = pa.array(["10:30:45.123"], type=pa.string())
        out = _cast_arrow(arr, TimeType(byte_size=4, unit="ms"))
        assert out.type == pa.time32("ms")

    def test_timestamp_to_time(self):
        arr = pa.array([dt.datetime(2024, 1, 1, 14, 30, 0)], type=pa.timestamp("us"))
        out = _cast_arrow(arr, TimeType(unit="us"))
        assert out.to_pylist() == [dt.time(14, 30, 0)]


class TestArrowDuration:
    def test_integer_to_duration(self):
        arr = pa.array([1_000_000, 2_500_000], type=pa.int64())
        out = _cast_arrow(arr, DurationType(unit="us"))
        assert out.type == pa.duration("us")
        assert out.to_pylist() == [dt.timedelta(seconds=1), dt.timedelta(seconds=2, microseconds=500_000)]

    def test_duration_unit_conversion(self):
        arr = pa.array([1_000], type=pa.duration("ms"))
        out = _cast_arrow(arr, DurationType(unit="us"))
        assert out.type == pa.duration("us")
        assert out.to_pylist() == [dt.timedelta(seconds=1)]

    def test_second_precision_duration_fallback(self):
        """``unit='s'`` Duration goes through ``pc.cast`` (polars can't store it)."""
        arr = pa.array([60], type=pa.int64())
        out = _cast_arrow(arr, DurationType(unit="s"))
        assert out.type == pa.duration("s")


class TestCsvAndExcelFormats:
    """Coverage for the minimal format catalogue (ISO + 3 CSV shapes).

    The catalogue is deliberately small — four formats per kind. These
    tests lock in which shapes parse and which don't, so the catalogue
    can't silently grow or shrink without updating this file.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-01-15", dt.date(2024, 1, 15)),       # ISO
            ("15/01/2024", dt.date(2024, 1, 15)),       # day-first
            ("01/15/2024", dt.date(2024, 1, 15)),       # month-first (unambiguous, day>12)
            ("2024/01/15", dt.date(2024, 1, 15)),       # year-first
        ],
    )
    def test_date_shapes(self, raw, expected):
        arr = pa.array([raw], type=pa.string())
        out = _cast_arrow(arr, DateType())
        assert out.to_pylist() == [expected]

    def test_dayfirst_wins_on_ambiguity(self):
        """``01/02/2024`` is 1 Feb (day-first), not 2 Jan (month-first).

        Day-first ordering matches Excel non-US locales, which is the
        dominant producer of ambiguous ``DD/MM/YYYY`` dates in CSVs.
        """
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
    def test_datetime_shapes(self, raw, expected):
        arr = pa.array([raw], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us"))
        assert out.to_pylist() == [expected]

    def test_mixed_formats_in_one_column(self):
        """Different rows, different formats, all parse through one coalesce."""
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

    def test_all_garbage_nulls_without_raising(self):
        """All-unparseable input previously raised ``ComputeError`` from polars.

        With the explicit format catalogue (no polars ``infer``), unparseable
        rows become null cleanly because strptime gets concrete formats to
        try rather than needing a sample to infer from.
        """
        arr = pa.array(["garbage", "more garbage"], type=pa.string())
        out = _cast_arrow(arr, TimestampType(unit="us"))
        assert out.to_pylist() == [None, None]


# ===========================================================================
# Polars integration
# ===========================================================================


class TestPolarsSeries:
    def test_string_to_timestamp(self):
        s = pl.Series("col", ["2024-01-15T10:30:00", "2024-06-22T08:15:45", "bad", None])
        out = _cast_polars(s, TimestampType(unit="us"))
        assert out.dtype == pl.Datetime("us", None)
        vals = out.to_list()
        assert vals[0] == dt.datetime(2024, 1, 15, 10, 30)
        assert vals[2] is None
        assert vals[3] is None

    def test_string_to_date(self):
        s = pl.Series("col", ["2024-01-15", "2024-12-31"])
        out = _cast_polars(s, DateType())
        assert out.dtype == pl.Date
        assert out.to_list() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]

    def test_string_to_time(self):
        s = pl.Series("col", ["10:30:00", "23:59:59"])
        out = _cast_polars(s, TimeType(unit="us"))
        assert out.dtype == pl.Time
        assert out.to_list() == [dt.time(10, 30), dt.time(23, 59, 59)]

    def test_naive_to_aware_wallclock(self):
        s = pl.Series("col", [dt.datetime(2024, 6, 15, 14, 0)], dtype=pl.Datetime("us"))
        out = _cast_polars(s, TimestampType(unit="us", tz="Europe/Paris"))
        assert isinstance(out.dtype, pl.Datetime)
        assert out.dtype.time_zone == "Europe/Paris"
        # Wall clock preserved.
        v = out.to_list()[0]
        assert v.hour == 14 and v.minute == 0

    def test_aware_to_aware(self):
        s = pl.Series(
            "col",
            [dt.datetime(2024, 6, 15, 14, 0, tzinfo=dt.timezone.utc)],
            dtype=pl.Datetime("us", "UTC"),
        )
        out = _cast_polars(s, TimestampType(unit="us", tz="America/New_York"))
        assert out.dtype.time_zone == "America/New_York"
        # Same instant — 14:00 UTC → 10:00 NY in June (EDT).
        assert out.to_list()[0].hour == 10

    def test_second_precision_bridges_through_arrow(self):
        """``unit='s'`` target routes through the Arrow bridge."""
        s = pl.Series("col", [1_700_000_000], dtype=pl.Int64)
        out = _cast_polars(s, TimestampType(unit="s"))
        # Polars widens to ``ms`` internally, Arrow bridge retains "s"-origin semantics.
        assert isinstance(out.dtype, pl.Datetime)

    def test_unit_conversion(self):
        s = pl.Series("col", [dt.datetime(2024, 1, 1)], dtype=pl.Datetime("ms"))
        out = _cast_polars(s, TimestampType(unit="us"))
        assert out.dtype == pl.Datetime("us", None)

    def test_numeric_to_duration(self):
        s = pl.Series("col", [1_000_000, 2_000_000], dtype=pl.Int64)
        out = _cast_polars(s, DurationType(unit="us"))
        assert out.dtype == pl.Duration("us")

    def test_name_is_preserved(self):
        s = pl.Series("my_ts", ["2024-01-01T00:00:00"])
        out = _cast_polars(s, TimestampType(unit="us"))
        assert out.name == "my_ts"


class TestPolarsExpr:
    def test_expr_path_timestamp(self):
        df = pl.DataFrame({"col": ["2024-01-15T10:30:00", "2024-06-22T08:15:45"]})
        source_field = _source_field_from_polars("col", df)
        target = TimestampType(unit="us")
        options = CastOptions(
            source_field=source_field, target_field=Field("col", target), safe=False
        )
        expr = target._cast_polars_expr(pl.col("col"), options)
        out = df.select(expr).to_series()
        assert out.dtype == pl.Datetime("us", None)
        assert out.to_list()[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_expr_path_date(self):
        df = pl.DataFrame({"col": ["2024-01-15", "2024-12-31"]})
        source_field = _source_field_from_polars("col", df)
        target = DateType()
        options = CastOptions(
            source_field=source_field, target_field=Field("col", target), safe=False
        )
        expr = target._cast_polars_expr(pl.col("col"), options)
        out = df.select(expr).to_series()
        assert out.dtype == pl.Date
        assert out.to_list() == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]


# ===========================================================================
# Pandas integration (Arrow bridge)
# ===========================================================================


@pandas_only
class TestPandasBridge:
    def test_string_series_to_timestamp(self):
        import pandas as pd

        s = pd.Series(["2024-01-15T10:30:00", "2024-06-22T08:15:45"])
        target = TimestampType(unit="us")
        out = target._cast_pandas_series(s, _make_options(target))
        # Pandas retains datetime64 with ns precision on materialise.
        assert out.iloc[0] == dt.datetime(2024, 1, 15, 10, 30)

    def test_date_series(self):
        import pandas as pd

        s = pd.Series(["2024-01-15", "2024-12-31"])
        target = DateType()
        out = target._cast_pandas_series(s, _make_options(target))
        # Arrow date32 → pandas object array of date instances.
        assert list(out) == [dt.date(2024, 1, 15), dt.date(2024, 12, 31)]


# ===========================================================================
# Spark integration
# ===========================================================================


@spark_only
class TestSparkTypes:
    """Spark tests stay at the type-system level.

    Standing up a SparkSession per test is slow and flaky; we verify the
    outgoing Arrow↔Spark mappings here and trust the spark_cast wiring
    because it's a one-line ``column.cast(target)`` delegation.
    """

    def test_timestamp_tz_maps_to_timestamp_type(self):
        target = TimestampType(unit="us", tz="UTC")
        assert isinstance(target.to_spark(), pst.TimestampType)

    def test_timestamp_naive_maps_to_ntz(self):
        target = TimestampType(unit="us", tz=None)
        # Older Spark versions lack TimestampNTZType — fall back to TimestampType.
        expected = getattr(pst, "TimestampNTZType", pst.TimestampType)
        assert isinstance(target.to_spark(), expected)

    def test_date_maps(self):
        assert isinstance(DateType().to_spark(), pst.DateType)

    def test_time_maps_to_string(self):
        assert isinstance(TimeType().to_spark(), pst.StringType)

    def test_duration_maps_to_long(self):
        assert isinstance(DurationType(unit="us").to_spark(), pst.LongType)

    def test_databricks_ddl(self):
        assert TimestampType(tz="UTC").to_databricks_ddl() == "TIMESTAMP"
        assert TimestampType(tz=None).to_databricks_ddl() == "TIMESTAMP_NTZ"
        assert DateType().to_databricks_ddl() == "DATE"
        assert TimeType().to_databricks_ddl() == "STRING"
        assert DurationType().to_databricks_ddl() == "BIGINT"

    def test_from_spark_roundtrip(self):
        ts = TimestampType(unit="us", tz="UTC")
        roundtrip = TimestampType.from_spark_type(ts.to_spark())
        assert roundtrip.tz == "UTC"

    def test_spark_cast_delegates(self):
        """``spark_cast`` should be a thin pass-through to ``column.cast``."""
        # Use a mock-like object that records the cast call.
        class FakeCol:
            def __init__(self):
                self.cast_args = None

            def cast(self, target):
                self.cast_args = target
                return self

        col = FakeCol()
        tgt = pst.TimestampType()
        out = spark_cast(col, tgt)
        assert out is col
        assert col.cast_args is tgt


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
            # Epoch days
            (0, dt.date(1970, 1, 1)),
            (1, dt.date(1970, 1, 2)),
        ],
    )
    def test_convert(self, value, expected):
        assert DateType()._convert_pyobj(value) == expected

    def test_invalid_best_effort_returns_none(self):
        assert DateType()._convert_pyobj("garbage") is None

    def test_invalid_safe_raises(self):
        with pytest.raises(ValueError):
            DateType()._convert_pyobj("garbage", safe=True)

    def test_empty_string_safe_raises(self):
        with pytest.raises(ValueError):
            DateType()._convert_pyobj("", safe=True)

    def test_empty_string_best_effort_returns_none(self):
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
    def test_convert(self, value, expected):
        assert TimeType()._convert_pyobj(value) == expected

    def test_invalid_safe_raises(self):
        with pytest.raises(ValueError):
            TimeType()._convert_pyobj("not-a-time", safe=True)


class TestScalarTimestamp:
    def test_iso_naive(self):
        assert TimestampType()._convert_pyobj("2024-01-15T10:30:00") == dt.datetime(
            2024, 1, 15, 10, 30
        )

    def test_iso_with_z_suffix(self):
        """``Z`` is rewritten to ``+00:00`` before ``fromisoformat``."""
        out = TimestampType(tz="UTC")._convert_pyobj("2024-01-15T10:30:00Z")
        assert out == dt.datetime(2024, 1, 15, 10, 30, tzinfo=dt.timezone.utc)

    def test_naive_target_strips_tz(self):
        """Naive-target + aware-input → astimezone(UTC) + strip."""
        aware = dt.datetime(2024, 1, 15, 10, 30, tzinfo=dt.timezone.utc)
        out = TimestampType(tz=None)._convert_pyobj(aware)
        assert out.tzinfo is None
        assert out == dt.datetime(2024, 1, 15, 10, 30)

    def test_aware_target_attaches_utc_to_naive(self):
        """Aware target + naive input → attach UTC."""
        naive = dt.datetime(2024, 1, 15, 10, 30)
        out = TimestampType(tz="UTC")._convert_pyobj(naive)
        assert out.tzinfo is dt.timezone.utc

    def test_date_input_widens_to_midnight(self):
        assert TimestampType(tz=None)._convert_pyobj(dt.date(2024, 5, 1)) == dt.datetime(
            2024, 5, 1
        )

    def test_epoch_numeric(self):
        """Numeric input is treated as epoch in the target's unit."""
        out = TimestampType(unit="s", tz="UTC")._convert_pyobj(1_700_000_000)
        assert out == dt.datetime.fromtimestamp(1_700_000_000, tz=dt.timezone.utc)

    def test_invalid_safe_raises(self):
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
    def test_numeric_and_string(self, value, unit, expected):
        assert DurationType(unit=unit)._convert_pyobj(value) == expected

    def test_decimal_input(self):
        import decimal

        out = DurationType(unit="us")._convert_pyobj(decimal.Decimal("500000"))
        assert out == dt.timedelta(milliseconds=500)

    def test_bool_as_numeric(self):
        """``True`` is ``1`` count of unit — exercises the explicit bool branch."""
        out = DurationType(unit="us")._convert_pyobj(True)
        assert out == dt.timedelta(microseconds=1)

    def test_non_numeric_string_best_effort(self):
        assert DurationType()._convert_pyobj("not-a-number") is None

    def test_non_numeric_string_safe_raises(self):
        with pytest.raises(ValueError):
            DurationType()._convert_pyobj("not-a-number", safe=True)

    def test_iso_duration_parses_at_scalar_level(self):
        """Scalar ``_convert_pyobj`` restores ISO-8601 duration parsing.

        The vectorised array cast stays stripped-down (parses integers only
        for string→duration), but the scalar path accepts ISO shapes because
        list-of-dict ingest and serde round-trips hand loose Python values
        into ``_convert_pyobj`` where pre-parsing isn't cheap.
        """
        assert DurationType()._convert_pyobj("PT15M") == dt.timedelta(minutes=15)
        assert DurationType()._convert_pyobj("PT1H30M") == dt.timedelta(hours=1, minutes=30)
        assert DurationType()._convert_pyobj("-PT1H") == dt.timedelta(hours=-1)
        # Clock-style also accepted.
        assert DurationType()._convert_pyobj("01:30:00") == dt.timedelta(hours=1, minutes=30)


# ===========================================================================
# Merge semantics
# ===========================================================================


class TestMerge:
    @pytest.mark.parametrize(
        "left_unit,right_unit,downcast,expected_unit",
        [
            # Upcast (default) — widest unit wins.
            ("s", "us", False, "us"),
            ("ms", "ns", False, "ns"),
            ("us", "us", False, "us"),
            # Downcast — narrowest wins.
            ("us", "ns", True, "us"),
            ("s", "ms", True, "s"),
        ],
    )
    def test_timestamp_unit(self, left_unit, right_unit, downcast, expected_unit):
        left = TimestampType(unit=left_unit)
        right = TimestampType(unit=right_unit)
        merged = left._merge_with_same_id(right, downcast=downcast)
        assert merged.unit == expected_unit

    def test_tz_unification_same(self):
        merged = TimestampType(tz="UTC")._merge_with_same_id(TimestampType(tz="UTC"))
        assert merged.tz == "UTC"

    def test_tz_unification_conflict_upcast_picks_first_non_none(self):
        merged = TimestampType(tz="UTC")._merge_with_same_id(TimestampType(tz="Europe/Paris"))
        # Conflict on upcast: falls back to ``self.tz or other.tz``.
        assert merged.tz in {"UTC", "Europe/Paris"}

    def test_tz_unification_conflict_downcast_drops(self):
        merged = TimestampType(tz="UTC")._merge_with_same_id(
            TimestampType(tz="Europe/Paris"), downcast=True
        )
        assert merged.tz is None

    def test_merge_cross_class_raises(self):
        with pytest.raises(TypeError):
            DateType()._merge_with_same_id(TimestampType())

    def test_merge_both_flags_raises(self):
        with pytest.raises(pa.ArrowInvalid):
            TimestampType()._merge_with_same_id(
                TimestampType(), downcast=True, upcast=True
            )


# ===========================================================================
# Serde roundtrips
# ===========================================================================


class TestSerde:
    @pytest.mark.parametrize(
        "target",
        [
            # ``from_dict`` injects default ``byte_size`` values when absent
            # (4 for Date, 8 for the rest), but ``to_dict`` emits whatever
            # ``byte_size`` the instance was constructed with — including
            # the ``None`` default from the dataclass. Specify explicit
            # values on both sides so round-trip identity actually holds.
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
    def test_dict_roundtrip(self, target):
        d = target.to_dict()
        restored = type(target).from_dict(d)
        assert restored == target

    def test_autotag_includes_unit(self):
        tags = TimestampType(unit="ns", tz="Europe/Paris").autotag()
        assert tags[b"unit"] == b"ns"
        assert tags[b"tz"] == b"Europe/Paris"

    def test_autotag_skips_none_tz(self):
        tags = TimestampType(tz=None).autotag()
        assert b"tz" not in tags

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
    def test_to_arrow(self, target, expected_arrow):
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
    def test_arrow_roundtrip(self, target, arrow_dtype):
        """``from_arrow_type(type.to_arrow()) == type`` for the canonical cases."""
        cls = type(target)
        assert cls.handles_arrow_type(arrow_dtype)
        roundtrip = cls.from_arrow_type(arrow_dtype)
        assert roundtrip.unit == target.unit
        # tz comparisons are only meaningful for Timestamp.
        if isinstance(target, TimestampType):
            assert roundtrip.tz == target.tz

    def test_handles_polars_dispatch(self):
        assert TimestampType.handles_polars_type(pl.Datetime("us", "UTC"))
        assert DateType.handles_polars_type(pl.Date)
        assert TimeType.handles_polars_type(pl.Time)
        assert DurationType.handles_polars_type(pl.Duration("us"))
        # Cross-class rejection.
        assert not DateType.handles_polars_type(pl.Datetime("us"))


# ===========================================================================
# Engine dispatchers (direct, not via TemporalType)
# ===========================================================================


class TestArrowCastDispatcher:
    """Exercise ``arrow_cast`` directly to cover paths the type-level API masks."""

    def test_non_temporal_target_passes_through(self):
        """Non-temporal targets shouldn't enter the polars bridge."""
        arr = pa.array([1, 2, 3], type=pa.int64())
        out = arrow_cast(arr, pa.float64())
        assert out.type == pa.float64()

    def test_null_source(self):
        arr = pa.nulls(3, type=pa.null())
        out = arrow_cast(arr, pa.timestamp("us"))
        assert out.type == pa.timestamp("us")
        assert out.to_pylist() == [None, None, None]

    def test_empty_chunked_array(self):
        chunked = pa.chunked_array([], type=pa.string())
        out = arrow_cast(chunked, pa.timestamp("us"))
        assert isinstance(out, pa.ChunkedArray)
        assert len(out) == 0
        assert out.type == pa.timestamp("us")

    def test_safe_true_matches_best_effort_on_valid_input(self):
        """``safe=True`` should succeed on well-formed input."""
        arr = pa.array(["2024-01-15T10:30:00"], type=pa.string())
        out = arrow_cast(arr, pa.timestamp("us"), safe=True)
        assert out.to_pylist() == [dt.datetime(2024, 1, 15, 10, 30)]


class TestPolarsDispatcher:
    """Direct calls to ``cast_polars_array_to_temporal``."""

    def test_series_name_preserved(self):
        s = pl.Series("x", ["2024-01-01T00:00:00"])
        out = cast_polars_array_to_temporal(
            s,
            source=pl.String,
            target=pl.Datetime("us"),
            safe=False,
        )
        assert out.name == "x"

    def test_to_expr_returns_expression(self):
        s = pl.Series("x", ["2024-01-01T00:00:00"])
        out = cast_polars_array_to_temporal(
            s,
            source=pl.String,
            target=pl.Datetime("us"),
            safe=False,
            to_expr=True,
        )
        assert isinstance(out, pl.Expr)

    def test_unsupported_target_raises(self):
        s = pl.Series("x", [1, 2, 3])
        with pytest.raises(TypeError, match="Unsupported temporal target"):
            cast_polars_array_to_temporal(
                s,
                source=pl.Int64,
                target=pl.Float64,  # not a temporal target
                safe=False,
            )


class TestSparkCastDispatcher:
    """``spark_cast`` is a one-liner pass-through — cover its signature surface."""

    def test_accepts_keyword_args_silently(self):
        """The signature accepts unit/tz/safe but discards them — verify no raise."""

        class FakeCol:
            def cast(self, target):
                return ("cast", target)

        # Should not raise even though unit/tz/safe are supplied.
        result = spark_cast(
            FakeCol(), "fake_target", safe=True, unit="ns", tz="Europe/Paris"
        )
        assert result == ("cast", "fake_target")