"""Vectorized best-effort temporal cast behavior on TemporalType subclasses.

Covers the Arrow path directly (canonical) and the Polars path via the
DataType.cast_polars_series entry point. Pandas runs through Arrow so
Arrow coverage is enough; Spark is optional and exercised only when PySpark
is installed.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.types import (
    DateType,
    DurationType,
    TimestampType,
    TimeType,
)
from yggdrasil.data.types import StringType
from yggdrasil.data.types._temporal_cast import (
    arrow_cast_to_date,
    arrow_cast_to_duration,
    arrow_cast_to_string,
    arrow_cast_to_timestamp,
    arrow_date_to_string,
    arrow_duration_to_string,
    arrow_str_to_date,
    arrow_str_to_duration,
    arrow_str_to_time,
    arrow_str_to_timestamp,
    arrow_temporal_to_string,
    arrow_time_to_string,
    arrow_timestamp_to_string,
    attach_fractional_seconds,
    nullify_empty_strings,
)

# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------


def test_arrow_str_to_timestamp_naive_multi_format() -> None:
    arr = pa.array(
        [
            "2023-01-02T03:04:05",
            "2023-05-17 14:30:00",
            "2024/01/15",
            "15/03/2024",
            None,
        ]
    )
    out = arrow_str_to_timestamp(arr, unit="us", tz=None)
    assert out.type == pa.timestamp("us")
    values = out.to_pylist()
    assert values[0] == dt.datetime(2023, 1, 2, 3, 4, 5)
    assert values[1] == dt.datetime(2023, 5, 17, 14, 30, 0)
    assert values[2] == dt.datetime(2024, 1, 15, 0, 0, 0)
    assert values[3] == dt.datetime(2024, 3, 15, 0, 0, 0)
    assert values[4] is None


def test_arrow_str_to_timestamp_tz_aware_emits_utc() -> None:
    arr = pa.array(["2023-01-02T03:04:05+02:00", "2023-05-17 14:30:00+0000"])
    out = arrow_str_to_timestamp(arr, unit="us", tz="UTC")
    assert out.type == pa.timestamp("us", tz="UTC")
    values = out.to_pylist()
    assert values[0] == dt.datetime(2023, 1, 2, 1, 4, 5, tzinfo=dt.timezone.utc)
    assert values[1] == dt.datetime(2023, 5, 17, 14, 30, 0, tzinfo=dt.timezone.utc)


def test_arrow_str_to_timestamp_blank_and_garbage_null() -> None:
    arr = pa.array(["", "   ", "not-a-date", None])
    out = arrow_str_to_timestamp(arr, unit="us")
    assert out.null_count == 4


def test_arrow_str_to_date_multi_format() -> None:
    arr = pa.array(["2023-01-02", "15/03/2024", "2023-05-17T14:30:00", "junk", None])
    out = arrow_str_to_date(arr)
    assert out.type == pa.date32()
    values = out.to_pylist()
    assert values[0] == dt.date(2023, 1, 2)
    assert values[1] == dt.date(2024, 3, 15)
    assert values[2] == dt.date(2023, 5, 17)
    assert values[3] is None
    assert values[4] is None


def test_arrow_str_to_time_multi_format() -> None:
    arr = pa.array(["10:30:00", "23:59:59.123", "00:00", "bad"])
    out = arrow_str_to_time(arr, unit="us")
    assert out.type == pa.time64("us")
    values = out.to_pylist()
    assert values[0] == dt.time(10, 30, 0)
    # Fractional seconds are folded back on by default.
    assert values[1] == dt.time(23, 59, 59, 123000)
    assert values[2] == dt.time(0, 0, 0)
    assert values[3] is None


def test_arrow_str_to_time_drop_fractional_opt_out() -> None:
    arr = pa.array(["23:59:59.987654321"])
    out = arrow_str_to_time(arr, unit="us", keep_fractional=False)
    assert out.to_pylist() == [dt.time(23, 59, 59)]


def test_arrow_str_to_duration_integer_strings() -> None:
    arr = pa.array(["60", "3600", None])
    out = arrow_str_to_duration(arr, unit="s")
    assert out.type == pa.duration("s")
    values = out.to_pylist()
    assert values[0] == dt.timedelta(seconds=60)
    assert values[1] == dt.timedelta(seconds=3600)
    assert values[2] is None


def test_arrow_cast_to_timestamp_handles_numeric_epoch_scales() -> None:
    arr = pa.array(
        [
            1_700_000_000,
            1_700_000_000_000,
            1_700_000_000_000_000,
        ],
        type=pa.int64(),
    )
    out = arrow_cast_to_timestamp(arr, unit="us")
    values = out.to_pylist()
    # All three encodings resolve to the same wall time.
    assert values[0] == values[1] == values[2]
    assert values[0] == dt.datetime(2023, 11, 14, 22, 13, 20)


def test_arrow_cast_to_timestamp_from_date_anchors_midnight() -> None:
    arr = pa.array([dt.date(2024, 1, 15), dt.date(2024, 5, 17)], type=pa.date32())
    out = arrow_cast_to_timestamp(arr, unit="us")
    assert out.to_pylist() == [
        dt.datetime(2024, 1, 15, 0, 0, 0),
        dt.datetime(2024, 5, 17, 0, 0, 0),
    ]


def test_arrow_cast_to_timestamp_from_time_anchors_epoch() -> None:
    arr = pa.array([dt.time(10, 30), dt.time(23, 59, 59)], type=pa.time64("us"))
    out = arrow_cast_to_timestamp(arr, unit="us")
    values = out.to_pylist()
    assert values[0].time() == dt.time(10, 30)
    assert values[0].date() == dt.date(1970, 1, 1)


def test_arrow_cast_to_date_from_timestamp_drops_time_component() -> None:
    arr = pa.array(
        [dt.datetime(2024, 1, 15, 14, 30), dt.datetime(2024, 5, 17, 23, 59)],
        type=pa.timestamp("us"),
    )
    out = arrow_cast_to_date(arr)
    assert out.to_pylist() == [dt.date(2024, 1, 15), dt.date(2024, 5, 17)]


def test_arrow_cast_to_duration_from_int_no_autoscale() -> None:
    arr = pa.array([60, 3600], type=pa.int64())
    out = arrow_cast_to_duration(arr, unit="s")
    assert out.to_pylist() == [dt.timedelta(seconds=60), dt.timedelta(hours=1)]


def test_arrow_cast_to_duration_from_timestamp_subtracts_epoch() -> None:
    arr = pa.array([dt.datetime(1970, 1, 2, 0, 0, 0)], type=pa.timestamp("s"))
    out = arrow_cast_to_duration(arr, unit="s")
    assert out.to_pylist() == [dt.timedelta(days=1)]


def test_arrow_temporal_to_string_roundtrip_duration() -> None:
    dur = pa.array([60, 3600], type=pa.duration("s"))
    s = arrow_temporal_to_string(dur)
    assert s.to_pylist() == ["60", "3600"]


def test_nullify_empty_strings_passes_through_non_strings() -> None:
    arr = pa.array([1, 2, 3])
    assert nullify_empty_strings(arr) is arr


def test_nullify_empty_strings_nulls_blanks() -> None:
    arr = pa.array(["x", " ", "", None, "y"])
    out = nullify_empty_strings(arr)
    assert out.to_pylist() == ["x", None, None, None, "y"]


# ---------------------------------------------------------------------------
# TemporalType subclasses — Arrow path
# ---------------------------------------------------------------------------


def test_timestamp_type_cast_arrow_array_from_string() -> None:
    arr = pa.array(["2023-01-02T03:04:05", "not-a-date", None])
    out = TimestampType(unit="us")._cast_arrow_array(arr, CastOptions())
    assert out.type == pa.timestamp("us")
    assert out.to_pylist()[0] == dt.datetime(2023, 1, 2, 3, 4, 5)
    assert out.null_count == 2


def test_timestamp_type_cast_arrow_array_mixed_epoch_scales() -> None:
    arr = pa.array([1_700_000_000, 1_700_000_000_000])
    out = TimestampType(unit="us")._cast_arrow_array(arr, CastOptions())
    values = out.to_pylist()
    assert values[0] == values[1]


def test_timestamp_type_cast_arrow_array_tz_target() -> None:
    arr = pa.array(["2023-05-17T14:30:00+02:00"])
    out = TimestampType(unit="us", tz="UTC")._cast_arrow_array(arr, CastOptions())
    assert out.type == pa.timestamp("us", tz="UTC")
    assert out.to_pylist()[0] == dt.datetime(
        2023, 5, 17, 12, 30, 0, tzinfo=dt.timezone.utc
    )


def test_date_type_cast_arrow_array_from_string_pulls_date() -> None:
    arr = pa.array(["2023-05-17T14:30:00", "2024-01-15", "bad"])
    out = DateType()._cast_arrow_array(arr, CastOptions())
    values = out.to_pylist()
    assert values[0] == dt.date(2023, 5, 17)
    assert values[1] == dt.date(2024, 1, 15)
    assert values[2] is None


def test_time_type_cast_arrow_array_from_string() -> None:
    arr = pa.array(["10:30", "23:59:59.5"])
    out = TimeType(unit="us")._cast_arrow_array(arr, CastOptions())
    values = out.to_pylist()
    assert values[0] == dt.time(10, 30)
    # Fractional seconds are folded in by default (best-effort mode).
    assert values[1] == dt.time(23, 59, 59, 500000)


def test_time_type_cast_arrow_array_safe_drops_fractional() -> None:
    arr = pa.array(["23:59:59.5"])
    out = TimeType(unit="us")._cast_arrow_array(arr, CastOptions(safe=True))
    assert out.to_pylist() == [dt.time(23, 59, 59)]


def test_duration_type_cast_arrow_array_from_integer_string() -> None:
    arr = pa.array(["60", "3600"])
    out = DurationType(unit="s")._cast_arrow_array(arr, CastOptions())
    assert out.to_pylist() == [dt.timedelta(seconds=60), dt.timedelta(hours=1)]


def test_duration_type_cast_arrow_array_from_timestamp_since_epoch() -> None:
    arr = pa.array([dt.datetime(1970, 1, 2)], type=pa.timestamp("s"))
    out = DurationType(unit="s")._cast_arrow_array(arr, CastOptions())
    assert out.to_pylist() == [dt.timedelta(days=1)]


# ---------------------------------------------------------------------------
# Polars path
# ---------------------------------------------------------------------------


@pytest.fixture
def pl():
    pl = pytest.importorskip("polars")
    import yggdrasil.polars.cast  # noqa: F401 — register converters

    return pl


def test_polars_timestamp_type_from_strings(pl) -> None:
    s = pl.Series("t", ["2023-01-02T03:04:05", "2024/01/15", None])
    out = TimestampType(unit="us")._cast_polars_series(s, CastOptions())
    assert out.dtype == pl.Datetime("us")
    values = out.to_list()
    assert values[0] == dt.datetime(2023, 1, 2, 3, 4, 5)
    assert values[1] == dt.datetime(2024, 1, 15, 0, 0, 0)
    assert values[2] is None


def test_polars_date_type_from_strings(pl) -> None:
    s = pl.Series("d", ["2023-01-02", "bad"])
    out = DateType()._cast_polars_series(s, CastOptions())
    assert out.dtype == pl.Date
    assert out.to_list() == [dt.date(2023, 1, 2), None]


def test_polars_time_type_from_strings(pl) -> None:
    s = pl.Series("t", ["10:30:00", "23:59:59"])
    out = TimeType(unit="us")._cast_polars_series(s, CastOptions())
    assert out.dtype == pl.Time
    assert out.to_list() == [dt.time(10, 30), dt.time(23, 59, 59)]


def test_polars_duration_type_second_unit_roundtrips_via_arrow(pl) -> None:
    s = pl.Series("dur", [60, 3600])
    out = DurationType(unit="s")._cast_polars_series(s, CastOptions())
    # Polars cannot store seconds; ms is the closest representation but values must match.
    assert out.to_list() == [dt.timedelta(seconds=60), dt.timedelta(hours=1)]


def test_polars_timestamp_type_from_date_series(pl) -> None:
    s = pl.Series("d", [dt.date(2023, 1, 2), dt.date(2024, 5, 17)])
    out = TimestampType(unit="us")._cast_polars_series(s, CastOptions())
    assert out.to_list() == [
        dt.datetime(2023, 1, 2, 0, 0, 0),
        dt.datetime(2024, 5, 17, 0, 0, 0),
    ]


# ---------------------------------------------------------------------------
# Pandas path (via Arrow)
# ---------------------------------------------------------------------------


def test_pandas_cast_string_to_timestamp_goes_through_arrow() -> None:
    pd = pytest.importorskip("pandas")

    s = pd.Series(["2023-01-02T03:04:05", "2024-01-15"], dtype="string")
    out = TimestampType(unit="us")._cast_pandas_series(s, CastOptions())
    assert list(out) == [
        dt.datetime(2023, 1, 2, 3, 4, 5),
        dt.datetime(2024, 1, 15, 0, 0),
    ]


def test_pandas_cast_int_to_duration_no_autoscale() -> None:
    pd = pytest.importorskip("pandas")

    s = pd.Series([60, 3600], dtype="int64")
    out = DurationType(unit="s")._cast_pandas_series(s, CastOptions())
    assert list(out) == [dt.timedelta(seconds=60), dt.timedelta(hours=1)]


# ---------------------------------------------------------------------------
# Fractional-second preservation
# ---------------------------------------------------------------------------


def test_attach_fractional_seconds_preserves_microseconds() -> None:
    ts = pa.array([dt.datetime(2023, 1, 2, 3, 4, 5)], type=pa.timestamp("us"))
    source = pa.array(["2023-01-02T03:04:05.123"])
    out = attach_fractional_seconds(ts, source, unit="us")
    assert out.to_pylist() == [dt.datetime(2023, 1, 2, 3, 4, 5, 123000)]


def test_attach_fractional_seconds_handles_mixed_rows() -> None:
    ts = pa.array(
        [
            dt.datetime(2023, 1, 2, 3, 4, 5),
            dt.datetime(2023, 5, 17, 14, 30, 0),
            None,
        ],
        type=pa.timestamp("us"),
    )
    source = pa.array(["2023-01-02T03:04:05.123456789", "no fraction here", None])
    out = attach_fractional_seconds(ts, source, unit="us")
    assert out.to_pylist() == [
        dt.datetime(2023, 1, 2, 3, 4, 5, 123456),  # truncated to us precision
        dt.datetime(2023, 5, 17, 14, 30, 0),
        None,
    ]


def test_arrow_str_to_timestamp_keeps_fractional_by_default() -> None:
    arr = pa.array(["2023-01-02T03:04:05.123", "2023-05-17 14:30:00.5"])
    out = arrow_str_to_timestamp(arr, unit="us")
    assert out.to_pylist() == [
        dt.datetime(2023, 1, 2, 3, 4, 5, 123000),
        dt.datetime(2023, 5, 17, 14, 30, 0, 500000),
    ]


def test_arrow_str_to_timestamp_drops_fractional_when_disabled() -> None:
    arr = pa.array(["2023-01-02T03:04:05.123"])
    out = arrow_str_to_timestamp(arr, unit="us", keep_fractional=False)
    assert out.to_pylist() == [dt.datetime(2023, 1, 2, 3, 4, 5)]


def test_timestamp_type_preserves_fractional_seconds() -> None:
    arr = pa.array(["2023-01-02T03:04:05.999"])
    out = TimestampType(unit="us")._cast_arrow_array(arr, CastOptions())
    assert out.to_pylist() == [dt.datetime(2023, 1, 2, 3, 4, 5, 999000)]


# ---------------------------------------------------------------------------
# Unsafe timezone casting
# ---------------------------------------------------------------------------


def test_arrow_str_to_timestamp_unsafe_tz_reinterprets_wall_clock() -> None:
    arr = pa.array(["2023-01-02T03:04:05"])
    out = arrow_str_to_timestamp(arr, unit="us", tz="Europe/Paris", unsafe_tz=True)
    assert out.type == pa.timestamp("us", tz="Europe/Paris")
    # Wall-clock 03:04:05 stays 03:04:05 in Paris.
    assert out.to_pylist()[0].replace(tzinfo=None) == dt.datetime(2023, 1, 2, 3, 4, 5)


def test_arrow_str_to_timestamp_safe_tz_assumes_utc() -> None:
    arr = pa.array(["2023-01-02T03:04:05"])
    out = arrow_str_to_timestamp(arr, unit="us", tz="Europe/Paris", unsafe_tz=False)
    # UTC 03:04:05 → Paris wall-clock 04:04:05 (CET offset +1).
    assert out.to_pylist()[0].replace(tzinfo=None) == dt.datetime(2023, 1, 2, 4, 4, 5)


def test_arrow_cast_to_timestamp_unsafe_tz_on_naive_source() -> None:
    naive = pa.array([dt.datetime(2023, 1, 2, 3, 4, 5)], type=pa.timestamp("us"))
    out = arrow_cast_to_timestamp(naive, unit="us", tz="Europe/Paris", unsafe_tz=True)
    assert out.type == pa.timestamp("us", tz="Europe/Paris")
    assert out.to_pylist()[0].replace(tzinfo=None) == dt.datetime(2023, 1, 2, 3, 4, 5)


def test_timestamp_type_safe_mode_uses_utc_assumption() -> None:
    arr = pa.array(["2023-01-02T03:04:05"])
    out = TimestampType(unit="us", tz="Europe/Paris")._cast_arrow_array(
        arr, CastOptions(safe=True)
    )
    assert out.to_pylist()[0].replace(tzinfo=None) == dt.datetime(2023, 1, 2, 4, 4, 5)


def test_timestamp_type_best_effort_reinterprets_wall_clock() -> None:
    arr = pa.array(["2023-01-02T03:04:05"])
    out = TimestampType(unit="us", tz="Europe/Paris")._cast_arrow_array(
        arr, CastOptions()
    )
    assert out.to_pylist()[0].replace(tzinfo=None) == dt.datetime(2023, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Temporal → string casters
# ---------------------------------------------------------------------------


def test_arrow_timestamp_to_string_default_format() -> None:
    arr = pa.array([dt.datetime(2023, 1, 2, 3, 4, 5, 123000)], type=pa.timestamp("us"))
    out = arrow_timestamp_to_string(arr)
    s = out.to_pylist()[0]
    assert s.startswith("2023-01-02T03:04:05")


def test_arrow_timestamp_to_string_custom_format() -> None:
    arr = pa.array([dt.datetime(2023, 1, 2, 3, 4, 5)], type=pa.timestamp("us"))
    out = arrow_timestamp_to_string(arr, fmt="%Y/%m/%d %H:%M")
    assert out.to_pylist() == ["2023/01/02 03:04"]


def test_arrow_date_to_string_default_iso() -> None:
    arr = pa.array([dt.date(2023, 5, 17)], type=pa.date32())
    out = arrow_date_to_string(arr)
    assert out.to_pylist() == ["2023-05-17"]


def test_arrow_time_to_string_default_vs_custom() -> None:
    arr = pa.array([dt.time(10, 30, 0)], type=pa.time64("us"))
    default_out = arrow_time_to_string(arr)
    assert default_out.to_pylist() == ["10:30:00.000000"]

    custom_out = arrow_time_to_string(arr, fmt="%H:%M")
    assert custom_out.to_pylist() == ["10:30"]


def test_arrow_duration_to_string_integer_roundtrip() -> None:
    arr = pa.array([60, 3600], type=pa.duration("s"))
    out = arrow_duration_to_string(arr)
    assert out.to_pylist() == ["60", "3600"]


def test_arrow_cast_to_string_dispatches_on_each_temporal() -> None:
    ts = arrow_cast_to_string(
        pa.array([dt.datetime(2023, 1, 2, 3, 4, 5)], type=pa.timestamp("us"))
    )
    assert ts.to_pylist()[0].startswith("2023-01-02")

    d = arrow_cast_to_string(pa.array([dt.date(2023, 1, 2)], type=pa.date32()))
    assert d.to_pylist() == ["2023-01-02"]

    dur = arrow_cast_to_string(pa.array([60], type=pa.duration("s")))
    assert dur.to_pylist() == ["60"]


def test_arrow_temporal_to_string_alias_keeps_working() -> None:
    arr = pa.array([dt.date(2024, 1, 15)], type=pa.date32())
    assert arrow_temporal_to_string(arr).to_pylist() == ["2024-01-15"]


# ---------------------------------------------------------------------------
# StringType target — temporal source dispatch
# ---------------------------------------------------------------------------


def test_string_type_cast_arrow_timestamp_source() -> None:
    arr = pa.array([dt.datetime(2023, 1, 2, 3, 4, 5)], type=pa.timestamp("us"))
    out = StringType()._cast_arrow_array(
        arr,
        CastOptions.check(source=arr),
    )
    assert out.to_pylist()[0].startswith("2023-01-02T03:04:05")


def test_string_type_cast_arrow_date_source() -> None:
    arr = pa.array([dt.date(2024, 5, 17)], type=pa.date32())
    out = StringType()._cast_arrow_array(
        arr,
        CastOptions.check(source=arr),
    )
    assert out.to_pylist() == ["2024-05-17"]


def test_string_type_cast_arrow_duration_source_survives() -> None:
    arr = pa.array([60, 3600], type=pa.duration("s"))
    out = StringType()._cast_arrow_array(
        arr,
        CastOptions.check(source=arr),
    )
    assert out.to_pylist() == ["60", "3600"]


def test_field_cast_preserves_default_field_roundtrip() -> None:
    from yggdrasil.data.data_field import Field

    # Ensure normal non-temporal string casting still works.
    arr = pa.array([1, 2, 3], type=pa.int64())
    f = Field("n", StringType())
    out = f.cast_arrow_array(arr)
    assert out.to_pylist() == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Datetime string shape coverage — fractional, no fractional, " UTC", Z
# ---------------------------------------------------------------------------

_UTC = dt.timezone.utc


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # No fractional seconds
        ("2023-01-02T03:04:05", dt.datetime(2023, 1, 2, 3, 4, 5)),
        ("2023-01-02 03:04:05", dt.datetime(2023, 1, 2, 3, 4, 5)),
        # Fractional seconds, various widths
        ("2023-01-02T03:04:05.1", dt.datetime(2023, 1, 2, 3, 4, 5, 100000)),
        ("2023-01-02T03:04:05.12", dt.datetime(2023, 1, 2, 3, 4, 5, 120000)),
        ("2023-01-02T03:04:05.123", dt.datetime(2023, 1, 2, 3, 4, 5, 123000)),
        ("2023-01-02T03:04:05.123456", dt.datetime(2023, 1, 2, 3, 4, 5, 123456)),
        # Over-precision — truncated to target unit
        ("2023-01-02T03:04:05.123456789", dt.datetime(2023, 1, 2, 3, 4, 5, 123456)),
    ],
)
def test_arrow_str_to_timestamp_naive_shape(value: str, expected: dt.datetime) -> None:
    arr = pa.array([value])
    out = arrow_str_to_timestamp(arr, unit="us", tz=None)
    assert out.type == pa.timestamp("us")
    assert out.to_pylist() == [expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Z suffix — no fractional
        ("2023-01-02T03:04:05Z", dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)),
        ("2023-01-02 03:04:05Z", dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)),
        # Z suffix with fractional
        (
            "2023-01-02T03:04:05.5Z",
            dt.datetime(2023, 1, 2, 3, 4, 5, 500000, tzinfo=_UTC),
        ),
        (
            "2023-01-02T03:04:05.123456Z",
            dt.datetime(2023, 1, 2, 3, 4, 5, 123456, tzinfo=_UTC),
        ),
        # " UTC" (space-separated) — no fractional
        ("2023-01-02T03:04:05 UTC", dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)),
        ("2023-01-02 03:04:05 UTC", dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)),
        # "UTC" (no space) — also normalized
        ("2023-01-02T03:04:05UTC", dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)),
        # " UTC" with fractional
        (
            "2023-01-02T03:04:05.123 UTC",
            dt.datetime(2023, 1, 2, 3, 4, 5, 123000, tzinfo=_UTC),
        ),
        (
            "2023-01-02 03:04:05.987654 UTC",
            dt.datetime(2023, 1, 2, 3, 4, 5, 987654, tzinfo=_UTC),
        ),
        # Numeric offset with fractional
        (
            "2023-01-02T05:04:05.5+02:00",
            dt.datetime(2023, 1, 2, 3, 4, 5, 500000, tzinfo=_UTC),
        ),
    ],
)
def test_arrow_str_to_timestamp_tz_aware_shape(
    value: str, expected: dt.datetime
) -> None:
    arr = pa.array([value])
    out = arrow_str_to_timestamp(arr, unit="us", tz="UTC")
    assert out.type == pa.timestamp("us", tz="UTC")
    assert out.to_pylist() == [expected]


def test_arrow_str_to_timestamp_mixed_utc_and_z_in_one_column() -> None:
    arr = pa.array(
        [
            "2023-01-02T03:04:05Z",
            "2023-01-02T03:04:05 UTC",
            "2023-01-02T03:04:05.123Z",
            "2023-01-02T03:04:05.123 UTC",
            "2023-01-02T03:04:05",  # naive — unsafe_tz assumes target zone
        ]
    )
    out = arrow_str_to_timestamp(arr, unit="us", tz="UTC")
    values = out.to_pylist()
    assert values[0] == dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)
    assert values[1] == dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)
    assert values[2] == dt.datetime(2023, 1, 2, 3, 4, 5, 123000, tzinfo=_UTC)
    assert values[3] == dt.datetime(2023, 1, 2, 3, 4, 5, 123000, tzinfo=_UTC)
    # Naive + unsafe_tz=True → wall-clock reinterpreted as UTC.
    assert values[4] == dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)


def test_arrow_str_to_timestamp_keep_fractional_false_drops_subsecond() -> None:
    arr = pa.array(
        [
            "2023-01-02T03:04:05Z",
            "2023-01-02T03:04:05.987Z",
            "2023-01-02T03:04:05.987 UTC",
        ]
    )
    out = arrow_str_to_timestamp(arr, unit="us", tz="UTC", keep_fractional=False)
    assert out.to_pylist() == [
        dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC),
        dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC),
        dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC),
    ]


def test_arrow_str_to_date_accepts_utc_and_z_suffixed_datetimes() -> None:
    arr = pa.array(
        [
            "2023-01-02T03:04:05Z",
            "2023-01-02T03:04:05 UTC",
            "2023-01-02T03:04:05.999 UTC",
            "2023-05-17",
        ]
    )
    out = arrow_str_to_date(arr)
    assert out.to_pylist() == [
        dt.date(2023, 1, 2),
        dt.date(2023, 1, 2),
        dt.date(2023, 1, 2),
        dt.date(2023, 5, 17),
    ]


@pytest.mark.parametrize(
    ("unit", "expected_micros"),
    [("ms", 123_000), ("us", 123_456), ("ns", 123_456_789)],
)
def test_arrow_str_to_timestamp_fractional_precision_per_unit(
    unit: str, expected_micros: int
) -> None:
    # Feed 9 fractional digits and verify each unit only keeps as many as it can.
    arr = pa.array(["2023-01-02T03:04:05.123456789"])
    out = arrow_str_to_timestamp(arr, unit=unit, tz=None)

    # For ns we can't decode to datetime (arrow clamps to us in to_pylist on
    # some versions); compare via int64 in that case.
    if unit == "ns":
        as_int = pa.compute.cast(out, pa.int64()).to_pylist()[0]
        # Epoch for 2023-01-02T03:04:05 = 1_672_628_645 seconds.
        expected = 1_672_628_645 * 1_000_000_000 + expected_micros
        assert as_int == expected
        return

    if unit == "ms":
        expected_dt = dt.datetime(2023, 1, 2, 3, 4, 5, 123000)
    else:
        expected_dt = dt.datetime(2023, 1, 2, 3, 4, 5, 123456)
    assert out.to_pylist() == [expected_dt]


def test_timestamp_type_handles_all_shapes_in_one_cast() -> None:
    arr = pa.array(
        [
            "2023-01-02T03:04:05",
            "2023-01-02T03:04:05.5",
            "2023-01-02T03:04:05Z",
            "2023-01-02T03:04:05.123Z",
            "2023-01-02T03:04:05 UTC",
            "2023-01-02T03:04:05.999 UTC",
        ]
    )
    out = TimestampType(unit="us", tz="UTC")._cast_arrow_array(arr, CastOptions())
    values = out.to_pylist()
    assert all(v.tzinfo is not None for v in values)
    # Wall-clock payloads are identical across the no-fractional rows.
    assert (
        values[0]
        == values[2]
        == values[4]
        == dt.datetime(2023, 1, 2, 3, 4, 5, tzinfo=_UTC)
    )
    # Fractional payloads preserve sub-second precision.
    assert values[1].microsecond == 500000
    assert values[3].microsecond == 123000
    assert values[5].microsecond == 999000
