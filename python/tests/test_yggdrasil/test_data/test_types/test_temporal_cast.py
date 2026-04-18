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
from yggdrasil.data.types._temporal_cast import (
    arrow_cast_to_date,
    arrow_cast_to_duration,
    arrow_cast_to_timestamp,
    arrow_str_to_date,
    arrow_str_to_duration,
    arrow_str_to_time,
    arrow_str_to_timestamp,
    arrow_temporal_to_string,
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
    # Sub-second precision isn't preserved — Arrow's strptime can't parse %f.
    assert values[1] == dt.time(23, 59, 59)
    assert values[2] == dt.time(0, 0, 0)
    assert values[3] is None


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
    # Fractional seconds are stripped — Arrow strptime doesn't parse %f.
    assert values[1] == dt.time(23, 59, 59)


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
