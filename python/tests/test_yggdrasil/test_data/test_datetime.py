"""``yggdrasil.data.cast.datetime`` — string + numeric → date / datetime.

These helpers are the single place that decides "what does this
loose datetime input mean?" — used by the temporal type's
``_convert_pyobj``, the cast registry, and any caller that wants to
parse a CSV column or a JSON timestamp without writing a regex.

Tests grouped by behavior:

* String normalization / parsing — colon-less offsets, slash dates,
  compact ISO, fractions, Z suffix.
* Numeric inference — int and float are interpreted as epoch values
  with the unit derived from magnitude.
* Truncation — ``truncate_datetime_value`` accepts ISO interval
  strings or interval-spec instances.
"""
from __future__ import annotations

import datetime as dt
import math
import time

import pytest

from yggdrasil.data.cast.datetime import (
    CURRENT_TZINFO,
    _coerce_interval,
    any_to_date,
    any_to_datetime,
    float_to_date,
    float_to_datetime,
    int_to_date,
    int_to_datetime,
    normalize_datetime_string,
    str_to_timedelta,
    truncate_datetime_value,
)


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------


class TestNormalize:

    def test_colon_less_positive_offset_gets_colon(self) -> None:
        assert normalize_datetime_string(
            "2020-01-01T00:00:00+0200"
        ).endswith("+02:00")

    def test_colon_less_negative_offset_gets_colon(self) -> None:
        assert normalize_datetime_string(
            "2020-01-01 00:00:00-0530"
        ).endswith("-05:30")


# ---------------------------------------------------------------------------
# String parsing — ISO + tolerant variants
# ---------------------------------------------------------------------------


class TestIsoParse:

    def test_iso_with_colon_less_offset(self) -> None:
        v = any_to_datetime("2020-01-02T03:04:05+0200")

        assert v.utcoffset() == dt.timedelta(hours=2)

    def test_iso_with_space_separator(self) -> None:
        v = any_to_datetime("2020-01-02 03:04:05+02:00")

        assert v.utcoffset() == dt.timedelta(hours=2)

    def test_iso_minutes_only(self) -> None:
        v = any_to_datetime("2020-01-02 03:04+02:00")

        assert (v.hour, v.minute) == (3, 4)
        assert v.utcoffset() == dt.timedelta(hours=2)

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            any_to_datetime("not-a-date-at-all")

    def test_naive_datetime_object_passes_through_naive(self) -> None:
        v = any_to_datetime(dt.datetime(2020, 1, 2, 3, 4, 5))

        assert v.tzinfo is None


class TestSlashAndCompactDates:

    def test_slash_date(self) -> None:
        v = any_to_datetime("2020/01/02")

        assert v.date() == dt.date(2020, 1, 2)
        assert v.tzinfo is CURRENT_TZINFO

    def test_compact_date_only(self) -> None:
        v = any_to_datetime("20240131")

        assert v.date() == dt.date(2024, 1, 31)
        assert v.tzinfo is CURRENT_TZINFO

    def test_compact_with_t_separator(self) -> None:
        v = any_to_datetime("20240131T235959")

        assert (v.year, v.month, v.day, v.hour, v.minute, v.second) == (
            2024, 1, 31, 23, 59, 59,
        )
        assert v.tzinfo is CURRENT_TZINFO

    def test_compact_no_separator(self) -> None:
        v = any_to_datetime("20240131235959")

        assert (v.year, v.month, v.day, v.hour, v.minute, v.second) == (
            2024, 1, 31, 23, 59, 59,
        )
        assert v.tzinfo is CURRENT_TZINFO

    def test_compact_with_fraction_and_z(self) -> None:
        v = any_to_datetime("20240131T235959.1Z")

        assert v.microsecond == 100000
        assert v.utcoffset() == dt.timedelta(0)

    def test_compact_with_fraction_and_offset(self) -> None:
        v = any_to_datetime("20240131T235959.123456+0130")

        assert v.microsecond == 123456
        assert v.utcoffset() == dt.timedelta(hours=1, minutes=30)


# ---------------------------------------------------------------------------
# Numeric epoch inference (seconds vs ms vs us)
# ---------------------------------------------------------------------------


class TestIntDatetime:

    def test_seconds_epoch(self) -> None:
        assert int_to_datetime(1_700_000_000) == dt.datetime(
            2023, 11, 14, 22, 13, 20, tzinfo=dt.timezone.utc
        )

    def test_milliseconds_epoch(self) -> None:
        assert int_to_datetime(1_700_000_000_123) == dt.datetime(
            2023, 11, 14, 22, 13, 20, 123000, tzinfo=dt.timezone.utc
        )

    def test_microseconds_epoch(self) -> None:
        assert int_to_datetime(1_700_000_000_123_456) == dt.datetime(
            2023, 11, 14, 22, 13, 20, 123456, tzinfo=dt.timezone.utc
        )

    def test_negative_small_value_inferred_as_seconds(self) -> None:
        assert int_to_datetime(-1_000) == dt.datetime(
            1969, 12, 31, 23, 43, 20, tzinfo=dt.timezone.utc
        )


class TestFloatDatetime:

    def test_seconds_epoch_with_fraction(self) -> None:
        assert float_to_datetime(1_700_000_000.123456) == dt.datetime(
            2023, 11, 14, 22, 13, 20, 123456, tzinfo=dt.timezone.utc
        )

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_non_finite_raises(self, value: float) -> None:
        with pytest.raises(ValueError):
            float_to_datetime(value)


class TestIntDate:

    def test_seconds(self) -> None:
        assert int_to_date(1_700_000_000) == dt.date(2023, 11, 14)

    def test_milliseconds(self) -> None:
        assert int_to_date(1_700_000_000_123) == dt.date(2023, 11, 14)

    def test_microseconds(self) -> None:
        assert int_to_date(1_700_000_000_123_456) == dt.date(2023, 11, 14)


class TestFloatDate:

    def test_seconds_with_fraction(self) -> None:
        assert float_to_date(1_700_000_000.123456) == dt.date(2023, 11, 14)

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_non_finite_raises(self, value: float) -> None:
        with pytest.raises(ValueError):
            float_to_date(value)


class TestNumericInference:

    def test_any_to_datetime_routes_int_through_inference(self) -> None:
        assert any_to_datetime(1_700_000_000_123) == dt.datetime(
            2023, 11, 14, 22, 13, 20, 123000, tzinfo=dt.timezone.utc
        )

    def test_any_to_date_routes_int_through_inference(self) -> None:
        assert any_to_date(1_700_000_000_123_456) == dt.date(2023, 11, 14)

    def test_inference_holds_around_current_epoch(self) -> None:
        now_s = int(time.time())

        sec = int_to_datetime(now_s)
        ms = int_to_datetime(now_s * 1_000)
        us = int_to_datetime(now_s * 1_000_000)

        assert abs(sec.timestamp() - now_s) < 1
        assert abs(ms.timestamp() - now_s) < 1
        assert abs(us.timestamp() - now_s) < 1


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncate:

    def test_with_iso_interval_string(self) -> None:
        out = truncate_datetime_value(
            "2024-01-31T23:59:59.123456+01:00", "PT15M"
        )

        assert out == dt.datetime(
            2024, 1, 31, 23, 45, tzinfo=dt.timezone(dt.timedelta(hours=1))
        )

    def test_with_interval_spec_instance(self) -> None:
        spec = _coerce_interval("P1M")

        out = truncate_datetime_value(
            dt.datetime(2024, 5, 17, 14, 37, 59, tzinfo=dt.timezone.utc),
            spec,
        )

        assert out == dt.datetime(
            2024, 5, 1, 0, 0, tzinfo=dt.timezone.utc
        )


# ---------------------------------------------------------------------------
# Timedelta parsing
# ---------------------------------------------------------------------------


class TestStrToTimedelta:

    @pytest.mark.parametrize(
        "td",
        [
            dt.timedelta(days=1),
            dt.timedelta(days=2),
            dt.timedelta(days=-1),
            dt.timedelta(seconds=30),
            dt.timedelta(days=1, hours=1, minutes=2, seconds=3),
            dt.timedelta(days=1, microseconds=500_000),
            dt.timedelta(seconds=-30),
            dt.timedelta(days=1, hours=10, minutes=20, seconds=30, microseconds=123_456),
        ],
    )
    def test_python_str_roundtrip(self, td: dt.timedelta) -> None:
        assert str_to_timedelta(str(td)) == td

    def test_compact_days_shorthand(self) -> None:
        assert str_to_timedelta("1d 0:00:00") == dt.timedelta(days=1)
        assert str_to_timedelta("-2d 1:00") == dt.timedelta(days=-2, hours=1)

    def test_unit_form(self) -> None:
        assert str_to_timedelta("15m") == dt.timedelta(minutes=15)
        assert str_to_timedelta("2.5h") == dt.timedelta(hours=2, minutes=30)

    def test_iso_duration(self) -> None:
        assert str_to_timedelta("PT15M") == dt.timedelta(minutes=15)

    def test_float_fallback(self) -> None:
        assert str_to_timedelta("3.5") == dt.timedelta(seconds=3, microseconds=500_000)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse timedelta"):
            str_to_timedelta("not a duration")
