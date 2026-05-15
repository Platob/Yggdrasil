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
* Additional converters — ``str_to_time``, ``str_to_timedelta``,
  ``str_to_tzinfo``, ``parse_http_date``, ``any_to_*``,
  ``datetime_to_*``, ``date_to_datetime``, ``time_to_datetime``.
* Interval helpers — ``iter_datetime_ranges``, ``truncate_datetime``,
  ``_coerce_interval`` LRU cache.
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
    any_to_time,
    any_to_timedelta,
    any_to_tzinfo,
    date_to_datetime,
    datetime_to_date,
    datetime_to_time,
    float_to_date,
    float_to_datetime,
    int_to_date,
    int_to_datetime,
    int_to_timedelta,
    float_to_timedelta,
    iter_datetime_ranges,
    normalize_datetime_string,
    normalize_fractional_seconds,
    parse_http_date,
    str_to_date,
    str_to_datetime,
    str_to_time,
    str_to_timedelta,
    str_to_tzinfo,
    time_to_datetime,
    timedelta_to_tzinfo,
    truncate_datetime,
    truncate_datetime_value,
    tzinfo_to_timedelta,
)

_UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# normalize_fractional_seconds
# ---------------------------------------------------------------------------


class TestNormalizeFractionalSeconds:

    def test_no_dot_returns_unchanged(self) -> None:
        assert normalize_fractional_seconds("2020-01-01T00:00:00+00:00") == "2020-01-01T00:00:00+00:00"

    def test_three_digit_fraction_padded_to_six(self) -> None:
        result = normalize_fractional_seconds("2020-01-01T00:00:00.123+00:00")
        assert result == "2020-01-01T00:00:00.123000+00:00"

    def test_six_digit_fraction_unchanged(self) -> None:
        assert normalize_fractional_seconds("2020-01-01T00:00:00.123456+00:00") == "2020-01-01T00:00:00.123456+00:00"

    def test_nine_digit_fraction_truncated_to_six(self) -> None:
        result = normalize_fractional_seconds("2020-01-01T00:00:00.123456789+00:00")
        assert result == "2020-01-01T00:00:00.123456+00:00"

    def test_fraction_at_end_no_offset(self) -> None:
        result = normalize_fractional_seconds("2020-01-01T00:00:00.1")
        assert result == "2020-01-01T00:00:00.100000"


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

    def test_z_suffix_replaced_with_utc_offset(self) -> None:
        result = normalize_datetime_string("2020-01-01T12:00:00Z")
        assert result.endswith("+00:00")

    def test_slash_date_converted_to_dashes(self) -> None:
        assert normalize_datetime_string("2020/01/01") == "2020-01-01"

    def test_compact_date_only_expanded(self) -> None:
        assert normalize_datetime_string("20240131") == "2024-01-31"

    def test_compact_datetime_with_t_separator(self) -> None:
        result = normalize_datetime_string("20240131T120000")
        assert result == "2024-01-31T12:00:00"

    def test_compact_datetime_with_fraction_and_z(self) -> None:
        result = normalize_datetime_string("20240131T120000.5Z")
        assert "+00:00" in result
        assert "500000" in result


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

    def test_utcnow_sentinel_returns_utc_aware(self) -> None:
        v = str_to_datetime("utcnow")
        assert v.tzinfo is not None

    def test_now_sentinel_returns_datetime(self) -> None:
        v = str_to_datetime("now")
        assert isinstance(v, dt.datetime)

    def test_z_suffix_parsed_as_utc(self) -> None:
        v = str_to_datetime("2020-06-15T10:00:00Z")
        assert v.utcoffset() == dt.timedelta(0)


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
# str_to_time
# ---------------------------------------------------------------------------


class TestStrToTime:

    def test_simple_hh_mm_ss(self) -> None:
        v = str_to_time("12:34:56")
        assert (v.hour, v.minute, v.second) == (12, 34, 56)

    def test_with_microseconds(self) -> None:
        v = str_to_time("12:34:56.123456")
        assert v.microsecond == 123456

    def test_with_utc_offset(self) -> None:
        v = str_to_time("12:34:56+02:00")
        assert v.utcoffset() == dt.timedelta(hours=2)

    def test_invalid_raises(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            str_to_time("not-a-time")


# ---------------------------------------------------------------------------
# str_to_timedelta
# ---------------------------------------------------------------------------


class TestStrToTimedelta:

    def test_hms_format(self) -> None:
        v = str_to_timedelta("1:30:00")
        assert v == dt.timedelta(hours=1, minutes=30)

    def test_hms_with_days(self) -> None:
        v = str_to_timedelta("2d 1:00:00")
        assert v == dt.timedelta(days=2, hours=1)

    def test_hms_with_fraction(self) -> None:
        v = str_to_timedelta("0:00:01.5")
        assert v == dt.timedelta(seconds=1, microseconds=500000)

    def test_unit_seconds(self) -> None:
        assert str_to_timedelta("30s") == dt.timedelta(seconds=30)

    def test_unit_minutes(self) -> None:
        assert str_to_timedelta("5m") == dt.timedelta(minutes=5)

    def test_unit_hours(self) -> None:
        assert str_to_timedelta("2h") == dt.timedelta(hours=2)

    def test_unit_days(self) -> None:
        assert str_to_timedelta("7d") == dt.timedelta(days=7)

    def test_unit_weeks(self) -> None:
        assert str_to_timedelta("2w") == dt.timedelta(weeks=2)

    def test_unit_case_insensitive(self) -> None:
        assert str_to_timedelta("10S") == dt.timedelta(seconds=10)

    def test_iso_duration_fixed(self) -> None:
        assert str_to_timedelta("PT1H30M") == dt.timedelta(hours=1, minutes=30)

    def test_iso_duration_days(self) -> None:
        assert str_to_timedelta("P2D") == dt.timedelta(days=2)

    def test_iso_duration_weeks(self) -> None:
        assert str_to_timedelta("P1W") == dt.timedelta(weeks=1)

    def test_iso_duration_calendar_raises(self) -> None:
        with pytest.raises(ValueError, match="calendar"):
            str_to_timedelta("P1M")

    def test_plain_float_seconds(self) -> None:
        assert str_to_timedelta("3600.0") == dt.timedelta(seconds=3600)

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError):
            str_to_timedelta("not-a-duration")


# ---------------------------------------------------------------------------
# str_to_tzinfo
# ---------------------------------------------------------------------------


class TestStrToTzinfo:

    def test_utc_lowercase(self) -> None:
        assert str_to_tzinfo("utc") is _UTC

    def test_utc_uppercase(self) -> None:
        assert str_to_tzinfo("UTC") is _UTC

    def test_z(self) -> None:
        assert str_to_tzinfo("Z") is _UTC

    def test_etc_utc_exact(self) -> None:
        # "Etc/UTC" is a valid IANA alias for UTC; the fast path normalises
        # to uppercase before set lookup so it catches any case variant.
        result = str_to_tzinfo("Etc/UTC")
        assert result is _UTC or result.utcoffset(dt.datetime.now()) == dt.timedelta(0)

    def test_etc_utc_uppercase(self) -> None:
        result = str_to_tzinfo("ETC/UTC")
        assert result is _UTC or result.utcoffset(dt.datetime.now()) == dt.timedelta(0)

    def test_positive_offset_with_colon(self) -> None:
        tz = str_to_tzinfo("+05:30")
        assert tz.utcoffset(None) == dt.timedelta(hours=5, minutes=30)  # type: ignore[arg-type]

    def test_negative_offset_with_colon(self) -> None:
        tz = str_to_tzinfo("-07:00")
        assert tz.utcoffset(None) == dt.timedelta(hours=-7)  # type: ignore[arg-type]

    def test_offset_without_colon(self) -> None:
        tz = str_to_tzinfo("+0530")
        assert tz.utcoffset(None) == dt.timedelta(hours=5, minutes=30)  # type: ignore[arg-type]

    def test_local_sentinel(self) -> None:
        assert str_to_tzinfo("local") is CURRENT_TZINFO
        assert str_to_tzinfo("CURRENT") is CURRENT_TZINFO
        assert str_to_tzinfo("now") is CURRENT_TZINFO

    def test_invalid_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            str_to_tzinfo("Not/A/Real/Zone/XXXX")


# ---------------------------------------------------------------------------
# timedelta ↔ tzinfo
# ---------------------------------------------------------------------------


class TestTimedeltaTzinfo:

    def test_timedelta_to_tzinfo_positive(self) -> None:
        tz = timedelta_to_tzinfo(dt.timedelta(hours=5, minutes=30))
        assert tz.utcoffset(None) == dt.timedelta(hours=5, minutes=30)  # type: ignore[arg-type]

    def test_timedelta_to_tzinfo_utc(self) -> None:
        tz = timedelta_to_tzinfo(dt.timedelta(0))
        assert tz.utcoffset(None) == dt.timedelta(0)  # type: ignore[arg-type]

    def test_timedelta_to_tzinfo_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="24h"):
            timedelta_to_tzinfo(dt.timedelta(hours=24))

    def test_tzinfo_to_timedelta_utc(self) -> None:
        assert tzinfo_to_timedelta(_UTC) == dt.timedelta(0)

    def test_tzinfo_to_timedelta_positive_offset(self) -> None:
        tz = dt.timezone(dt.timedelta(hours=3))
        assert tzinfo_to_timedelta(tz) == dt.timedelta(hours=3)


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
# int/float → timedelta
# ---------------------------------------------------------------------------


class TestIntFloatToTimedelta:

    def test_int_to_timedelta(self) -> None:
        assert int_to_timedelta(3600) == dt.timedelta(hours=1)

    def test_float_to_timedelta(self) -> None:
        assert float_to_timedelta(1.5) == dt.timedelta(seconds=1, microseconds=500000)


# ---------------------------------------------------------------------------
# datetime ↔ date / time cross-converters
# ---------------------------------------------------------------------------


class TestDatetimeCrossConverters:

    def test_datetime_to_date(self) -> None:
        v = dt.datetime(2024, 6, 15, 12, 30, tzinfo=_UTC)
        assert datetime_to_date(v) == dt.date(2024, 6, 15)

    def test_datetime_to_time_aware(self) -> None:
        v = dt.datetime(2024, 6, 15, 12, 30, 45, tzinfo=_UTC)
        t = datetime_to_time(v)
        assert (t.hour, t.minute, t.second) == (12, 30, 45)
        assert t.tzinfo is _UTC

    def test_datetime_to_time_naive(self) -> None:
        v = dt.datetime(2024, 6, 15, 12, 30, 45)
        t = datetime_to_time(v)
        assert t.tzinfo is None

    def test_date_to_datetime(self) -> None:
        v = date_to_datetime(dt.date(2024, 6, 15))
        assert v.date() == dt.date(2024, 6, 15)
        assert v.tzinfo is not None

    def test_time_to_datetime(self) -> None:
        v = time_to_datetime(dt.time(12, 30, 0, tzinfo=_UTC))
        assert (v.hour, v.minute) == (12, 30)
        assert v.tzinfo is _UTC

    def test_time_to_datetime_naive_uses_current_tzinfo(self) -> None:
        v = time_to_datetime(dt.time(9, 0, 0))
        assert isinstance(v, dt.datetime)


# ---------------------------------------------------------------------------
# any_to_* polymorphic dispatch
# ---------------------------------------------------------------------------


class TestAnyToDatetime:

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_datetime(None)

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            any_to_datetime(True)

    def test_datetime_passthrough(self) -> None:
        v = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        assert any_to_datetime(v) is v

    def test_date_converted(self) -> None:
        v = any_to_datetime(dt.date(2024, 6, 1))
        assert v.date() == dt.date(2024, 6, 1)

    def test_time_converted(self) -> None:
        v = any_to_datetime(dt.time(10, 0, tzinfo=_UTC))
        assert v.hour == 10

    def test_int_converted(self) -> None:
        assert any_to_datetime(1_700_000_000).year == 2023

    def test_float_converted(self) -> None:
        assert any_to_datetime(1_700_000_000.0).year == 2023

    def test_str_converted(self) -> None:
        assert any_to_datetime("2024-01-15T00:00:00Z").date() == dt.date(2024, 1, 15)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_datetime(object())


class TestAnyToDate:

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_date(None)

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            any_to_date(True)

    def test_date_passthrough(self) -> None:
        v = dt.date(2024, 1, 1)
        assert any_to_date(v) is v

    def test_datetime_strips_time(self) -> None:
        assert any_to_date(dt.datetime(2024, 6, 15, 12, 0, tzinfo=_UTC)) == dt.date(2024, 6, 15)

    def test_str_converted(self) -> None:
        assert any_to_date("2024-06-15") == dt.date(2024, 6, 15)

    def test_int_converted(self) -> None:
        assert any_to_date(1_700_000_000) == dt.date(2023, 11, 14)


class TestAnyToTime:

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_time(None)

    def test_time_passthrough(self) -> None:
        v = dt.time(12, 0)
        assert any_to_time(v) is v

    def test_datetime_extracts_time(self) -> None:
        t = any_to_time(dt.datetime(2024, 1, 1, 9, 30, tzinfo=_UTC))
        assert (t.hour, t.minute) == (9, 30)

    def test_str_converted(self) -> None:
        t = any_to_time("08:15:00")
        assert (t.hour, t.minute) == (8, 15)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_time(42)


class TestAnyToTimedelta:

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_timedelta(None)

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            any_to_timedelta(True)

    def test_timedelta_passthrough(self) -> None:
        v = dt.timedelta(hours=1)
        assert any_to_timedelta(v) is v

    def test_str_hms(self) -> None:
        assert any_to_timedelta("1:30:00") == dt.timedelta(hours=1, minutes=30)

    def test_int(self) -> None:
        assert any_to_timedelta(3600) == dt.timedelta(hours=1)

    def test_float(self) -> None:
        assert any_to_timedelta(90.0) == dt.timedelta(seconds=90)

    def test_tzinfo_converts_to_offset(self) -> None:
        tz = dt.timezone(dt.timedelta(hours=2))
        assert any_to_timedelta(tz) == dt.timedelta(hours=2)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_timedelta(object())


class TestAnyToTzinfo:

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_tzinfo(None)

    def test_bool_raises(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            any_to_tzinfo(True)

    def test_tzinfo_passthrough(self) -> None:
        assert any_to_tzinfo(_UTC) is _UTC

    def test_str_utc(self) -> None:
        assert any_to_tzinfo("UTC") is _UTC

    def test_timedelta_converts(self) -> None:
        tz = any_to_tzinfo(dt.timedelta(hours=3))
        assert tz.utcoffset(None) == dt.timedelta(hours=3)  # type: ignore[arg-type]

    def test_int_converts(self) -> None:
        tz = any_to_tzinfo(3600)
        assert isinstance(tz, dt.tzinfo)

    def test_float_converts(self) -> None:
        tz = any_to_tzinfo(0.0)
        assert isinstance(tz, dt.tzinfo)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TypeError):
            any_to_tzinfo(object())


# ---------------------------------------------------------------------------
# parse_http_date
# ---------------------------------------------------------------------------


class TestParseHttpDate:

    def test_rfc_1123_imf_fixdate(self) -> None:
        v = parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT")
        assert v is not None
        assert v.year == 1994
        assert v.month == 11
        assert v.day == 6
        assert v.tzinfo is not None

    def test_rfc_850_obsolete(self) -> None:
        v = parse_http_date("Sunday, 06-Nov-94 08:49:37 GMT")
        assert v is not None
        assert v.day == 6

    def test_asctime_format(self) -> None:
        v = parse_http_date("Sun Nov  6 08:49:37 1994")
        assert v is not None
        assert v.year == 1994
        assert v.tzinfo is not None

    def test_invalid_returns_none(self) -> None:
        assert parse_http_date("not a date") is None

    def test_asctime_invalid_month_returns_none(self) -> None:
        assert parse_http_date("Sun Xyz  6 08:49:37 1994") is None


# ---------------------------------------------------------------------------
# str_to_date
# ---------------------------------------------------------------------------


class TestStrToDate:

    def test_iso_date(self) -> None:
        assert str_to_date("2024-06-15") == dt.date(2024, 6, 15)

    def test_slash_date(self) -> None:
        assert str_to_date("2024/06/15") == dt.date(2024, 6, 15)

    def test_compact_date(self) -> None:
        assert str_to_date("20240615") == dt.date(2024, 6, 15)


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

    def test_truncate_datetime_add_interval_false(self) -> None:
        v = dt.datetime(2024, 3, 15, 10, 30, tzinfo=_UTC)
        out = truncate_datetime(v, "P1M")
        assert out == dt.datetime(2024, 3, 1, tzinfo=_UTC)

    def test_truncate_datetime_add_interval_true_advances(self) -> None:
        # value is not aligned, so add_interval=True returns next boundary
        v = dt.datetime(2024, 3, 15, 10, 30, tzinfo=_UTC)
        out = truncate_datetime(v, "P1M", add_interval=True)
        assert out == dt.datetime(2024, 4, 1, tzinfo=_UTC)

    def test_truncate_datetime_add_interval_true_already_aligned(self) -> None:
        # already aligned → no advancement
        v = dt.datetime(2024, 3, 1, 0, 0, tzinfo=_UTC)
        out = truncate_datetime(v, "P1M", add_interval=True)
        assert out == dt.datetime(2024, 3, 1, tzinfo=_UTC)

    def test_truncate_to_year(self) -> None:
        out = truncate_datetime_value(
            dt.datetime(2023, 8, 20, 14, tzinfo=_UTC), "P1Y"
        )
        assert out == dt.datetime(2023, 1, 1, tzinfo=_UTC)

    def test_truncate_fixed_seconds(self) -> None:
        out = truncate_datetime_value(
            dt.datetime(2024, 1, 1, 0, 0, 17, tzinfo=_UTC), "PT15S"
        )
        assert out == dt.datetime(2024, 1, 1, 0, 0, 15, tzinfo=_UTC)

    def test_truncate_naive_datetime(self) -> None:
        out = truncate_datetime_value(dt.datetime(2024, 1, 15, 10, 30), "P1D")
        assert out == dt.datetime(2024, 1, 15, 0, 0, 0)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            truncate_datetime_value(dt.datetime(2024, 1, 1, tzinfo=_UTC), "PT0S")


# ---------------------------------------------------------------------------
# iter_datetime_ranges
# ---------------------------------------------------------------------------


class TestIterDatetimeRanges:

    def test_daily_ranges(self) -> None:
        start = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        end = dt.datetime(2024, 1, 4, tzinfo=_UTC)
        ranges = list(iter_datetime_ranges(start, end, "P1D"))

        assert len(ranges) == 3
        assert ranges[0] == (dt.datetime(2024, 1, 1, tzinfo=_UTC), dt.datetime(2024, 1, 2, tzinfo=_UTC))
        assert ranges[-1] == (dt.datetime(2024, 1, 3, tzinfo=_UTC), dt.datetime(2024, 1, 4, tzinfo=_UTC))

    def test_monthly_ranges(self) -> None:
        start = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        end = dt.datetime(2024, 4, 1, tzinfo=_UTC)
        ranges = list(iter_datetime_ranges(start, end, "P1M"))

        assert len(ranges) == 3
        assert ranges[0][0] == dt.datetime(2024, 1, 1, tzinfo=_UTC)
        assert ranges[2][0] == dt.datetime(2024, 3, 1, tzinfo=_UTC)

    def test_15_minute_ranges(self) -> None:
        start = dt.datetime(2024, 1, 1, 0, 0, tzinfo=_UTC)
        end = dt.datetime(2024, 1, 1, 1, 0, tzinfo=_UTC)
        ranges = list(iter_datetime_ranges(start, end, "PT15M"))

        assert len(ranges) == 4

    def test_start_equals_end_yields_nothing(self) -> None:
        v = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        assert list(iter_datetime_ranges(v, v, "P1D")) == []

    def test_start_after_end_yields_nothing(self) -> None:
        start = dt.datetime(2024, 1, 5, tzinfo=_UTC)
        end = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        assert list(iter_datetime_ranges(start, end, "P1D")) == []

    def test_string_inputs_parsed(self) -> None:
        ranges = list(iter_datetime_ranges("2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z", "P1D"))
        assert len(ranges) == 2

    def test_interval_spec_cached(self) -> None:
        # _coerce_interval should be LRU-cached; calling iter_datetime_ranges
        # multiple times with the same string interval must not re-parse it.
        from yggdrasil.data.cast.datetime import _coerce_interval
        info_before = _coerce_interval.cache_info()

        start = dt.datetime(2024, 1, 1, tzinfo=_UTC)
        end = dt.datetime(2024, 1, 4, tzinfo=_UTC)

        list(iter_datetime_ranges(start, end, "P1D"))
        list(iter_datetime_ranges(start, end, "P1D"))

        info_after = _coerce_interval.cache_info()
        # Second call must have hit the cache at least once.
        assert info_after.hits > info_before.hits
