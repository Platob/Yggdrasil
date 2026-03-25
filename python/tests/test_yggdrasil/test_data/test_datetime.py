# tests/test_cast_datetime_strings.py
from __future__ import annotations

import datetime as dt
import math
import time

import pytest

from yggdrasil.data.cast.datetime import (
    CURRENT_TZINFO,
    any_to_date,
    any_to_datetime,
    float_to_date,
    float_to_datetime,
    int_to_date,
    int_to_datetime,
    normalize_datetime_string,
)


def test_normalize_offsets_no_colon() -> None:
    assert normalize_datetime_string("2020-01-01T00:00:00+0200").endswith("+02:00")
    assert normalize_datetime_string("2020-01-01 00:00:00-0530").endswith("-05:30")


def test_parse_iso_with_offset_no_colon() -> None:
    v = any_to_datetime("2020-01-02T03:04:05+0200")
    assert v.utcoffset() == dt.timedelta(hours=2)


def test_parse_iso_with_space_and_offset() -> None:
    v = any_to_datetime("2020-01-02 03:04:05+02:00")
    assert v.utcoffset() == dt.timedelta(hours=2)


def test_parse_minutes_only() -> None:
    v = any_to_datetime("2020-01-02 03:04+02:00")
    assert (v.hour, v.minute) == (3, 4)
    assert v.utcoffset() == dt.timedelta(hours=2)


def test_parse_slash_date() -> None:
    v = any_to_datetime("2020/01/02")
    assert v.date() == dt.date(2020, 1, 2)
    assert v.tzinfo is CURRENT_TZINFO  # string parsing defaults tz


def test_parse_compact_date_only() -> None:
    v = any_to_datetime("20240131")
    assert v.date() == dt.date(2024, 1, 31)
    assert v.tzinfo is CURRENT_TZINFO


def test_parse_compact_datetime_T() -> None:
    v = any_to_datetime("20240131T235959")
    assert (v.year, v.month, v.day, v.hour, v.minute, v.second) == (2024, 1, 31, 23, 59, 59)
    assert v.tzinfo is CURRENT_TZINFO


def test_parse_compact_datetime_space() -> None:
    v = any_to_datetime("20240131235959")
    assert (v.year, v.month, v.day, v.hour, v.minute, v.second) == (2024, 1, 31, 23, 59, 59)
    assert v.tzinfo is CURRENT_TZINFO


def test_parse_compact_with_fraction_and_z() -> None:
    v = any_to_datetime("20240131T235959.1Z")
    assert v.microsecond == 100000
    assert v.utcoffset() == dt.timedelta(0)


def test_parse_compact_with_fraction_and_offset_no_colon() -> None:
    v = any_to_datetime("20240131T235959.123456+0130")
    assert v.microsecond == 123456
    assert v.utcoffset() == dt.timedelta(hours=1, minutes=30)


def test_naive_datetime_object_not_defaulted() -> None:
    v = any_to_datetime(dt.datetime(2020, 1, 2, 3, 4, 5))
    assert v.tzinfo is None


def test_garbage_raises() -> None:
    with pytest.raises(ValueError):
        any_to_datetime("not-a-date-at-all")


def test_int_datetime_seconds_epoch() -> None:
    v = int_to_datetime(1_700_000_000)
    assert v == dt.datetime(2023, 11, 14, 22, 13, 20, tzinfo=dt.timezone.utc)


def test_int_datetime_milliseconds_epoch() -> None:
    v = int_to_datetime(1_700_000_000_123)
    assert v == dt.datetime(2023, 11, 14, 22, 13, 20, 123000, tzinfo=dt.timezone.utc)


def test_int_datetime_microseconds_epoch() -> None:
    v = int_to_datetime(1_700_000_000_123_456)
    assert v == dt.datetime(2023, 11, 14, 22, 13, 20, 123456, tzinfo=dt.timezone.utc)


def test_float_datetime_seconds_epoch() -> None:
    v = float_to_datetime(1_700_000_000.123456)
    assert v == dt.datetime(2023, 11, 14, 22, 13, 20, 123456, tzinfo=dt.timezone.utc)


def test_negative_int_datetime_small_value_is_inferred_as_seconds() -> None:
    v = int_to_datetime(-1_000)
    assert v == dt.datetime(1969, 12, 31, 23, 43, 20, tzinfo=dt.timezone.utc)


def test_int_date_seconds_epoch() -> None:
    v = int_to_date(1_700_000_000)
    assert v == dt.date(2023, 11, 14)


def test_int_date_milliseconds_epoch() -> None:
    v = int_to_date(1_700_000_000_123)
    assert v == dt.date(2023, 11, 14)


def test_int_date_microseconds_epoch() -> None:
    v = int_to_date(1_700_000_000_123_456)
    assert v == dt.date(2023, 11, 14)


def test_float_date_seconds_epoch() -> None:
    v = float_to_date(1_700_000_000.123456)
    assert v == dt.date(2023, 11, 14)


def test_any_to_datetime_int_uses_numeric_inference() -> None:
    v = any_to_datetime(1_700_000_000_123)
    assert v == dt.datetime(2023, 11, 14, 22, 13, 20, 123000, tzinfo=dt.timezone.utc)


def test_any_to_date_int_uses_numeric_inference() -> None:
    v = any_to_date(1_700_000_000_123_456)
    assert v == dt.date(2023, 11, 14)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_float_datetime_non_finite_raises(value: float) -> None:
    with pytest.raises(ValueError):
        float_to_datetime(value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_float_date_non_finite_raises(value: float) -> None:
    with pytest.raises(ValueError):
        float_to_date(value)


def test_numeric_inference_near_current_epoch_scales() -> None:
    now_s = int(time.time())

    sec = int_to_datetime(now_s)
    ms = int_to_datetime(now_s * 1_000)
    us = int_to_datetime(now_s * 1_000_000)

    assert abs(sec.timestamp() - now_s) < 1
    assert abs(ms.timestamp() - now_s) < 1
    assert abs(us.timestamp() - now_s) < 1