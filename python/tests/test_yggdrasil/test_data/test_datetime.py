# tests/test_cast_datetime_strings.py
from __future__ import annotations

import datetime as dt
import pytest

from yggdrasil.data.cast.datetime import (
    CURRENT_TZINFO,
    any_to_datetime,
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