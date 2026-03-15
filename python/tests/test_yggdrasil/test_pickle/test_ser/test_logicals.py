from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def test_decimal_serialized() -> None:
    coefficient = 12345
    ser = Serialized.build(
        tag=Tags.DECIMAL,
        data=coefficient.to_bytes(8, "big", signed=True),
        metadata={b"scale": b"2"},
    )

    assert ser.as_python() == Decimal("123.45")


def test_decimal_requires_scale_metadata() -> None:
    ser = Serialized.build(
        tag=Tags.DECIMAL,
        data=(123).to_bytes(8, "big", signed=True),
    )
    try:
        _ = ser.as_python()
    except ValueError as exc:
        assert "DECIMAL metadata must include b'scale'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_date_serialized() -> None:
    epoch = date(1970, 1, 1)
    d = date(2026, 3, 14)
    days = (d - epoch).days

    ser = Serialized.build(
        tag=Tags.DATE,
        data=days.to_bytes(4, "big", signed=True),
    )

    assert ser.as_python() == d


def test_datetime_serialized_default_unit_us_utc() -> None:
    dt = datetime(2026, 3, 14, 12, 30, 45, 123456, tzinfo=UTC)
    epoch_us = int((dt - datetime(1970, 1, 1, tzinfo=UTC)).total_seconds() * 1_000_000)

    ser = Serialized.build(
        tag=Tags.DATETIME,
        data=epoch_us.to_bytes(8, "big", signed=True),
        metadata={b"unit": b"us"},
    )

    assert ser.as_python() == dt


def test_datetime_serialized_with_tz_metadata() -> None:
    dt_utc = datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC)
    epoch_s = int(dt_utc.timestamp())

    ser = Serialized.build(
        tag=Tags.DATETIME,
        data=epoch_s.to_bytes(8, "big", signed=True),
        metadata={b"unit": b"s", b"tz": b"+01:00"},
    )

    got = ser.as_python()
    assert got.utcoffset() == timedelta(hours=1)
    assert got.astimezone(UTC) == dt_utc


def test_time_serialized_us() -> None:
    micros = ((12 * 3600) + (34 * 60) + 56) * 1_000_000 + 789
    ser = Serialized.build(
        tag=Tags.TIME,
        data=micros.to_bytes(8, "big", signed=False),
        metadata={b"unit": b"us"},
    )

    assert ser.as_python() == time(12, 34, 56, 789)


def test_time_serialized_rejects_more_than_day() -> None:
    micros = 86_400 * 1_000_000
    ser = Serialized.build(
        tag=Tags.TIME,
        data=micros.to_bytes(8, "big", signed=False),
        metadata={b"unit": b"us"},
    )

    try:
        _ = ser.as_python()
    except ValueError as exc:
        assert "less than one day" in str(exc)
    else:
        raise AssertionError("Expected ValueError")