from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import ClassVar, Generic, Mapping

from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

__all__ = [
    "LogicalSerialized",
    "DecimalSerialized",
    "DatetimeSerialized",
    "DateSerialized",
    "TimeSerialized",
]

_EPOCH_DATE = date(1970, 1, 1)
_EPOCH_DATETIME = datetime(1970, 1, 1, tzinfo=UTC)
_DAY_MICROS = 86_400 * 1_000_000


def _metadata_bytes(
    metadata: dict[bytes, bytes] | None,
    key: bytes,
) -> bytes | None:
    if not metadata:
        return None
    return metadata.get(key)


def _metadata_text(
    metadata: dict[bytes, bytes] | None,
    key: bytes,
    default: str | None = None,
) -> str | None:
    raw = _metadata_bytes(metadata, key)
    if raw is None:
        return default
    return raw.decode("utf-8")


def _metadata_int(
    metadata: dict[bytes, bytes] | None,
    key: bytes,
    default: int | None = None,
) -> int | None:
    raw = _metadata_bytes(metadata, key)
    if raw is None:
        return default
    return int(raw.decode("ascii"))


def _unpack_i32(data: bytes, *, tag_name: str) -> int:
    if len(data) != 4:
        raise ValueError(f"{tag_name} payload must be exactly 4 bytes, got {len(data)}")
    return int(struct.unpack(">i", data)[0])


def _unpack_i64(data: bytes, *, tag_name: str) -> int:
    if len(data) != 8:
        raise ValueError(f"{tag_name} payload must be exactly 8 bytes, got {len(data)}")
    return int(struct.unpack(">q", data)[0])


def _unpack_u64(data: bytes, *, tag_name: str) -> int:
    if len(data) != 8:
        raise ValueError(f"{tag_name} payload must be exactly 8 bytes, got {len(data)}")
    return int(struct.unpack(">Q", data)[0])


def _load_tzinfo(metadata: dict[bytes, bytes] | None) -> tzinfo | None:
    tz_name = _metadata_text(metadata, b"tz")
    if not tz_name:
        return None

    if tz_name == "UTC":
        return UTC

    if tz_name.startswith(("+", "-")):
        sign = 1 if tz_name[0] == "+" else -1
        hh, mm = tz_name[1:].split(":", 1)
        delta = timedelta(hours=int(hh), minutes=int(mm))
        return timezone(sign * delta)

    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass

    raise ValueError(f"Invalid or unsupported timezone metadata: {tz_name!r}")


def _datetime_from_epoch(value: int, unit: str) -> datetime:
    if unit == "s":
        return datetime.fromtimestamp(value, tz=UTC)
    if unit == "ms":
        return _EPOCH_DATETIME + timedelta(milliseconds=value)
    if unit == "us":
        return _EPOCH_DATETIME + timedelta(microseconds=value)
    if unit == "ns":
        micros, _ = divmod(value, 1_000)
        return _EPOCH_DATETIME + timedelta(microseconds=micros)
    raise ValueError(f"Unsupported DATETIME unit: {unit!r}")


def _time_from_offset(value: int, unit: str) -> time:
    if value < 0:
        raise ValueError("TIME payload must be >= 0")

    if unit == "s":
        total_micros = value * 1_000_000
    elif unit == "ms":
        total_micros = value * 1_000
    elif unit == "us":
        total_micros = value
    elif unit == "ns":
        total_micros = value // 1_000
    else:
        raise ValueError(f"Unsupported TIME unit: {unit!r}")

    if total_micros >= _DAY_MICROS:
        raise ValueError("TIME payload must be less than one day")

    hour, rem = divmod(total_micros, 3_600_000_000)
    minute, rem = divmod(rem, 60_000_000)
    second, microsecond = divmod(rem, 1_000_000)

    return time(
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        microsecond=int(microsecond),
    )


@dataclass(frozen=True, slots=True)
class LogicalSerialized(Serialized[T], Generic[T]):
    """Base class for logical/semantic payloads."""

    TAG: ClassVar[int]

    @property
    def value(self) -> T:
        raise NotImplementedError

    def as_python(self) -> T:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, Decimal):
            sign, digits, exponent = obj.as_tuple()
            coefficient = int("".join(map(str, digits))) if digits else 0
            if sign:
                coefficient = -coefficient
            scale = -exponent

            merged = dict(metadata or {})
            merged[b"scale"] = str(scale).encode("ascii")

            if not (-0x8000000000000000 <= coefficient <= 0x7FFFFFFFFFFFFFFF):
                raise OverflowError("Decimal coefficient does not fit int64 payload")

            return Serialized.build(
                tag=Tags.DECIMAL,
                data=coefficient.to_bytes(8, "big", signed=True),
                metadata=merged,
                codec=codec,
            )

        if isinstance(obj, datetime):
            merged = dict(metadata or {})

            if obj.tzinfo is None:
                dt_utc = obj.replace(tzinfo=UTC)
            else:
                dt_utc = obj.astimezone(UTC)
                tzname = obj.tzinfo.tzname(obj)
                if tzname:
                    merged.setdefault(b"tz", tzname.encode("utf-8"))

            epoch_us = int(dt_utc.timestamp() * 1_000_000)
            merged.setdefault(b"unit", b"us")

            return Serialized.build(
                tag=Tags.DATETIME,
                data=epoch_us.to_bytes(8, "big", signed=True),
                metadata=merged,
                codec=codec,
            )

        if isinstance(obj, date) and not isinstance(obj, datetime):
            epoch = date(1970, 1, 1)
            days = (obj - epoch).days
            return Serialized.build(
                tag=Tags.DATE,
                data=days.to_bytes(4, "big", signed=True),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, time):
            merged = dict(metadata or {})
            total_us = (
                ((obj.hour * 60 + obj.minute) * 60 + obj.second) * 1_000_000
                + obj.microsecond
            )
            if obj.tzinfo is not None:
                tzname = obj.tzinfo.tzname(None)
                if tzname:
                    merged.setdefault(b"tz", tzname.encode("utf-8"))
            merged.setdefault(b"unit", b"us")

            return Serialized.build(
                tag=Tags.TIME,
                data=total_us.to_bytes(8, "big", signed=False),
                metadata=merged,
                codec=codec,
            )

        return None

@dataclass(frozen=True, slots=True)
class DecimalSerialized(LogicalSerialized[Decimal]):
    """
    Decimal encoded as signed int64 coefficient + scale metadata.

    Payload
    -------
    big-endian int64 coefficient

    Metadata
    --------
    b"scale" -> ASCII integer, required
    b"precision" -> ASCII integer, optional
    """

    TAG: ClassVar[int] = Tags.DECIMAL

    @property
    def value(self) -> Decimal:
        scale = _metadata_int(self.metadata, b"scale")
        if scale is None:
            raise ValueError("DECIMAL metadata must include b'scale'")
        coefficient = _unpack_i64(self.decode(), tag_name="DECIMAL")
        return Decimal(coefficient).scaleb(-scale)


@dataclass(frozen=True, slots=True)
class DatetimeSerialized(LogicalSerialized[datetime]):
    """
    Datetime encoded as epoch integer plus optional timezone metadata.

    Payload
    -------
    big-endian int64 epoch timestamp

    Metadata
    --------
    b"unit" -> b"s" | b"ms" | b"us" | b"ns" (default b"us")
    b"tz"   -> optional timezone name or offset, e.g.
               b"UTC", b"Europe/Paris", b"+01:00"

    Semantics
    ---------
    Payload stores the absolute instant. If timezone metadata is present, the
    decoded datetime is converted to that timezone for presentation.
    """

    TAG: ClassVar[int] = Tags.DATETIME

    @property
    def value(self) -> datetime:
        unit = _metadata_text(self.metadata, b"unit", "us") or "us"
        epoch_value = _unpack_i64(self.decode(), tag_name="DATETIME")
        dt = _datetime_from_epoch(epoch_value, unit)

        tz = _load_tzinfo(self.metadata)
        if tz is not None:
            dt = dt.astimezone(tz)

        return dt


@dataclass(frozen=True, slots=True)
class DateSerialized(LogicalSerialized[date]):
    """
    Date encoded as signed int32 days since Unix epoch.
    """

    TAG: ClassVar[int] = Tags.DATE

    @property
    def value(self) -> date:
        days = _unpack_i32(self.decode(), tag_name="DATE")
        return _EPOCH_DATE + timedelta(days=days)


@dataclass(frozen=True, slots=True)
class TimeSerialized(LogicalSerialized[time]):
    """
    Time encoded as unsigned int64 offset since midnight.

    Payload
    -------
    big-endian uint64 offset from midnight

    Metadata
    --------
    b"unit" -> b"s" | b"ms" | b"us" | b"ns" (default b"us")
    b"tz"   -> optional timezone metadata, attached as ``tzinfo`` if given
    """

    TAG: ClassVar[int] = Tags.TIME

    @property
    def value(self) -> time:
        unit = _metadata_text(self.metadata, b"unit", "us") or "us"
        raw_value = _unpack_u64(self.decode(), tag_name="TIME")
        t = _time_from_offset(raw_value, unit)

        tz = _load_tzinfo(self.metadata)
        if tz is not None:
            t = t.replace(tzinfo=tz)

        return t


for cls in LogicalSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)