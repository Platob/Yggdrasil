from __future__ import annotations

import datetime as dt
import struct
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar

from yggdrasil.io import BytesIO

from .registry import REGISTRY
from .serialized import NestedScalar, PrimitiveSerialized, Serialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

__all__ = [
    "NoneSerialized",
    "BytesSerialized",
    "StringSerialized",
    "BoolSerialized",
    "IntSerialized",
    "FloatSerialized",
    "DateSerialized",
    "DateTimeSerialized",
    "DecimalSerialized",
    "UUIDSerialized",
]


def _merge_metadata(
    metadata: dict[bytes, bytes] | None,
    extra: dict[bytes, bytes] | None = None,
) -> dict[bytes, bytes]:
    out = {} if metadata is None else dict(metadata)
    if extra:
        out.update(extra)
    return out


def _tz_to_bytes(tzinfo: dt.tzinfo | None) -> bytes:
    if tzinfo is None:
        return b"naive"

    now = dt.datetime.now(tzinfo)
    offset = now.utcoffset()
    if offset is None:
        return b"naive"

    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hh, rem = divmod(total_seconds, 3600)
    mm = rem // 60
    return f"{sign}{hh:02d}:{mm:02d}".encode("ascii")


@dataclass(frozen=True, slots=True)
class NoneSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.NONE

    @property
    def value(self) -> None:
        return None

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "NoneSerialized":
        if value is not None:
            raise TypeError(f"{cls.__name__} only supports None")
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=BytesIO(),
            size=0,
            start_index=0,
        )


@dataclass(frozen=True, slots=True)
class BytesSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.BYTES

    @property
    def value(self) -> bytes:
        return self.payload()

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "BytesSerialized":
        if not isinstance(value, bytes):
            raise TypeError(f"{cls.__name__} only supports bytes")
        return cls.from_raw(value, metadata=metadata)


@dataclass(frozen=True, slots=True)
class StringSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.STRING

    @property
    def value(self) -> str:
        encoding = self.metadata.get(b"encoding", b"utf-8").decode("ascii")
        return self.payload().decode(encoding)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "StringSerialized":
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} only supports str")

        md = {} if metadata is None else dict(metadata)
        encoding = md.get(b"encoding", b"utf-8").decode("ascii")
        return cls.from_raw(value.encode(encoding), metadata=md)


@dataclass(frozen=True, slots=True)
class BoolSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.BOOL

    @property
    def value(self) -> bool:
        return self.payload() != b"\x00"

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "BoolSerialized":
        if not isinstance(value, bool):
            raise TypeError(f"{cls.__name__} only supports bool")
        return cls.from_raw(b"\x01" if value else b"\x00", metadata=metadata)


@dataclass(frozen=True, slots=True)
class IntSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.INT

    @property
    def value(self) -> int:
        return int.from_bytes(self.payload(), "big", signed=True)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "IntSerialized":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{cls.__name__} only supports int")

        if value == 0:
            raw = b"\x00"
        else:
            nbytes = max(1, (value.bit_length() + 8) // 8)
            raw = value.to_bytes(nbytes, "big", signed=True)

        return cls.from_raw(raw, metadata=metadata)


@dataclass(frozen=True, slots=True)
class FloatSerialized(PrimitiveSerialized):
    TAG: ClassVar[int] = SerdeTags.FLOAT

    @property
    def value(self) -> float:
        return struct.unpack(">d", self.payload())[0]

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "FloatSerialized":
        if not isinstance(value, float):
            raise TypeError(f"{cls.__name__} only supports float")
        return cls.from_raw(struct.pack(">d", value), metadata=metadata)


@dataclass(frozen=True, slots=True)
class DateSerialized(NestedScalar):
    TAG: ClassVar[int] = SerdeTags.DATE

    def _build_inner(self) -> Serialized:
        return IntSerialized(
            metadata=self.metadata,
            data=self.data,
            size=self.size,
            start_index=self.start_index,
        )

    @property
    def value(self) -> dt.date:
        micros = self.inner.value
        return dt.datetime.fromtimestamp(micros / 1_000_000, tz=dt.timezone.utc).date()

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "DateSerialized":
        if isinstance(value, dt.datetime) or not isinstance(value, dt.date):
            raise TypeError(f"{cls.__name__} only supports date")

        epoch_us = int(
            dt.datetime(
                value.year,
                value.month,
                value.day,
                tzinfo=dt.timezone.utc,
            ).timestamp() * 1_000_000
        )
        inner = IntSerialized.from_value(epoch_us)
        return cls(
            metadata=_merge_metadata(metadata, {b"tz": b"+00:00"}),
            data=inner.data,
            size=inner.size,
            start_index=inner.start_index,
        )


@dataclass(frozen=True, slots=True)
class DateTimeSerialized(NestedScalar):
    TAG: ClassVar[int] = SerdeTags.DATETIME

    def _build_inner(self) -> Serialized:
        return IntSerialized(
            metadata=self.metadata,
            data=self.data,
            size=self.size,
            start_index=self.start_index,
        )

    @property
    def value(self) -> dt.datetime:
        micros = self.inner.value
        return dt.datetime.fromtimestamp(micros / 1_000_000, tz=dt.timezone.utc)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "DateTimeSerialized":
        if not isinstance(value, dt.datetime):
            raise TypeError(f"{cls.__name__} only supports datetime")

        if value.tzinfo is None:
            value_utc = value.replace(tzinfo=dt.timezone.utc)
            extra = {b"tz": b"naive"}
        else:
            value_utc = value.astimezone(dt.timezone.utc)
            extra = {b"tz": _tz_to_bytes(value.tzinfo)}

        if value.fold:
            extra[b"fold"] = b"1"

        inner = IntSerialized.from_value(int(value_utc.timestamp() * 1_000_000))
        return cls(
            metadata=_merge_metadata(metadata, extra),
            data=inner.data,
            size=inner.size,
            start_index=inner.start_index,
        )


@dataclass(frozen=True, slots=True)
class DecimalSerialized(NestedScalar):
    TAG: ClassVar[int] = SerdeTags.DECIMAL

    def _build_inner(self) -> Serialized:
        return StringSerialized(
            metadata=self.metadata,
            data=self.data,
            size=self.size,
            start_index=self.start_index,
        )

    @property
    def value(self) -> Decimal:
        return Decimal(self.inner.value)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "DecimalSerialized":
        if not isinstance(value, Decimal):
            raise TypeError(f"{cls.__name__} only supports Decimal")

        inner = StringSerialized.from_value(format(value, "f"))
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=inner.data,
            size=inner.size,
            start_index=inner.start_index,
        )


@dataclass(frozen=True, slots=True)
class UUIDSerialized(NestedScalar):
    TAG: ClassVar[int] = SerdeTags.UUID

    def _build_inner(self) -> Serialized:
        return BytesSerialized(
            metadata=self.metadata,
            data=self.data,
            size=self.size,
            start_index=self.start_index,
        )

    @property
    def value(self) -> uuid.UUID:
        return uuid.UUID(bytes=self.inner.value)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "UUIDSerialized":
        if not isinstance(value, uuid.UUID):
            raise TypeError(f"{cls.__name__} only supports UUID")
        inner = BytesSerialized.from_value(value.bytes)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=inner.data,
            size=inner.size,
            start_index=inner.start_index,
        )


REGISTRY.register_python_type(type(None), NoneSerialized)
REGISTRY.register_python_type(bool, BoolSerialized)
REGISTRY.register_python_type(int, IntSerialized)
REGISTRY.register_python_type(float, FloatSerialized)
REGISTRY.register_python_type(bytes, BytesSerialized)
REGISTRY.register_python_type(str, StringSerialized)
REGISTRY.register_python_type(dt.datetime, DateTimeSerialized)
REGISTRY.register_python_type(dt.date, DateSerialized)
REGISTRY.register_python_type(Decimal, DecimalSerialized)
REGISTRY.register_python_type(uuid.UUID, UUIDSerialized)