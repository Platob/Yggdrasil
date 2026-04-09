from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar, Generic, Mapping

from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "PrimitiveSerialized",
    "NoneSerialized",
    "BoolSerialized",
    "Utf8StringSerialized",
    "Latin1StringSerialized",
    "UInt8Serialized",
    "Int8Serialized",
    "UInt16Serialized",
    "Int16Serialized",
    "UInt32Serialized",
    "Int32Serialized",
    "UInt64Serialized",
    "Int64Serialized",
    "Float16Serialized",
    "Float32Serialized",
    "Float64Serialized",
]


def _unpack_int(fmt: str, data: bytes, *, tag_name: str) -> int:
    expected = struct.calcsize(fmt)
    actual = len(data)
    if actual != expected:
        raise ValueError(
            f"{tag_name} payload must be exactly {expected} bytes, got {actual}"
        )
    return int(struct.unpack(fmt, data)[0])


def _unpack_float(fmt: str, data: bytes, *, tag_name: str) -> float:
    expected = struct.calcsize(fmt)
    actual = len(data)
    if actual != expected:
        raise ValueError(
            f"{tag_name} payload must be exactly {expected} bytes, got {actual}"
        )
    return float(struct.unpack(fmt, data)[0])


@dataclass(frozen=True, slots=True)
class PrimitiveSerialized(Serialized[T], Generic[T]):
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
        if obj is None:
            return Serialized.build(tag=Tags.NONE, data=b"", metadata=metadata, codec=codec)

        if isinstance(obj, bool):
            return Serialized.build(
                tag=Tags.BOOL,
                data=b"\x01" if obj else b"\x00",
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, memoryview):
            obj = obj.tobytes()

        if isinstance(obj, bytearray):
            obj = bytes(obj)

        if isinstance(obj, bytes):
            return Serialized.build(tag=Tags.BYTES, data=obj, metadata=metadata, codec=codec)

        if isinstance(obj, str):
            return Serialized.build(
                tag=Tags.UTF8_STRING,
                data=obj.encode("utf-8"),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, int) and not isinstance(obj, bool):
            if obj >= 0:
                if obj <= 0xFF:
                    return Serialized.build(tag=Tags.UINT8, data=obj.to_bytes(1, "big"), metadata=metadata, codec=codec)
                if obj <= 0xFFFF:
                    return Serialized.build(tag=Tags.UINT16, data=obj.to_bytes(2, "big"), metadata=metadata, codec=codec)
                if obj <= 0xFFFFFFFF:
                    return Serialized.build(tag=Tags.UINT32, data=obj.to_bytes(4, "big"), metadata=metadata, codec=codec)
                if obj <= 0xFFFFFFFFFFFFFFFF:
                    return Serialized.build(tag=Tags.UINT64, data=obj.to_bytes(8, "big"), metadata=metadata, codec=codec)

            if -0x80 <= obj <= 0x7F:
                return Serialized.build(tag=Tags.INT8, data=obj.to_bytes(1, "big", signed=True), metadata=metadata, codec=codec)
            if -0x8000 <= obj <= 0x7FFF:
                return Serialized.build(tag=Tags.INT16, data=obj.to_bytes(2, "big", signed=True), metadata=metadata, codec=codec)
            if -0x80000000 <= obj <= 0x7FFFFFFF:
                return Serialized.build(tag=Tags.INT32, data=obj.to_bytes(4, "big", signed=True), metadata=metadata, codec=codec)
            if -0x8000000000000000 <= obj <= 0x7FFFFFFFFFFFFFFF:
                return Serialized.build(tag=Tags.INT64, data=obj.to_bytes(8, "big", signed=True), metadata=metadata, codec=codec)

            raise OverflowError("Integer does not fit supported fixed-width primitive tags")

        if isinstance(obj, float):
            return Serialized.build(
                tag=Tags.FLOAT64,
                data=struct.pack(">d", obj),
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class NoneSerialized(PrimitiveSerialized[None]):
    TAG: ClassVar[int] = Tags.NONE

    @property
    def value(self) -> None:
        data = self.decode()
        if data != b"":
            raise ValueError(f"NONE payload must be empty, got {len(data)} bytes")
        return None


@dataclass(frozen=True, slots=True)
class BoolSerialized(PrimitiveSerialized[bool]):
    TAG: ClassVar[int] = Tags.BOOL

    @property
    def value(self) -> bool:
        data = self.decode()
        if data == b"\x00":
            return False
        if data == b"\x01":
            return True
        raise ValueError(
            f"BOOL payload must be exactly b'\\x00' or b'\\x01', got {data!r}"
        )


@dataclass(frozen=True, slots=True)
class Utf8StringSerialized(PrimitiveSerialized[str]):
    TAG: ClassVar[int] = Tags.UTF8_STRING

    @property
    def value(self) -> str:
        return self.decode().decode("utf-8")


@dataclass(frozen=True, slots=True)
class Latin1StringSerialized(PrimitiveSerialized[str]):
    TAG: ClassVar[int] = Tags.LATIN1_STRING

    @property
    def value(self) -> str:
        return self.decode().decode("latin-1")


@dataclass(frozen=True, slots=True)
class UInt8Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.UINT8

    @property
    def value(self) -> int:
        return _unpack_int(">B", self.decode(), tag_name="UINT8")


@dataclass(frozen=True, slots=True)
class Int8Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.INT8

    @property
    def value(self) -> int:
        return _unpack_int(">b", self.decode(), tag_name="INT8")


@dataclass(frozen=True, slots=True)
class UInt16Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.UINT16

    @property
    def value(self) -> int:
        return _unpack_int(">H", self.decode(), tag_name="UINT16")


@dataclass(frozen=True, slots=True)
class Int16Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.INT16

    @property
    def value(self) -> int:
        return _unpack_int(">h", self.decode(), tag_name="INT16")


@dataclass(frozen=True, slots=True)
class UInt32Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.UINT32

    @property
    def value(self) -> int:
        return _unpack_int(">I", self.decode(), tag_name="UINT32")


@dataclass(frozen=True, slots=True)
class Int32Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.INT32

    @property
    def value(self) -> int:
        return _unpack_int(">i", self.decode(), tag_name="INT32")


@dataclass(frozen=True, slots=True)
class UInt64Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.UINT64

    @property
    def value(self) -> int:
        return _unpack_int(">Q", self.decode(), tag_name="UINT64")


@dataclass(frozen=True, slots=True)
class Int64Serialized(PrimitiveSerialized[int]):
    TAG: ClassVar[int] = Tags.INT64

    @property
    def value(self) -> int:
        return _unpack_int(">q", self.decode(), tag_name="INT64")


@dataclass(frozen=True, slots=True)
class Float16Serialized(PrimitiveSerialized[float]):
    TAG: ClassVar[int] = Tags.FLOAT16

    @property
    def value(self) -> float:
        return _unpack_float(">e", self.decode(), tag_name="FLOAT16")


@dataclass(frozen=True, slots=True)
class Float32Serialized(PrimitiveSerialized[float]):
    TAG: ClassVar[int] = Tags.FLOAT32

    @property
    def value(self) -> float:
        return _unpack_float(">f", self.decode(), tag_name="FLOAT32")


@dataclass(frozen=True, slots=True)
class Float64Serialized(PrimitiveSerialized[float]):
    TAG: ClassVar[int] = Tags.FLOAT64

    @property
    def value(self) -> float:
        return _unpack_float(">d", self.decode(), tag_name="FLOAT64")

for cls in PrimitiveSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)

for t, cls in (
    (type(None), NoneSerialized),
    (bool, BoolSerialized),
    (str, Utf8StringSerialized),
    (int, Int64Serialized),
    (float, Float64Serialized),
):
    Tags.register_class(cls, pytype=t)

NoneSerialized = Tags.get_class(Tags.NONE) or NoneSerialized
BoolSerialized = Tags.get_class(Tags.BOOL) or BoolSerialized
Utf8StringSerialized = Tags.get_class(Tags.UTF8_STRING) or Utf8StringSerialized
Latin1StringSerialized = Tags.get_class(Tags.LATIN1_STRING) or Latin1StringSerialized
UInt8Serialized = Tags.get_class(Tags.UINT8) or UInt8Serialized
Int8Serialized = Tags.get_class(Tags.INT8) or Int8Serialized
UInt16Serialized = Tags.get_class(Tags.UINT16) or UInt16Serialized
Int16Serialized = Tags.get_class(Tags.INT16) or Int16Serialized
UInt32Serialized = Tags.get_class(Tags.UINT32) or UInt32Serialized
Int32Serialized = Tags.get_class(Tags.INT32) or Int32Serialized
UInt64Serialized = Tags.get_class(Tags.UINT64) or UInt64Serialized
Int64Serialized = Tags.get_class(Tags.INT64) or Int64Serialized
Float16Serialized = Tags.get_class(Tags.FLOAT16) or Float16Serialized
Float32Serialized = Tags.get_class(Tags.FLOAT32) or Float32Serialized
Float64Serialized = Tags.get_class(Tags.FLOAT64) or Float64Serialized

for t, cls in (
    (type(None), NoneSerialized),
    (bool, BoolSerialized),
    (str, Utf8StringSerialized),
    (int, Int64Serialized),
    (float, Float64Serialized),
):
    Tags.TYPES[t] = cls

