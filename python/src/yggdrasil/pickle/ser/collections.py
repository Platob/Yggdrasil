from __future__ import annotations

from collections.abc import Generator as AbcGenerator
from collections.abc import Iterator as AbcIterator
from dataclasses import dataclass
from typing import ClassVar, Generic, Iterator, Mapping

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "CollectionSerialized",
    "LargeCollectionSerialized",
    "ArraySerialized",
    "LargeArraySerialized",
    "ListSerialized",
    "LargeListSerialized",
    "TupleSerialized",
    "LargeTupleSerialized",
    "SetSerialized",
    "LargeSetSerialized",
    "MappingSerialized",
    "LargeMappingSerialized",
    "GeneratorSerialized",
    "LargeGeneratorSerialized",
    "IteratorSerialized",
    "LargeIteratorSerialized",
]


def _read_u32(buffer: BytesIO) -> int:
    raw = buffer.read(4)
    if len(raw) != 4:
        raise ValueError(f"Expected 4 bytes for u32, got {len(raw)}")
    return int.from_bytes(raw, "big", signed=False)


def _read_u64(buffer: BytesIO) -> int:
    raw = buffer.read(8)
    if len(raw) != 8:
        raise ValueError(f"Expected 8 bytes for u64, got {len(raw)}")
    return int.from_bytes(raw, "big", signed=False)


def _iter_items(buffer: BytesIO, count: int) -> Iterator[Serialized[object]]:
    for _ in range(count):
        start = buffer.tell()
        item = Serialized.read_from(buffer, pos=start)
        buffer.seek(item.head.payload_end)
        yield item


def _write_count(buffer: BytesIO, count: int, *, large: bool) -> None:
    buffer.write(count.to_bytes(8 if large else 4, "big", signed=False))


def _build_collection_payload(
    items: Iterator[object],
    *,
    count: int,
    large: bool,
) -> BytesIO:
    payload = BytesIO()
    _write_count(payload, count, large=large)
    for item in items:
        Serialized.from_python_object(item).write_to(payload)
    return payload


def _build_mapping_payload(
    items: Iterator[tuple[object, object]],
    *,
    count: int,
    large: bool,
) -> BytesIO:
    payload = BytesIO()
    _write_count(payload, count, large=large)
    for key, value in items:
        Serialized.from_python_object(key).write_to(payload)
        Serialized.from_python_object(value).write_to(payload)
    return payload


def _materialize_iterable(obj: AbcIterator[object]) -> tuple[tuple[object, ...], int]:
    """
    Consume an iterator/generator into an immutable snapshot so we can
    emit the required count-prefixed wire format.
    """
    values = tuple(obj)
    return values, len(values)


@dataclass(frozen=True, slots=True)
class CollectionSerialized(Serialized[T], Generic[T]):
    """
    Base class for standard collection payloads.

    Standard collection wire format
    -------------------------------
    Ordered collections:
        [count:u32][item_0][item_1]...[item_n]

    Mapping collections:
        [count:u32][key_0][value_0]...[key_n][value_n]
    """

    TAG: ClassVar[int]

    def _payload_buffer(self) -> BytesIO:
        if self.codec == CODEC_NONE:
            return BytesIO(self.data.to_bytes())
        return BytesIO(self.decode())

    def _read_count(self, buffer: BytesIO) -> int:
        return _read_u32(buffer)

    def _iter_from_payload(self) -> Iterator[Serialized[object]]:
        buf = self._payload_buffer()
        count = self._read_count(buf)
        yield from _iter_items(buf, count)

    def iter_(self) -> Iterator[Serialized[object]]:
        yield from self._iter_from_payload()

    @property
    def items(self) -> tuple[Serialized[object], ...]:
        return tuple(self.iter_())

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
        if isinstance(obj, list):
            large = len(obj) > 0xFFFFFFFF
            tag = Tags.LARGE_LIST if large else Tags.LIST
            payload = _build_collection_payload(
                iter(obj),
                count=len(obj),
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, tuple):
            large = len(obj) > 0xFFFFFFFF
            tag = Tags.LARGE_TUPLE if large else Tags.TUPLE
            payload = _build_collection_payload(
                iter(obj),
                count=len(obj),
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, set):
            large = len(obj) > 0xFFFFFFFF
            tag = Tags.LARGE_SET if large else Tags.SET
            payload = _build_collection_payload(
                iter(obj),
                count=len(obj),
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, dict):
            large = len(obj) > 0xFFFFFFFF
            tag = Tags.LARGE_MAPPING if large else Tags.MAPPING
            payload = _build_mapping_payload(
                iter(obj.items()),
                count=len(obj),
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, AbcGenerator):
            values, count = _materialize_iterable(obj)
            large = count > 0xFFFFFFFF
            tag = Tags.LARGE_GENERATOR if large else Tags.GENERATOR
            payload = _build_collection_payload(
                iter(values),
                count=count,
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, AbcIterator):
            values, count = _materialize_iterable(obj)
            large = count > 0xFFFFFFFF
            tag = Tags.LARGE_ITERATOR if large else Tags.ITERATOR
            payload = _build_collection_payload(
                iter(values),
                count=count,
                large=large,
            )
            return Serialized.build(
                tag=tag,
                data=payload.to_bytes(),
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class LargeCollectionSerialized(CollectionSerialized[T], Generic[T]):
    def _read_count(self, buffer: BytesIO) -> int:
        return _read_u64(buffer)


@dataclass(frozen=True, slots=True)
class ArraySerialized(CollectionSerialized[list[object]]):
    TAG: ClassVar[int] = Tags.ARRAY

    @property
    def value(self) -> list[object]:
        return [item.as_python() for item in self.iter_()]


@dataclass(frozen=True, slots=True)
class LargeArraySerialized(LargeCollectionSerialized[list[object]]):
    TAG: ClassVar[int] = Tags.LARGE_ARRAY

    @property
    def value(self) -> list[object]:
        return [item.as_python() for item in self.iter_()]


@dataclass(frozen=True, slots=True)
class ListSerialized(CollectionSerialized[list[object]]):
    TAG: ClassVar[int] = Tags.LIST

    @property
    def value(self) -> list[object]:
        return [item.as_python() for item in self.iter_()]


@dataclass(frozen=True, slots=True)
class LargeListSerialized(LargeCollectionSerialized[list[object]]):
    TAG: ClassVar[int] = Tags.LARGE_LIST

    @property
    def value(self) -> list[object]:
        return [item.as_python() for item in self.iter_()]


@dataclass(frozen=True, slots=True)
class TupleSerialized(CollectionSerialized[tuple[object, ...]]):
    TAG: ClassVar[int] = Tags.TUPLE

    @property
    def value(self) -> tuple[object, ...]:
        return tuple(item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class LargeTupleSerialized(LargeCollectionSerialized[tuple[object, ...]]):
    TAG: ClassVar[int] = Tags.LARGE_TUPLE

    @property
    def value(self) -> tuple[object, ...]:
        return tuple(item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class SetSerialized(CollectionSerialized[set[object]]):
    TAG: ClassVar[int] = Tags.SET

    @property
    def value(self) -> set[object]:
        return {item.as_python() for item in self.iter_()}


@dataclass(frozen=True, slots=True)
class LargeSetSerialized(LargeCollectionSerialized[set[object]]):
    TAG: ClassVar[int] = Tags.LARGE_SET

    @property
    def value(self) -> set[object]:
        return {item.as_python() for item in self.iter_()}


@dataclass(frozen=True, slots=True)
class GeneratorSerialized(CollectionSerialized[Iterator[object]]):
    TAG: ClassVar[int] = Tags.GENERATOR

    @property
    def value(self) -> Iterator[object]:
        return (item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class LargeGeneratorSerialized(LargeCollectionSerialized[Iterator[object]]):
    TAG: ClassVar[int] = Tags.LARGE_GENERATOR

    @property
    def value(self) -> Iterator[object]:
        return (item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class IteratorSerialized(CollectionSerialized[Iterator[object]]):
    TAG: ClassVar[int] = Tags.ITERATOR

    @property
    def value(self) -> Iterator[object]:
        return (item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class LargeIteratorSerialized(LargeCollectionSerialized[Iterator[object]]):
    TAG: ClassVar[int] = Tags.LARGE_ITERATOR

    @property
    def value(self) -> Iterator[object]:
        return (item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class _BaseMappingSerialized(CollectionSerialized[dict[object, object]]):
    def iter_(self) -> Iterator[Serialized[object]]:
        buf = self._payload_buffer()
        count = self._read_count(buf)

        for _ in range(count):
            key_start = buf.tell()
            key = Serialized.read_from(buf, pos=key_start)
            buf.seek(key.head.payload_end)
            yield key

            value_start = buf.tell()
            value = Serialized.read_from(buf, pos=value_start)
            buf.seek(value.head.payload_end)
            yield value

    def iter_entries(self) -> Iterator[tuple[Serialized[object], Serialized[object]]]:
        buf = self._payload_buffer()
        count = self._read_count(buf)

        for _ in range(count):
            key_start = buf.tell()
            key = Serialized.read_from(buf, pos=key_start)
            buf.seek(key.head.payload_end)

            value_start = buf.tell()
            value = Serialized.read_from(buf, pos=value_start)
            buf.seek(value.head.payload_end)

            yield key, value

    @property
    def entries(self) -> tuple[tuple[Serialized[object], Serialized[object]], ...]:
        return tuple(self.iter_entries())

    @property
    def value(self) -> dict[object, object]:
        return {
            key.as_python(): value.as_python()
            for key, value in self.iter_entries()
        }


@dataclass(frozen=True, slots=True)
class MappingSerialized(_BaseMappingSerialized):
    TAG: ClassVar[int] = Tags.MAPPING


@dataclass(frozen=True, slots=True)
class LargeMappingSerialized(
    _BaseMappingSerialized,
    LargeCollectionSerialized[dict[object, object]],
):
    TAG: ClassVar[int] = Tags.LARGE_MAPPING


for cls in CollectionSerialized.__subclasses__():
    Tags.register_class(cls)

for cls in LargeCollectionSerialized.__subclasses__():
    Tags.register_class(cls)

for cls in _BaseMappingSerialized.__subclasses__():
    Tags.register_class(cls)