from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Generator as AbcGenerator
from collections.abc import Iterator as AbcIterator
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from types import MappingProxyType
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
    "FrozenSetSerialized",
    "LargeFrozenSetSerialized",
    "DequeSerialized",
    "LargeDequeSerialized",
    "MappingSerialized",
    "LargeMappingSerialized",
    "MappingProxySerialized",
    "LargeMappingProxySerialized",
    "GeneratorSerialized",
    "LargeGeneratorSerialized",
    "IteratorSerialized",
    "LargeIteratorSerialized",
]

_U32_MAX = 0xFFFFFFFF


# ============================================================================
# low-level count helpers
# ============================================================================

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


def _write_count(buffer: BytesIO, count: int, *, large: bool) -> None:
    buffer.write(count.to_bytes(8 if large else 4, "big", signed=False))


def _is_large_count(count: int) -> bool:
    return count > _U32_MAX


# ============================================================================
# payload builders / readers
# ============================================================================

def _iter_items(buffer: BytesIO, count: int) -> Iterator[Serialized[object]]:
    for _ in range(count):
        start = buffer.tell()
        item = Serialized.read_from(buffer, pos=start)
        buffer.seek(item.head.payload_end)
        yield item


def _iter_entry_pairs(
    buffer: BytesIO,
    count: int,
) -> Iterator[tuple[Serialized[object], Serialized[object]]]:
    for _ in range(count):
        key_start = buffer.tell()
        key = Serialized.read_from(buffer, pos=key_start)
        buffer.seek(key.head.payload_end)

        value_start = buffer.tell()
        value = Serialized.read_from(buffer, pos=value_start)
        buffer.seek(value.head.payload_end)

        yield key, value


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
    Snapshot a one-shot iterator/generator into an immutable tuple so the wire
    format can include an upfront count.
    """
    values = tuple(obj)
    return values, len(values)


def _build_sequence_serialized(
    *,
    tag_small: int,
    tag_large: int,
    items: Iterator[object],
    count: int,
    metadata: Mapping[bytes, bytes] | None,
    codec: int | None,
) -> Serialized[object]:
    large = _is_large_count(count)
    payload = _build_collection_payload(items, count=count, large=large)
    return Serialized.build(
        tag=tag_large if large else tag_small,
        data=payload.to_bytes(),
        metadata=metadata,
        codec=codec,
    )


def _build_mapping_serialized(
    *,
    tag_small: int,
    tag_large: int,
    items: Iterator[tuple[object, object]],
    count: int,
    metadata: Mapping[bytes, bytes] | None,
    codec: int | None,
) -> Serialized[object]:
    large = _is_large_count(count)
    payload = _build_mapping_payload(items, count=count, large=large)
    return Serialized.build(
        tag=tag_large if large else tag_small,
        data=payload.to_bytes(),
        metadata=metadata,
        codec=codec,
    )


# ============================================================================
# base classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class CollectionSerialized(Serialized[T], Generic[T]):
    """
    Base class for count-prefixed collection payloads.

    Standard wire format
    --------------------
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
        # ------------------------------------------------------------------
        # concrete builtins / stdlib containers with known length
        # ------------------------------------------------------------------
        if isinstance(obj, list):
            return _build_sequence_serialized(
                tag_small=Tags.LIST,
                tag_large=Tags.LARGE_LIST,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, tuple):
            return _build_sequence_serialized(
                tag_small=Tags.TUPLE,
                tag_large=Tags.LARGE_TUPLE,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, set):
            return _build_sequence_serialized(
                tag_small=Tags.SET,
                tag_large=Tags.LARGE_SET,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, frozenset):
            return _build_sequence_serialized(
                tag_small=Tags.FROZENSET,
                tag_large=Tags.LARGE_FROZENSET,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, deque):
            return _build_sequence_serialized(
                tag_small=Tags.DEQUE,
                tag_large=Tags.LARGE_DEQUE,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, array):
            return _build_sequence_serialized(
                tag_small=Tags.ARRAY,
                tag_large=Tags.LARGE_ARRAY,
                items=iter(obj),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # mappings
        # ------------------------------------------------------------------
        if isinstance(obj, dict):
            return _build_mapping_serialized(
                tag_small=Tags.MAPPING,
                tag_large=Tags.LARGE_MAPPING,
                items=iter(obj.items()),
                count=len(obj),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, MappingProxyType):
            items = tuple(obj.items())
            return _build_mapping_serialized(
                tag_small=Tags.MAPPING_PROXY,
                tag_large=Tags.LARGE_MAPPING_PROXY,
                items=iter(items),
                count=len(items),
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, AbcMapping):
            items = tuple(obj.items())
            return _build_mapping_serialized(
                tag_small=Tags.MAPPING,
                tag_large=Tags.LARGE_MAPPING,
                items=iter(items),
                count=len(items),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # one-shot iterables
        # ------------------------------------------------------------------
        if isinstance(obj, AbcGenerator):
            values, count = _materialize_iterable(obj)
            return _build_sequence_serialized(
                tag_small=Tags.GENERATOR,
                tag_large=Tags.LARGE_GENERATOR,
                items=iter(values),
                count=count,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, AbcIterator):
            values, count = _materialize_iterable(obj)
            return _build_sequence_serialized(
                tag_small=Tags.ITERATOR,
                tag_large=Tags.LARGE_ITERATOR,
                items=iter(values),
                count=count,
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class LargeCollectionSerialized(CollectionSerialized[T], Generic[T]):
    def _read_count(self, buffer: BytesIO) -> int:
        return _read_u64(buffer)


# ============================================================================
# sequence-like concrete serializers
# ============================================================================

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
class FrozenSetSerialized(CollectionSerialized[frozenset[object]]):
    TAG: ClassVar[int] = Tags.FROZENSET

    @property
    def value(self) -> frozenset[object]:
        return frozenset(item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class LargeFrozenSetSerialized(LargeCollectionSerialized[frozenset[object]]):
    TAG: ClassVar[int] = Tags.LARGE_FROZENSET

    @property
    def value(self) -> frozenset[object]:
        return frozenset(item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class DequeSerialized(CollectionSerialized[deque[object]]):
    TAG: ClassVar[int] = Tags.DEQUE

    @property
    def value(self) -> deque[object]:
        return deque(item.as_python() for item in self.iter_())


@dataclass(frozen=True, slots=True)
class LargeDequeSerialized(LargeCollectionSerialized[deque[object]]):
    TAG: ClassVar[int] = Tags.LARGE_DEQUE

    @property
    def value(self) -> deque[object]:
        return deque(item.as_python() for item in self.iter_())


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


# ============================================================================
# mapping serializers
# ============================================================================

@dataclass(frozen=True, slots=True)
class _BaseMappingSerialized(CollectionSerialized[dict[object, object]]):
    def _iter_entry_pairs(self) -> Iterator[tuple[Serialized[object], Serialized[object]]]:
        buf = self._payload_buffer()
        count = self._read_count(buf)
        yield from _iter_entry_pairs(buf, count)

    def iter_(self) -> Iterator[Serialized[object]]:
        for key, value in self._iter_entry_pairs():
            yield key
            yield value

    def iter_entries(self) -> Iterator[tuple[Serialized[object], Serialized[object]]]:
        yield from self._iter_entry_pairs()

    @property
    def entries(self) -> tuple[tuple[Serialized[object], Serialized[object]], ...]:
        return tuple(self.iter_entries())

    def _as_dict(self) -> dict[object, object]:
        return {
            key.as_python(): value.as_python()
            for key, value in self.iter_entries()
        }

    @property
    def value(self) -> dict[object, object]:
        return self._as_dict()


@dataclass(frozen=True, slots=True)
class MappingSerialized(_BaseMappingSerialized):
    TAG: ClassVar[int] = Tags.MAPPING


@dataclass(frozen=True, slots=True)
class LargeMappingSerialized(
    _BaseMappingSerialized,
    LargeCollectionSerialized[dict[object, object]],
):
    TAG: ClassVar[int] = Tags.LARGE_MAPPING


@dataclass(frozen=True, slots=True)
class MappingProxySerialized(_BaseMappingSerialized):
    TAG: ClassVar[int] = Tags.MAPPING_PROXY

    @property
    def value(self) -> MappingProxyType:
        return MappingProxyType(self._as_dict())


@dataclass(frozen=True, slots=True)
class LargeMappingProxySerialized(
    _BaseMappingSerialized,
    LargeCollectionSerialized[dict[object, object]],
):
    TAG: ClassVar[int] = Tags.LARGE_MAPPING_PROXY

    @property
    def value(self) -> MappingProxyType:
        return MappingProxyType(self._as_dict())


# ============================================================================
# registration
# ============================================================================

for cls in CollectionSerialized.__subclasses__():
    Tags.register_class(cls)

for cls in LargeCollectionSerialized.__subclasses__():
    Tags.register_class(cls)

for cls in _BaseMappingSerialized.__subclasses__():
    Tags.register_class(cls)

for t, cls in (
    (list, ListSerialized),
    (tuple, TupleSerialized),
    (set, SetSerialized),
    (frozenset, FrozenSetSerialized),
    (dict, MappingSerialized),
    (deque, DequeSerialized),
    (array, ArraySerialized),
    (MappingProxyType, MappingProxySerialized),
):
    Tags.register_class(cls, pytype=t)