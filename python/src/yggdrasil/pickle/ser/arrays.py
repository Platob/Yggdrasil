from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from yggdrasil.io import BytesIO

from .registry import REGISTRY
from .serialized import ArraySerialized, Serialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

__all__ = [
    "ListSerialized",
    "TupleSerialized",
    "SetSerialized",
    "FrozenSetSerialized",
]


@dataclass(frozen=True, slots=True)
class ListSerialized(ArraySerialized):
    TAG: ClassVar[int] = SerdeTags.LIST

    @property
    def value(self) -> list[Any]:
        return [item.value for item in self.iter_()]

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "ListSerialized":
        if not isinstance(value, list):
            raise TypeError(f"{cls.__name__} only supports list")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for item in value:
            Serialized.from_python(item).bwrite(payload)

        size = payload.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(payload, size, start_index, byte_limit=byte_limit)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class TupleSerialized(ArraySerialized):
    TAG: ClassVar[int] = SerdeTags.TUPLE

    @property
    def value(self) -> tuple[Any, ...]:
        return tuple(item.value for item in self.iter_())

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "TupleSerialized":
        if not isinstance(value, tuple):
            raise TypeError(f"{cls.__name__} only supports tuple")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for item in value:
            Serialized.from_python(item).bwrite(payload)

        size = payload.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(payload, size, start_index, byte_limit=byte_limit)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class SetSerialized(ArraySerialized):
    TAG: ClassVar[int] = SerdeTags.SET

    @property
    def value(self) -> set[Any]:
        return {item.value for item in self.iter_()}

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "SetSerialized":
        if not isinstance(value, set):
            raise TypeError(f"{cls.__name__} only supports set")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for item in value:
            Serialized.from_python(item).bwrite(payload)

        size = payload.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(payload, size, start_index, byte_limit=byte_limit)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class FrozenSetSerialized(ArraySerialized):
    TAG: ClassVar[int] = SerdeTags.FROZENSET

    @property
    def value(self) -> frozenset[Any]:
        return frozenset(item.value for item in self.iter_())

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "FrozenSetSerialized":
        if not isinstance(value, frozenset):
            raise TypeError(f"{cls.__name__} only supports frozenset")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for item in value:
            Serialized.from_python(item).bwrite(payload)

        size = payload.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(payload, size, start_index, byte_limit=byte_limit)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


REGISTRY.register_python_type(list, ListSerialized)
REGISTRY.register_python_type(tuple, TupleSerialized)
REGISTRY.register_python_type(set, SetSerialized)
REGISTRY.register_python_type(frozenset, FrozenSetSerialized)