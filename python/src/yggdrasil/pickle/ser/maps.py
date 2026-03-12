from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, ClassVar

from yggdrasil.io import BytesIO

from .registry import REGISTRY
from .serialized import MapSerialized, Serialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

__all__ = [
    "DictSerialized",
    "OrderedDictSerialized",
]


@dataclass(frozen=True, slots=True)
class DictSerialized(MapSerialized):
    TAG: ClassVar[int] = SerdeTags.DICT

    @property
    def value(self) -> dict[Any, Any]:
        it = self.iter_()
        return {k.value: v.value for k, v in zip(it, it)}

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "DictSerialized":
        if not isinstance(value, dict) or isinstance(value, OrderedDict):
            raise TypeError(f"{cls.__name__} only supports dict")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for k, v in value.items():
            Serialized.from_python(k).bwrite(payload)
            Serialized.from_python(v).bwrite(payload)

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
class OrderedDictSerialized(MapSerialized):
    TAG: ClassVar[int] = SerdeTags.ORDEREDDICT

    @property
    def value(self) -> OrderedDict[Any, Any]:
        it = self.iter_()
        return OrderedDict((k.value, v.value) for k, v in zip(it, it))

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "OrderedDictSerialized":
        if not isinstance(value, OrderedDict):
            raise TypeError(f"{cls.__name__} only supports OrderedDict")

        payload = BytesIO() if payload is None else payload
        start_index = payload.tell()
        for k, v in value.items():
            Serialized.from_python(k).bwrite(payload)
            Serialized.from_python(v).bwrite(payload)

        size = payload.tell() - start_index
        data, size, start_index, codec = cls._maybe_compress(payload, size, start_index, byte_limit=byte_limit)
        return cls(
            metadata={} if metadata is None else dict(metadata),
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


REGISTRY.register_python_type(dict, DictSerialized)
REGISTRY.register_python_type(OrderedDict, OrderedDictSerialized)