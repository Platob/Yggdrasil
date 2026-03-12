from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, ClassVar

from yggdrasil.io import BytesIO
from .registry import REGISTRY
from .serialized import PrimitiveSerialized, Serialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

try:
    import cloudpickle as _cloudpickle
except ImportError:  # pragma: no cover
    _cloudpickle = None  # type: ignore[assignment]

__all__ = ["ObjectSerialized"]

_PICKLER_NATIVE: bytes = b"native"
_PICKLER_STDLIB: bytes = b"pickle"
_PICKLER_CLOUD: bytes = b"cloudpickle"


def _pickle_dumps(value: Any, protocol: int) -> tuple[bytes, bytes]:
    """Serialise *value* via stdlib pickle then cloudpickle fallback."""
    try:
        return pickle.dumps(value, protocol=protocol), _PICKLER_STDLIB
    except (pickle.PicklingError, AttributeError, TypeError):
        if _cloudpickle is None:
            raise
        return _cloudpickle.dumps(value, protocol=protocol), _PICKLER_CLOUD


def _pickle_loads(raw: bytes, pickler: bytes) -> Any:
    """Deserialise *raw* using the pickler that produced it."""
    if pickler == _PICKLER_CLOUD:
        if _cloudpickle is None:
            raise RuntimeError(
                "cloudpickle is required to deserialise this object; "
                "install it with: pip install cloudpickle"
            )
        return _cloudpickle.loads(raw)
    return pickle.loads(raw)


@dataclass(frozen=True, slots=True)
class ObjectSerialized(PrimitiveSerialized):
    """Fallback serializer for arbitrary Python objects.

    Serialisation strategy (tried in order):

    1. **Native** – :meth:`Serialized.from_python` for any type already
       handled by a registered serializer (scalars, collections, functions,
       modules …).  The inner wire bytes are stored verbatim in the payload
       and decoded via :meth:`Serialized.pread_from` on read.
       Metadata key ``pickler = b"native"``.
    2. **stdlib pickle** – ``pickle.dumps`` for regular picklable objects.
       Metadata key ``pickler = b"pickle"``.
    3. **cloudpickle** – for local classes, lambdas and other objects that
       stdlib pickle cannot handle.
       Metadata key ``pickler = b"cloudpickle"``.

    .. warning::
        Deserialising pickled data from untrusted sources is **unsafe**.
        Only use this serializer in trusted environments.
    """

    TAG: ClassVar[int] = SerdeTags.OBJECT

    @property
    def value(self) -> Any:
        pickler = self.metadata.get(b"pickler", _PICKLER_STDLIB)
        raw = self.payload()
        if pickler == _PICKLER_NATIVE:
            inner, _ = Serialized.pread_from(BytesIO(raw), 0)
            return inner.value
        return _pickle_loads(raw, pickler)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "ObjectSerialized":
        type_name: bytes = type(value).__qualname__.encode("utf-8")

        protocol = pickle.HIGHEST_PROTOCOL
        raw_bytes, pickler = _pickle_dumps(value, protocol)

        md = {} if metadata is None else dict(metadata)
        md[b"type"] = type_name
        md[b"pickler"] = pickler

        if payload is None:
            start_index, payload = 0, BytesIO(raw_bytes)
        else:
            start_index = payload.tell()
            payload.write(raw_bytes)

        size = len(raw_bytes)
        data, size, start_index, codec = cls._maybe_compress(
            payload, size, start_index, byte_limit=byte_limit,
        )
        return cls(
            metadata=md,
            data=data,
            size=size,
            start_index=start_index,
            codec=codec,
        )


REGISTRY.register_tag(ObjectSerialized.TAG, ObjectSerialized)
REGISTRY.register_python_type(object, ObjectSerialized)