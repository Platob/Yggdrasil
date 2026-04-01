from __future__ import annotations

import pickle
import socket
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "SensitiveObjectSerialized",
    "is_sensitive_object",
]


_SENSITIVE_TYPES: tuple[type[object], ...] = (
    socket.socket,
    type(threading.Lock()),
    type(threading.RLock()),
)


def is_sensitive_object(obj: object) -> bool:
    if isinstance(obj, _SENSITIVE_TYPES):
        return True

    mod = type(obj).__module__
    if mod in {"socket", "_thread", "threading"}:
        return True
    return False


@dataclass(frozen=True, slots=True)
class SensitiveObjectSerialized(Serialized[dict[str, str]]):
    """
    Safe placeholder serialization for sensitive runtime-bound objects.

    We intentionally do not attempt to reconstruct live resources such as
    sockets and thread locks during deserialization.
    """

    TAG: ClassVar[int] = Tags.SENSITIVE_OBJECT

    @property
    def value(self) -> dict[str, str]:
        return pickle.loads(self.decode())

    def as_python(self) -> dict[str, str]:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if not is_sensitive_object(obj):
            return None

        payload = {
            "kind": "sensitive_object",
            "module": type(obj).__module__,
            "qualname": type(obj).__qualname__,
            "repr": repr(obj),
        }
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        return Serialized.build(tag=cls.TAG, data=data, metadata=metadata, codec=codec)


Tags.register_class(SensitiveObjectSerialized, tag=SensitiveObjectSerialized.TAG)
