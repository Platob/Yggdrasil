from __future__ import annotations

import importlib
import types
from dataclasses import dataclass
from typing import Any, ClassVar

from yggdrasil.io import BytesIO

from .registry import REGISTRY
from .serialized import PrimitiveSerialized, _COMPRESS_THRESHOLD
from .tags import SerdeTags

__all__ = ["ModuleSerialized"]


@dataclass(frozen=True, slots=True)
class ModuleSerialized(PrimitiveSerialized):
    """Serialize Python modules **by reference** (module name only).

    The payload contains the fully-qualified module name encoded as
    UTF-8.  On deserialization the module is re-imported via
    :func:`importlib.import_module`.  This avoids pickling the entire
    module object and guarantees that the *live* module in the target
    interpreter is returned.
    """

    TAG: ClassVar[int] = SerdeTags.MODULE

    @property
    def value(self) -> types.ModuleType:
        name = self.payload().decode("utf-8")
        return importlib.import_module(name)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        payload: BytesIO | None = None,
        metadata: dict[bytes, bytes] | None = None,
        byte_limit: int | None = _COMPRESS_THRESHOLD,
    ) -> "ModuleSerialized":
        if not isinstance(value, types.ModuleType):
            raise TypeError(f"{cls.__name__} only supports module objects")

        name = getattr(value, "__name__", None)
        if name is None:
            raise ValueError("Module has no __name__ attribute")

        raw = name.encode("utf-8")

        md = {} if metadata is None else dict(metadata)
        md[b"module"] = raw

        return cls.from_raw(raw, metadata=md)


REGISTRY.register_python_type(types.ModuleType, ModuleSerialized)

