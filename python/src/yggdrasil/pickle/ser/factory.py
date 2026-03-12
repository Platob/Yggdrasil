from __future__ import annotations

from typing import Any

from yggdrasil.io import BytesIO

from .serialized import Serialized

__all__ = ["dumps", "loads"]


def dumps(obj: Any, *, metadata: dict[bytes, bytes] | None = None) -> bytes:
    return Serialized.from_python(obj, metadata=metadata).to_bytes()


def loads(data: bytes) -> Any:
    return Serialized.pread(BytesIO(data)).value