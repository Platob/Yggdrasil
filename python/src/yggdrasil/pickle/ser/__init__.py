from __future__ import annotations

import base64
import binascii
from typing import Any, Mapping, IO

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.constants import MAGIC
from yggdrasil.pickle.ser.errors import (
    HeaderDecodeError,
    InvalidCodecError,
    MetadataDecodeError,
    SerializationError,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "Tags",
    "Serialized",
    "SerializationError",
    "HeaderDecodeError",
    "MetadataDecodeError",
    "InvalidCodecError",
    "dump",
    "dumps",
    "load",
    "loads",
]

def dump(
    obj: Any,
    fp: IO[bytes],
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> bytes:
    dumped = Serialized.from_python_object(
        obj,
        metadata=metadata,
        codec=codec
    )

    fp.write(MAGIC)
    fp.write(dumped.data.parent.to_bytes())


def dumps(
    obj: Any,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
    b64: bool = False,
) -> bytes | str:
    with BytesIO() as buffer:
        dump(obj, buffer, metadata=metadata, codec=codec)

        value = buffer.getvalue()

        if not b64:
            return value

        return base64.urlsafe_b64encode(value).decode("ascii")


def load(
    fp: IO[bytes],
    *,
    unpickle: bool = True,
) -> Any:
    buffer = BytesIO(fp, copy=False)

    mag = buffer.read(len(MAGIC))
    if mag != MAGIC:
        raise SerializationError("Invalid magic header")

    read = Serialized.read_from(buffer)

    if unpickle:
        return read.as_python()
    return read


def loads(
    s: bytes | str,
    *,
    unpickle: bool = True,
) -> Any:
    if isinstance(s, str):
        try:
            s = base64.urlsafe_b64decode(s.encode("ascii"))
        except (binascii.Error, UnicodeEncodeError) as e:
            err_msg = s if len(s) < 100 else f"{s[:10]}...{s[-10:]}"

            raise HeaderDecodeError(
                f"Invalid base64-encoded string {err_msg!r}"
            ) from e

    with BytesIO(s, copy=False) as buffer:
        return load(buffer, unpickle=unpickle)
