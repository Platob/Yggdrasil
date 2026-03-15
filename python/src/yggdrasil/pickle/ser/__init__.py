from __future__ import annotations

import base64
import binascii
from pathlib import Path
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


def _dump(
    obj: Any,
    fp: IO[bytes],
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
):
    dumped = Serialized.from_python_object(
        obj,
        metadata=metadata,
        codec=codec,
    )

    fp.write(MAGIC)
    fp.write(dumped.data.parent.to_bytes())


def _dump_path(
    obj: Any,
    path: Path,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> bytes:
    try:
        with path.open("wb") as f:
            return _dump(
                obj,
                f,
                metadata=metadata,
                codec=codec,
            )
    except (OSError, IOError):
        # likely parent does not exist; create it and retry once
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("wb") as f:
            return _dump(
                obj,
                f,
                metadata=metadata,
                codec=codec,
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise


def dump(
    obj: Any,
    fp: IO[bytes] | Path | str,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> bytes:
    if hasattr(fp, "write"):
        return _dump(
            obj,
            fp,  # type: ignore[arg-type]
            metadata=metadata,
            codec=codec,
        )

    if isinstance(fp, Path):
        return _dump_path(
            obj,
            fp,
            metadata=metadata,
            codec=codec,
        )

    if isinstance(fp, str):
        return _dump_path(
            obj,
            Path(fp),
            metadata=metadata,
            codec=codec,
        )

    raise TypeError(
        f"Cannot write to object of type {type(fp).__name__!r}. "
        "Expected a file-like object, pathlib.Path, or string path."
    )


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
    fp: IO[bytes] | Path | str,
    *,
    unpickle: bool = True,
) -> Any:
    buffer = BytesIO(fp, copy=False)

    mag = buffer.read(len(MAGIC))
    if mag != MAGIC:
        if isinstance(fp, (str, Path)):
            # check size 0 and clean
            p = Path(fp)
            if p.is_file() and p.stat().st_size == 0:
                p.unlink(missing_ok=True)
                raise SerializationError(f"File {fp!r} is empty")
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