from __future__ import annotations

import base64
import binascii
import logging
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
    "serialize",
]


LOGGER = logging.getLogger(__name__)


def serialize(
    obj: Any,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> Serialized:
    return Serialized.from_python_object(
        obj,
        metadata=metadata,
        codec=codec,
    )


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
    LOGGER.info("Dumping %s in %s", obj, path)

    try:
        with path.open("wb") as f:
            _dump(
                obj,
                f,
                metadata=metadata,
                codec=codec,
            )
            return None
    except (OSError, IOError):
        # likely parent does not exist; create it and retry once
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("wb") as f:
            _dump(
                obj,
                f,
                metadata=metadata,
                codec=codec,
            )
            return None
    except BaseException:
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

    if isinstance(fp, (Path, str)):
        fp = Path(fp) if isinstance(fp, str) else fp

        try:
            return _dump_path(
                obj,
                fp,
                metadata=metadata,
                codec=codec,
            )
        except BaseException:
            fp.unlink(missing_ok=True)
            raise

    raise SerializationError(
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
    clean_corrupted: bool = False,
    default: Any = None,
) -> Any:
    buffer = BytesIO(fp, copy=False)

    mag = buffer.read(len(MAGIC))
    if mag != MAGIC:
        if isinstance(fp, (str, Path)):
            p = Path(fp)

            if clean_corrupted:
                if not p.exists():
                    return default

                if p.stat().st_size > 0:
                    LOGGER.warning(
                        "Invalid magic header in %r; file may be corrupted. Removing file.", fp
                    )

                p.unlink(missing_ok=True)
                return default

            raise SerializationError(f"Invalid magic header in file {fp!r}")
        raise SerializationError("Invalid magic header")

    try:
        read = Serialized.read_from(buffer)
    except Exception:
        if clean_corrupted:
            LOGGER.warning(
                "Failed to read serialized data from %r; file may be corrupted. Removing file.", fp
            )
            if isinstance(fp, (str, Path)):
                p = Path(fp)
                p.unlink(missing_ok=True)
                return default
        raise

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