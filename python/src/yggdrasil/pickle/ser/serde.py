"""Serialisation / deserialisation entry points for the yggdrasil wire format.

Public API
----------
serialize   — wrap any Python object in a :class:`~yggdrasil.pickle.ser.serialized.Serialized`
dump        — write a serialised object to a file-like object or path
dumps       — serialise to ``bytes`` (or base-64 ``str``)
load        — read and optionally unpickle from a file-like object or path
loads       — read and optionally unpickle from ``bytes`` or a base-64 ``str``
"""
from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path
from typing import Any, IO, Mapping

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.constants import is_valid_magic, MAGIC_LENGTH, MAGIC
from yggdrasil.pickle.ser.errors import (
    HeaderDecodeError,
    SerializationError,
)
from yggdrasil.pickle.ser.serialized import Serialized

__all__ = [
    "serialize",
    "dump",
    "dumps",
    "load",
    "loads",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# serialize
# ---------------------------------------------------------------------------

def serialize(
    obj: Any,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> Serialized:
    """Wrap *obj* in a :class:`Serialized` instance without writing to a buffer."""
    return Serialized.from_python_object(obj, metadata=metadata, codec=codec)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _dump(
    obj: Any,
    fp: IO[bytes],
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> None:
    dumped = Serialized.from_python_object(obj, metadata=metadata, codec=codec)
    fp.write(MAGIC)
    fp.write(dumped.data.parent.to_bytes())


def _dump_path(
    obj: Any,
    path: Path,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> None:
    LOGGER.info("Dumping %s to %s", obj, path)

    try:
        with path.open("wb") as f:
            _dump(obj, f, metadata=metadata, codec=codec)
        return
    except (OSError, IOError):
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("wb") as f:
            _dump(obj, f, metadata=metadata, codec=codec)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# dump / dumps
# ---------------------------------------------------------------------------

def dump(
    obj: Any,
    fp: IO[bytes] | Path | str,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
    codec: int | None = None,
) -> None:
    """Serialise *obj* and write it to *fp* (file-like, :class:`~pathlib.Path`, or str path)."""
    if hasattr(fp, "write"):
        return _dump(obj, fp, metadata=metadata, codec=codec)  # type: ignore[arg-type]

    if isinstance(fp, (Path, str)):
        path = Path(fp) if isinstance(fp, str) else fp
        try:
            return _dump_path(obj, path, metadata=metadata, codec=codec)
        except BaseException:
            path.unlink(missing_ok=True)
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
    """Serialise *obj* to ``bytes``, or to a URL-safe base-64 ``str`` when *b64* is ``True``."""
    with BytesIO() as buffer:
        dump(obj, buffer, metadata=metadata, codec=codec)
        value = buffer.getvalue()

    if not b64:
        return value
    return base64.urlsafe_b64encode(value).decode("ascii")


# ---------------------------------------------------------------------------
# load / loads
# ---------------------------------------------------------------------------

def load(
    fp: IO[bytes] | Path | str,
    *,
    unpickle: bool = True,
    clean_corrupted: bool = False,
    default: Any = None,
) -> Any:
    """Read a serialised payload from *fp* and optionally unpickle it."""
    buffer = BytesIO(fp, copy=False)

    mag = buffer.read(MAGIC_LENGTH)
    if not is_valid_magic(mag):
        if isinstance(fp, (str, Path)):
            p = Path(fp)
            if clean_corrupted:
                if not p.exists():
                    return default
                if p.stat().st_size > 0:
                    LOGGER.warning(
                        "Invalid magic header in %r; file may be corrupted. Removing.", fp
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
                "Failed to read serialized data from %r; removing.", fp
            )
            if isinstance(fp, (str, Path)):
                Path(fp).unlink(missing_ok=True)
                return default
        raise

    return read.as_python() if unpickle else read


def loads(
    s: bytes | str,
    *,
    unpickle: bool = True,
) -> Any:
    """Deserialise from *s* (``bytes`` or URL-safe base-64 ``str``)."""
    if isinstance(s, str):
        try:
            s = base64.urlsafe_b64decode(s.encode("ascii"))
        except (binascii.Error, UnicodeEncodeError) as e:
            label = s if len(s) < 100 else f"{s[:10]}...{s[-10:]}"
            raise HeaderDecodeError(
                f"Invalid base64-encoded string {label!r}"
            ) from e

    with BytesIO(s, copy=False) as buffer:
        return load(buffer, unpickle=unpickle)

