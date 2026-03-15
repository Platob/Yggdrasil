# yggdrasil.pickle.ser.ios
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import ClassVar, Mapping

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "IOSerialized",
]


def _merge_metadata(
    base: Mapping[bytes, bytes] | None,
    extra: Mapping[bytes, bytes] | None = None,
) -> dict[bytes, bytes] | None:
    if not base and not extra:
        return None
    out: dict[bytes, bytes] = {}
    if base:
        out.update(base)
    if extra:
        out.update(extra)
    return out


def _read_io_data(obj: io.IOBase) -> bytes:
    seekable = False
    pos = None

    try:
        seekable = obj.seekable()
    except Exception:
        seekable = False

    if seekable:
        try:
            pos = obj.tell()
            obj.seek(0)
            data = obj.read()
            obj.seek(pos)
        except Exception:
            data = obj.read()
    else:
        data = obj.read()

    if data is None:
        return b""
    if isinstance(data, str):
        encoding = getattr(obj, "encoding", None) or "utf-8"
        errors = getattr(obj, "errors", None) or "strict"
        return data.encode(encoding, errors=errors)
    return bytes(data)


def _io_metadata(obj: io.IOBase) -> dict[bytes, bytes]:
    extra: dict[bytes, bytes] = {}

    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        extra[b"io_name"] = name.encode("utf-8", errors="replace")

    mode = getattr(obj, "mode", None)
    if isinstance(mode, str) and mode:
        extra[b"io_mode"] = mode.encode("utf-8", errors="replace")

    if isinstance(obj, io.TextIOBase):

        encoding = getattr(obj, "encoding", None)
        if isinstance(encoding, str) and encoding:
            extra[b"io_encoding"] = encoding.encode("utf-8", errors="replace")

        errors = getattr(obj, "errors", None)
        if isinstance(errors, str) and errors:
            extra[b"io_errors"] = errors.encode("utf-8", errors="replace")

        newline = getattr(obj, "newlines", None)
        if isinstance(newline, str):
            extra[b"io_newline"] = newline.encode("utf-8", errors="replace")
        elif newline is None:
            # preserve that newline translation was unspecified / not observed
            extra[b"io_newline"] = b""
    else:
        extra[b"io_kind"] = b"binary"

    return extra


@dataclass(frozen=True, slots=True)
class IOSerialized(Serialized[BytesIO]):
    TAG: ClassVar[int] = Tags.IO

    @property
    def value(self) -> BytesIO:
        return BytesIO(self.decode())

    def as_python(self) -> BytesIO:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, io.IOBase):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None

    @classmethod
    def from_value(
        cls,
        obj: io.IOBase,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = _read_io_data(obj)
        merged = _merge_metadata(metadata, _io_metadata(obj))

        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=merged,
            codec=codec,
        )


Tags.register_class(IOSerialized)

for cls in IOSerialized.__subclasses__():
    Tags.register_class(cls)