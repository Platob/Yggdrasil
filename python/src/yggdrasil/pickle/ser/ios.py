from __future__ import annotations

import io
from dataclasses import dataclass
from typing import ClassVar, Mapping

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "IOSerialized",
    "BinaryIOSerialized",
    "TextIOSerialized",
    "BytesBufferSerialized",
    "StringBufferSerialized",
]

# compact metadata keys
_M_NAME = b"n"   # stream name
_M_MODE = b"m"   # stream mode
_M_KIND = b"k"   # io family discriminator
_M_ENC = b"e"    # text encoding
_M_ERR = b"r"    # text errors
_M_NL = b"nl"    # newline / observed newline state

# compact metadata values
_K_BIN = b"b"    # binary stream
_K_TXT = b"t"    # text stream
_K_BIO = b"bb"   # io.BytesIO
_K_SIO = b"sb"   # io.StringIO


# ============================================================================
# helpers
# ============================================================================

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


def _read_binary_io_data(obj: io.IOBase) -> bytes:
    """
    Read bytes from a binary IO object.

    If the stream is seekable, preserve the current cursor position.
    """
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
        # defensive fallback: weird stream lied about binary-ness
        return data.encode("utf-8", errors="replace")

    return bytes(data)


def _read_text_io_data(obj: io.TextIOBase) -> str:
    """
    Read text from a text stream.

    If the stream is seekable, preserve the current cursor position.
    """
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
        return ""

    if isinstance(data, bytes):
        encoding = getattr(obj, "encoding", None) or "utf-8"
        errors = getattr(obj, "errors", None) or "strict"
        return data.decode(encoding, errors=errors)

    return str(data)


def _encode_text_payload(
    text: str,
    *,
    encoding: str,
    errors: str,
) -> bytes:
    return text.encode(encoding, errors=errors)


def _decode_text_payload(
    data: bytes,
    *,
    encoding: str,
    errors: str,
) -> str:
    return data.decode(encoding, errors=errors)


def _io_base_metadata(obj: io.IOBase) -> dict[bytes, bytes]:
    extra: dict[bytes, bytes] = {}

    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        extra[_M_NAME] = name.encode("utf-8", errors="replace")

    mode = getattr(obj, "mode", None)
    if isinstance(mode, str) and mode:
        extra[_M_MODE] = mode.encode("utf-8", errors="replace")

    return extra


def _text_io_metadata(obj: io.TextIOBase) -> dict[bytes, bytes]:
    extra = _io_base_metadata(obj)
    extra[_M_KIND] = _K_TXT

    encoding = getattr(obj, "encoding", None)
    if isinstance(encoding, str) and encoding:
        extra[_M_ENC] = encoding.encode("utf-8", errors="replace")

    errors = getattr(obj, "errors", None)
    if isinstance(errors, str) and errors:
        extra[_M_ERR] = errors.encode("utf-8", errors="replace")

    newlines = getattr(obj, "newlines", None)
    if isinstance(newlines, str):
        extra[_M_NL] = newlines.encode("utf-8", errors="replace")
    elif newlines is None:
        extra[_M_NL] = b""

    return extra


def _binary_io_metadata(obj: io.IOBase) -> dict[bytes, bytes]:
    extra = _io_base_metadata(obj)
    extra[_M_KIND] = _K_BIN
    return extra


def _metadata_text(
    metadata: Mapping[bytes, bytes] | None,
    key: bytes,
    default: str,
) -> str:
    if not metadata:
        return default
    raw = metadata.get(key)
    if raw is None:
        return default
    return raw.decode("utf-8", errors="replace")


# ============================================================================
# base serializer
# ============================================================================

@dataclass(frozen=True, slots=True)
class IOSerialized(Serialized[object]):
    """
    Base serializer for Python IO-like objects.

    Subclasses specialize common in-memory forms:
    - binary streams -> BytesIO
    - text streams -> StringIO
    - exact buffers -> io.BytesIO / io.StringIO
    """

    TAG: ClassVar[int] = Tags.IO

    @property
    def value(self) -> object:
        kind = (self.metadata or {}).get(_M_KIND, _K_BIN)

        if kind in (_K_TXT, _K_SIO):
            encoding = _metadata_text(self.metadata, _M_ENC, "utf-8")
            errors = _metadata_text(self.metadata, _M_ERR, "strict")
            text = _decode_text_payload(self.decode(), encoding=encoding, errors=errors)
            return io.StringIO(text)

        return BytesIO(self.decode())

    def as_python(self) -> object:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        # order matters: most specific first
        if isinstance(obj, io.BytesIO):
            return BytesBufferSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, io.StringIO):
            return StringBufferSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, io.TextIOBase):
            return TextIOSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, io.IOBase):
            return BinaryIOSerialized.from_value(obj, metadata=metadata, codec=codec)

        return None


# ============================================================================
# binary IO
# ============================================================================

@dataclass(frozen=True, slots=True)
class BinaryIOSerialized(IOSerialized):
    """
    Generic binary IO serializer.

    Decodes to yggdrasil BytesIO.
    """

    TAG: ClassVar[int] = Tags.IO_BINARY

    @property
    def value(self) -> BytesIO:
        return BytesIO(self.decode())

    def as_python(self) -> BytesIO:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: io.IOBase,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = _read_binary_io_data(obj)
        merged = _merge_metadata(metadata, _binary_io_metadata(obj))
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=merged,
            codec=codec,
        )


# ============================================================================
# text IO
# ============================================================================

@dataclass(frozen=True, slots=True)
class TextIOSerialized(IOSerialized):
    """
    Generic text IO serializer.

    Payload is encoded text bytes plus compact text metadata.
    Decodes to io.StringIO.
    """

    TAG: ClassVar[int] = Tags.IO_TEXT

    @property
    def value(self) -> io.StringIO:
        encoding = _metadata_text(self.metadata, _M_ENC, "utf-8")
        errors = _metadata_text(self.metadata, _M_ERR, "strict")
        text = _decode_text_payload(self.decode(), encoding=encoding, errors=errors)
        return io.StringIO(text)

    def as_python(self) -> io.StringIO:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: io.TextIOBase,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        text = _read_text_io_data(obj)
        encoding = getattr(obj, "encoding", None) or "utf-8"
        errors = getattr(obj, "errors", None) or "strict"
        payload = _encode_text_payload(text, encoding=encoding, errors=errors)

        merged = _merge_metadata(metadata, _text_io_metadata(obj))
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=merged,
            codec=codec,
        )


# ============================================================================
# exact in-memory buffers
# ============================================================================

@dataclass(frozen=True, slots=True)
class BytesBufferSerialized(BinaryIOSerialized):
    """
    Exact serializer for io.BytesIO.

    Decodes back to io.BytesIO rather than yggdrasil BytesIO.
    """

    TAG: ClassVar[int] = Tags.IO_BYTES_BUFFER

    @property
    def value(self) -> io.BytesIO:
        return io.BytesIO(self.decode())

    def as_python(self) -> io.BytesIO:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: io.BytesIO,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        payload = obj.getvalue()
        merged = _merge_metadata(
            metadata,
            {
                **_binary_io_metadata(obj),
                _M_KIND: _K_BIO,
            },
        )
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class StringBufferSerialized(TextIOSerialized):
    """
    Exact serializer for io.StringIO.

    Decodes back to io.StringIO.
    """

    TAG: ClassVar[int] = Tags.IO_STRING_BUFFER

    @property
    def value(self) -> io.StringIO:
        encoding = _metadata_text(self.metadata, _M_ENC, "utf-8")
        errors = _metadata_text(self.metadata, _M_ERR, "strict")
        text = _decode_text_payload(self.decode(), encoding=encoding, errors=errors)
        return io.StringIO(text)

    def as_python(self) -> io.StringIO:
        return self.value

    @classmethod
    def from_value(
        cls,
        obj: io.StringIO,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        text = obj.getvalue()
        encoding = "utf-8"
        errors = "strict"
        payload = _encode_text_payload(text, encoding=encoding, errors=errors)

        merged = _merge_metadata(
            metadata,
            {
                **_text_io_metadata(obj),
                _M_KIND: _K_SIO,
                _M_ENC: b"utf-8",
                _M_ERR: b"strict",
            },
        )
        return cls.build(
            tag=cls.TAG,
            data=payload,
            metadata=merged,
            codec=codec,
        )


# ============================================================================
# registration
# ============================================================================

for pytype, cls in (
    (BytesIO, IOSerialized),
    (io.IOBase, BinaryIOSerialized),
    (io.TextIOBase, TextIOSerialized),
    (io.BytesIO, BytesBufferSerialized),
    (io.StringIO, StringBufferSerialized),
):
    Tags.register_class(cls, pytype=pytype)