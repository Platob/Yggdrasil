from __future__ import annotations

import io
from dataclasses import dataclass
from typing import ClassVar, Mapping

from yggdrasil.io import BytesIO
from yggdrasil.io.base import IO
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
_M_MT = b"mt"    # media type (mime[+codec])

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


def _encode_media_type(obj: io.IOBase) -> bytes | None:
    """Return compact ``b"MIME_NAME"`` or ``b"MIME_NAME+codec"`` for the media type, or *None*."""
    # yggdrasil :class:`IO` carries its media type either directly
    # (storage IO — :class:`Memory` / :class:`Path`) or on the bound
    # ``_parent`` (cursor / format-leaf IO). Probe both, and read
    # through the :attr:`media_type` property when available so the
    # lazy URL-extension resolution fires.
    mt = None
    try:
        mt = obj.media_type
    except Exception:
        mt = None
    if mt is None:
        parent = getattr(obj, "_parent", None)
        if parent is not None:
            try:
                mt = parent.media_type
            except Exception:
                mt = None
    if mt is None:
        mt = getattr(obj, "_media_type", None)
        # ``...`` is the lazy-resolution sentinel on :class:`IO` —
        # a holder whose media_type slot was never observed. Treat it
        # the same as a real ``None`` so the encoder skips the field
        # instead of emitting a malformed wire byte for the Ellipsis.
        if mt is ...:
            mt = None
    if mt is None:
        return None
    wire = mt.mime_type.name.encode("utf-8")
    if mt.codec is not None:
        wire = wire + b"+" + mt.codec.name.encode("utf-8")
    return wire


def _decode_media_type(metadata: Mapping[bytes, bytes] | None):
    """Return the ``MediaType`` encoded in *metadata*, or ``None``.

    Returned to the caller (rather than mutated onto an existing buffer)
    so it can be passed to ``BytesIO(data, media_type=...)``. That route
    runs the registry dispatch in :meth:`BytesIO.__new__` and lands on
    the right registered leaf (ParquetFile, JSONFile, …); a post-hoc
    ``buf._stats.media_type = ...`` would leave the class as the opaque
    ``BytesIO``.
    """
    if not metadata:
        return None
    raw = metadata.get(_M_MT)
    if raw is None:
        return None
    from yggdrasil.enums.codec import Codec
    from yggdrasil.enums.media_type import MediaType
    from yggdrasil.enums.mime_type import MimeType

    text = raw.decode("utf-8")
    if "+" in text:
        mime_name, codec_name = text.split("+", 1)
    else:
        mime_name = text
        codec_name = None

    mt = MimeType._BY_NAME.get(mime_name.lower())
    if mt is None:
        return None

    c = None
    if codec_name:
        c = Codec.from_(codec_name)
        if c is None:
            return None

    return MediaType(mime_type=mt, codec=c)


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

        media_type = _decode_media_type(self.metadata)
        if media_type is not None:
            return BytesIO(self.decode(), media_type=media_type)
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
        # order matters: most specific first.
        # yggdrasil BytesIO does NOT inherit ``io.IOBase`` (it's a
        # Tabular under :mod:`yggdrasil.io.buffer`), so the stdlib
        # checks below would miss it. Match it first and route through
        # the binary path, which already preserves ``_media_type`` via
        # :func:`_encode_media_type`.
        if isinstance(obj, IO):
            return BinaryIOSerialized.from_value(obj, metadata=metadata, codec=codec)

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
        media_type = _decode_media_type(self.metadata)
        if media_type is not None:
            return BytesIO(self.decode(), media_type=media_type)
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
        extra = _binary_io_metadata(obj)
        mt_wire = _encode_media_type(obj)
        if mt_wire is not None:
            extra[_M_MT] = mt_wire
        merged = _merge_metadata(metadata, extra)
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
    (IO, IOSerialized),
    (BytesIO, IOSerialized),
    (io.IOBase, BinaryIOSerialized),
    (io.TextIOBase, TextIOSerialized),
    (io.BytesIO, BytesBufferSerialized),
    (io.StringIO, StringBufferSerialized),
):
    Tags.register_class(cls, pytype=pytype)