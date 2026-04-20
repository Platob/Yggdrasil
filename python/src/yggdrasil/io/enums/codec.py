# yggdrasil/io/enums/codec.py
"""Compression codec abstraction.

A :class:`Codec` encapsulates a compression scheme. Each concrete codec
provides:

* **Bytes roundtrip** — :meth:`compress_bytes`, :meth:`decompress_bytes`
  for small/medium payloads held entirely in memory.
* **Streaming roundtrip** — :meth:`compress`, :meth:`decompress` stream
  chunk-by-chunk between :class:`BytesIO` buffers when the underlying
  library exposes a streaming encoder/decoder (gzip, zstd, lz4, bz2,
  xz, lzma). Snappy and Brotli fall back to a bytes roundtrip since
  their Python bindings don't expose streaming.
* **Partial decode** — :meth:`read_start_end` reads only the head and
  tail of the uncompressed stream without materializing the entire
  decoded payload.

Streaming support is advertised by the :attr:`is_streaming` flag so
callers can decide whether to trust :meth:`compress` / :meth:`decompress`
with a large input, or fall back to a different strategy for
non-streaming codecs.
"""
from __future__ import annotations

import abc
import importlib
import io as _io
import zlib
from typing import IO, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..buffer import BytesIO
    from .mime_type import MimeType

__all__ = [
    "Codec",
    "GZIP",
    "ZSTD",
    "LZ4",
    "BZIP2",
    "XZ",
    "SNAPPY",
    "BROTLI",
    "ZLIB",
    "LZMA",
    "_drain",
]

_CHUNK = 256 * 1024  # 256 KiB streaming chunk size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _runtime_import(module_name: str, pip_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        from yggdrasil.environ import runtime_import_module
        return runtime_import_module(
            module_name=module_name,
            pip_name=pip_name,
            install=True,
        )


def _drain(fh: "IO[bytes]") -> bytes:
    """Read all remaining bytes from *fh* from the current cursor, restoring cursor.

    This reads from ``fh.tell()`` to EOF and then seeks back to where
    we started. It does NOT start at offset 0 — callers that want to
    read the whole stream from the beginning must seek(0) first.
    """
    pos = fh.tell()
    try:
        return fh.read()
    finally:
        fh.seek(pos)


def _codec_from_mime(mt: "MimeType | None") -> "Codec | None":
    if mt is None or not mt.is_codec:
        return None
    return _CODEC_BY_MIME.get(mt)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Codec(abc.ABC):
    """Abstract compression codec."""

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def mime_type(self) -> "MimeType": ...

    @property
    def extensions(self) -> list[str]:
        return list(self.mime_type.extensions)

    @property
    def extension(self) -> str:
        return self.mime_type.extension

    @property
    def is_streaming(self) -> bool:
        """True when both compress and decompress have streaming paths.

        Callers with large (GiB-scale) inputs should check this before
        passing them to :meth:`compress` / :meth:`decompress`. When
        ``False``, those methods fall back to materializing the full
        payload in memory through :meth:`compress_bytes` /
        :meth:`decompress_bytes`.
        """
        # Probe the subclass hooks without side effects. A subclass
        # that overrides either hook to return a non-None reader/writer
        # opts into streaming.
        return (
            type(self)._open_decompress_reader is not Codec._open_decompress_reader
            and type(self)._open_compress_writer is not Codec._open_compress_writer
        )

    # ------------------------------------------------------------------
    # Bytes API (abstract)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def compress_bytes(self, data: bytes) -> bytes: ...

    @abc.abstractmethod
    def decompress_bytes(self, data: bytes) -> bytes: ...

    # ------------------------------------------------------------------
    # Streaming hooks (subclasses override to opt into streaming)
    # ------------------------------------------------------------------

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        """Return a streaming decompressor over fh (positioned at stream start).

        Subclasses override when the underlying library exposes a
        streaming decoder. Returning ``None`` signals the base class
        to fall back to :meth:`decompress_bytes` on the full stream.
        """
        return None

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        """Return a streaming compressor that writes into fh.

        Subclasses override when the underlying library exposes a
        streaming encoder. Returning ``None`` signals the base class
        to fall back to :meth:`compress_bytes` on the full input.
        """
        return None

    # ------------------------------------------------------------------
    # Streaming compress / decompress
    # ------------------------------------------------------------------

    def compress(
        self,
        src: "IO[bytes] | BytesIO",
    ) -> "BytesIO":
        """Compress *src* into a new :class:`BytesIO`.

        Streams chunk-by-chunk when :meth:`_open_compress_writer` is
        available. Otherwise falls back to a full-in-memory bytes
        roundtrip — callers with multi-GiB inputs should inspect
        :attr:`is_streaming` first.

        The source cursor is restored on exit.
        """
        return self._stream_roundtrip(src, _compress=True)

    def decompress(
        self,
        src: "IO[bytes] | BytesIO",
    ) -> "BytesIO":
        """Decompress *src* into a new :class:`BytesIO`.

        Streams chunk-by-chunk when :meth:`_open_decompress_reader` is
        available. Otherwise falls back to a full-in-memory bytes
        roundtrip — callers with multi-GiB compressed inputs should
        inspect :attr:`is_streaming` first.

        The source cursor is restored on exit.
        """
        return self._stream_roundtrip(src, _compress=False)

    def _stream_roundtrip(
        self,
        src: "IO[bytes] | BytesIO",
        *,
        _compress: bool,
    ) -> "BytesIO":
        """Internal helper shared by :meth:`compress` and :meth:`decompress`.

        Both directions follow the same shape: wrap src, seek-0,
        stream-or-fallback, return a new BytesIO, restore cursor in
        the finally.
        """
        from ..buffer.bytes_io import BytesIO

        fh = BytesIO.parse(src)
        saved = fh.tell()
        try:
            fh.seek(0)
            out = BytesIO()

            if _compress:
                writer = self._open_compress_writer(out)
                if writer is None:
                    # Non-streaming codec — fall back to full-in-memory.
                    out.write(self.compress_bytes(fh.read()))
                else:
                    try:
                        while True:
                            chunk = fh.read(_CHUNK)
                            if not chunk:
                                break
                            writer.write(chunk)
                    finally:
                        try:
                            writer.close()
                        except Exception:
                            pass
            else:
                reader = self._open_decompress_reader(fh)
                if reader is None:
                    # Non-streaming codec — fall back to full-in-memory.
                    out.write(self.decompress_bytes(fh.read()))
                else:
                    try:
                        while True:
                            chunk = reader.read(_CHUNK)
                            if not chunk:
                                break
                            out.write(chunk)
                    finally:
                        try:
                            reader.close()
                        except Exception:
                            pass

            out.seek(0)
            return out
        finally:
            try:
                fh.seek(saved)
            except Exception:
                pass

    def roundtrip(self, data: bytes) -> bool:
        return self.decompress_bytes(self.compress_bytes(data)) == data

    # ------------------------------------------------------------------
    # Partial uncompressed read (head/tail)
    # ------------------------------------------------------------------

    def read_start_end(
        self,
        src: "IO[bytes] | BytesIO | bytes | bytearray | memoryview",
        *,
        n_start: int = 64,
        n_end: int = 64,
        chunk_size: int = _CHUNK,
    ) -> tuple[bytes, bytes]:
        """Return the first *n_start* and last *n_end* bytes of the decoded stream.

        Streams the decompression and keeps only a bounded amount of
        state in memory (:math:`n\\_end + chunk\\_size` bytes) — safe
        for very large compressed inputs when the codec supports
        streaming decompression.

        When the codec does NOT expose a streaming decoder, falls back
        to a full :meth:`decompress_bytes` call, which materializes
        the whole uncompressed payload in memory.
        """
        from ..buffer import BytesIO

        if n_start < 0 or n_end < 0:
            raise ValueError("n_start and n_end must be >= 0")

        # No-op short-circuit: caller wants nothing.
        if n_start == 0 and n_end == 0:
            return b"", b""

        fh = BytesIO(src, copy=False).view(pos=0)
        saved = fh.tell()

        reader: IO[bytes] | None = None
        try:
            fh.seek(0)

            reader = self._open_decompress_reader(fh)
            if reader is None:
                data = self.decompress_bytes(_drain(fh))
                return data[:n_start], (data[-n_end:] if n_end else b"")

            return self._collect_head_tail(reader, n_start, n_end, chunk_size)
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
            try:
                fh.seek(saved)
            except Exception:
                pass

    @staticmethod
    def _collect_head_tail(
        reader: "IO[bytes]",
        n_start: int,
        n_end: int,
        chunk_size: int,
    ) -> tuple[bytes, bytes]:
        """Collect head/tail windows from a streaming reader.

        Shared helper — also used by :class:`_ZlibCodec` which drives
        its own :func:`zlib.decompressobj` loop.
        """
        head = bytearray()
        tail = bytearray()

        while True:
            chunk = reader.read(chunk_size)
            if not chunk:
                break

            if n_start and len(head) < n_start:
                need = n_start - len(head)
                head += chunk[:need]

            if n_end:
                tail += chunk
                if len(tail) > n_end:
                    del tail[:-n_end]

        return bytes(head), (bytes(tail) if n_end else b"")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, default: "Codec | None" = None) -> "Codec | None":
        """Parse an arbitrary input into a Codec instance.

        Accepts:
        - :class:`Codec` instances (returned as-is).
        - Short names like ``"gzip"``, ``"zstd"`` (case-insensitive).
        - Anything :meth:`MimeType.parse` can resolve to a codec mime.
        - ``None`` → returns *default*.
        """
        if isinstance(obj, cls):
            return obj

        if obj is None:
            return default

        # Short-name shortcut: accept bare codec names for ergonomics.
        if isinstance(obj, str):
            hit = _CODEC_BY_NAME.get(obj.lower().strip())
            if hit is not None:
                return hit

        from .mime_type import MimeType as _MimeType

        mt = _MimeType.parse(obj, default=None)
        if mt is None or not mt.is_codec:
            return default

        return cls.from_mime(mt) or default

    @classmethod
    def from_mime(cls, mime: "MimeType | str") -> "Codec | None":
        from .mime_type import MimeType as _MimeType
        mt = mime if isinstance(mime, _MimeType) else _MimeType.parse_str(mime)
        return _codec_from_mime(mt)

    @classmethod
    def all(cls) -> list["Codec"]:
        return list(_ALL_CODECS)

    def __repr__(self) -> str:
        return f"<Codec:{self.name}>"


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class _GzipCodec(Codec):
    @property
    def name(self) -> str:
        return "gzip"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.GZIP

    def compress_bytes(self, data: bytes) -> bytes:
        import gzip
        return gzip.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import gzip
        return gzip.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import gzip
        # GzipFile.close() does NOT close fileobj when passed via fileobj=.
        return gzip.GzipFile(fileobj=fh, mode="rb")

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import gzip
        return gzip.GzipFile(fileobj=fh, mode="wb")


class _ZstdCodec(Codec):
    @property
    def name(self) -> str:
        return "zstd"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.ZSTD

    def compress_bytes(self, data: bytes) -> bytes:
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdCompressor().compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdDecompressor().decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdDecompressor().stream_reader(fh, closefd=False)

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdCompressor().stream_writer(fh, closefd=False)


class _Lz4Codec(Codec):
    @property
    def name(self) -> str:
        return "lz4"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.LZ4

    def compress_bytes(self, data: bytes) -> bytes:
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.open(fh, mode="rb")

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.open(fh, mode="wb")


class _Bzip2Codec(Codec):
    @property
    def name(self) -> str:
        return "bzip2"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.BZ2

    def compress_bytes(self, data: bytes) -> bytes:
        import bz2
        return bz2.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import bz2
        return bz2.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import bz2
        return bz2.BZ2File(fh, mode="rb")

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import bz2
        return bz2.BZ2File(fh, mode="wb")


class _XzCodec(Codec):
    @property
    def name(self) -> str:
        return "xz"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.XZ

    def compress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="rb")

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="wb")


class _SnappyCodec(Codec):
    """Snappy has no streaming API in Python — bytes-only roundtrip."""

    @property
    def name(self) -> str:
        return "snappy"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.SNAPPY

    def compress_bytes(self, data: bytes) -> bytes:
        cj = _runtime_import("cramjam", "cramjam")
        return bytes(cj.snappy.compress(data))

    def decompress_bytes(self, data: bytes) -> bytes:
        cj = _runtime_import("cramjam", "cramjam")
        return bytes(cj.snappy.decompress(data))
    # No _open_*_* overrides → base class falls back to bytes roundtrip
    # and is_streaming → False.


class _BrotliCodec(Codec):
    """Brotli Python bindings don't expose streaming — bytes-only roundtrip."""

    @property
    def name(self) -> str:
        return "brotli"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.BROTLI

    def compress_bytes(self, data: bytes) -> bytes:
        brotli = _runtime_import("brotli", "brotli")
        return brotli.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        brotli = _runtime_import("brotli", "brotli")
        return brotli.decompress(data)
    # No _open_*_* overrides → base class falls back to bytes roundtrip
    # and is_streaming → False.


class _ZlibCodec(Codec):
    """Raw zlib stream. Uses :class:`zlib.decompressobj` for streaming
    decompression and :class:`zlib.compressobj` for streaming
    compression, driven by the base streaming loop via the shared
    ``_open_*`` adapter classes below."""

    @property
    def name(self) -> str:
        return "zlib"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.ZLIB

    def compress_bytes(self, data: bytes) -> bytes:
        return zlib.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        return zlib.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        return _ZlibStreamReader(fh)

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        return _ZlibStreamWriter(fh)


class _ZlibStreamReader(_io.RawIOBase):
    """Minimal read-only file-like over a zlib-compressed fh.

    Reads from the underlying fh in chunks and returns decompressed
    bytes via :func:`zlib.decompressobj`. Supports :meth:`read` only —
    no seek/tell. Close does NOT close the underlying fh.
    """

    def __init__(self, fh: "IO[bytes]", chunk_size: int = _CHUNK) -> None:
        super().__init__()
        self._fh = fh
        self._chunk = chunk_size
        self._decomp = zlib.decompressobj()
        self._buf = bytearray()
        self._eof = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            # Drain everything.
            while not self._eof:
                self._fill()
            out = bytes(self._buf)
            self._buf.clear()
            return out

        while len(self._buf) < size and not self._eof:
            self._fill()

        out = bytes(self._buf[:size])
        del self._buf[:size]
        return out

    def _fill(self) -> None:
        if self._eof:
            return
        comp = self._fh.read(self._chunk)
        if not comp:
            # Flush any remaining buffered state.
            tail = self._decomp.flush()
            if tail:
                self._buf += tail
            self._eof = True
            return
        out = self._decomp.decompress(comp)
        if out:
            self._buf += out

    def close(self) -> None:
        # Do NOT close self._fh — it's owned by the caller.
        super().close()


class _ZlibStreamWriter(_io.RawIOBase):
    """Minimal write-only file-like that zlib-compresses into a fh.

    Close flushes the final block into fh. Does NOT close fh.
    """

    def __init__(self, fh: "IO[bytes]", level: int = -1) -> None:
        super().__init__()
        self._fh = fh
        self._comp = zlib.compressobj(level)
        self._closed_for_writes = False

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        if self._closed_for_writes:
            raise ValueError("write on closed zlib stream")
        if not data:
            return 0
        out = self._comp.compress(data)
        if out:
            self._fh.write(out)
        return len(data)

    def close(self) -> None:
        if not self._closed_for_writes:
            self._closed_for_writes = True
            try:
                tail = self._comp.flush()
                if tail:
                    self._fh.write(tail)
            finally:
                pass
        # Do NOT close self._fh — it's owned by the caller.
        super().close()


class _LzmaCodec(Codec):
    @property
    def name(self) -> str:
        return "lzma"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeTypes
        return MimeTypes.LZMA

    def compress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.compress(data, format=lzma.FORMAT_ALONE)

    def decompress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data, format=lzma.FORMAT_ALONE)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="rb", format=lzma.FORMAT_ALONE)

    def _open_compress_writer(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="wb", format=lzma.FORMAT_ALONE)


# ---------------------------------------------------------------------------
# Singletons + maps
# ---------------------------------------------------------------------------

GZIP: Codec = _GzipCodec()
ZSTD: Codec = _ZstdCodec()
LZ4: Codec = _Lz4Codec()
BZIP2: Codec = _Bzip2Codec()
XZ: Codec = _XzCodec()
SNAPPY: Codec = _SnappyCodec()
BROTLI: Codec = _BrotliCodec()
ZLIB: Codec = _ZlibCodec()
LZMA: Codec = _LzmaCodec()

_ALL_CODECS: list[Codec] = [GZIP, ZSTD, LZ4, BZIP2, XZ, SNAPPY, BROTLI, ZLIB, LZMA]
_CODEC_BY_NAME: dict[str, Codec] = {c.name: c for c in _ALL_CODECS}


def _build_codec_by_mime() -> dict["MimeType", Codec]:
    from .mime_type import MimeType as _MimeType

    out: dict[_MimeType, Codec] = {}
    for c in _ALL_CODECS:
        mt = c.mime_type
        out[mt] = c

        by_value = _MimeType.get(mt.value)
        if by_value is not None:
            out[by_value] = c
    return out


_CODEC_BY_MIME = _build_codec_by_mime()