# yggdrasil/io/enums/codec.py
from __future__ import annotations

import abc
import importlib
from typing import IO, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..buffer import BytesIO
    from .mime_type import MimeType

__all__ = [
    "Codec",
    "detect",
    "detect_bytes",
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
    """Read all remaining bytes from *fh*, cursor preserved."""
    pos = fh.tell()
    try:
        return fh.read()
    finally:
        fh.seek(pos)


def _peek_from_start(fh: "IO[bytes]", n: int) -> bytes:
    """
    Read n bytes from fh starting at offset 0, and restore cursor.

    This is the behavior we want for sniff/detect APIs: independent of current cursor.
    """
    pos = fh.tell()
    try:
        fh.seek(0)
        return fh.read(n)
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

    @abc.abstractmethod
    def compress_bytes(self, data: bytes) -> bytes: ...

    @abc.abstractmethod
    def decompress_bytes(self, data: bytes) -> bytes: ...

    # ------------------------------------------------------------------
    # Streaming API (default: bytes round-trip)
    # ------------------------------------------------------------------

    def compress(self, src: "IO[bytes] | BytesIO") -> "BytesIO":
        from ..buffer.bytes_io import BytesIO

        fh = BytesIO.parse(src)
        saved = fh.tell()
        try:
            fh.seek(0)
            out = BytesIO(self.compress_bytes(_drain(fh)))
            out.seek(0)
            return out  # type: ignore[return-value]
        finally:
            fh.seek(saved)

    def open(self, src: "IO[bytes] | BytesIO") -> "BytesIO":
        from ..buffer.bytes_io import BytesIO

        fh = BytesIO.parse(src)
        saved = fh.tell()
        try:
            fh.seek(0)
            out = BytesIO(self.decompress_bytes(_drain(fh)))
            out.seek(0)
            return out  # type: ignore[return-value]
        finally:
            fh.seek(saved)

    def roundtrip(self, data: bytes) -> bool:
        return self.decompress_bytes(self.compress_bytes(data)) == data

    # ------------------------------------------------------------------
    # Partial uncompressed read (head/tail)
    # ------------------------------------------------------------------

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        """Return a streaming decompressor over fh (fh positioned at stream start)."""
        return None

    def read_start_end(
        self,
        src: "IO[bytes] | BytesIO | bytes | bytearray | memoryview",
        *,
        n_start: int = 64,
        n_end: int = 64,
        chunk_size: int = _CHUNK,
    ) -> tuple[bytes, bytes]:
        if n_start < 0 or n_end < 0:
            raise ValueError("n_start and n_end must be >= 0")

        is_bytes = isinstance(src, (bytes, bytearray, memoryview))
        if is_bytes:
            import io as _io
            fh: IO[bytes] = _io.BytesIO(bytes(src))
            saved = 0
        else:
            from ..buffer.bytes_io import BytesIO
            fh = BytesIO.wrap(src)
            saved = fh.tell()

        reader: IO[bytes] | None = None
        try:
            # critical fix: always decode from start of compressed stream
            fh.seek(0)

            reader = self._open_decompress_reader(fh)
            if reader is None:
                data = self.decompress_bytes(_drain(fh))
                return data[:n_start], (data[-n_end:] if n_end else b"")

            start = bytearray()
            tail = bytearray()

            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break

                if n_start and len(start) < n_start:
                    need = n_start - len(start)
                    start += chunk[:need]

                if n_end:
                    tail += chunk
                    if len(tail) > n_end:
                        del tail[:-n_end]

            return bytes(start), (bytes(tail) if n_end else b"")
        finally:
            try:
                if reader is not None:
                    reader.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            if not is_bytes:
                fh.seek(saved)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, default: "Codec | None" = None) -> "Codec | None":
        if isinstance(obj, cls):
            return obj

        if obj is None:
            return default

        from .mime_type import MimeType as _MimeType

        if isinstance(obj, str):
            mt = _MimeType.parse_str(obj, default=None)
            hit = _codec_from_mime(mt)
            if hit is not None:
                return hit
            return _CODEC_BY_NAME.get(obj.strip().lower(), default)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return detect_bytes(obj) or default

        if hasattr(obj, "read"):
            return detect(obj) or default

        return default

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
        from .mime_type import MimeType
        return MimeType.GZIP

    def compress_bytes(self, data: bytes) -> bytes:
        import gzip
        return gzip.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import gzip
        return gzip.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import gzip
        return gzip.GzipFile(fileobj=fh, mode="rb")


class _ZstdCodec(Codec):
    @property
    def name(self) -> str:
        return "zstd"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.ZSTD

    def compress_bytes(self, data: bytes) -> bytes:
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdCompressor().compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdDecompressor().decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        zstd = _runtime_import("zstandard", "zstandard")
        return zstd.ZstdDecompressor().stream_reader(fh, closefd=False)


class _Lz4Codec(Codec):
    @property
    def name(self) -> str:
        return "lz4"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.LZ4

    def compress_bytes(self, data: bytes) -> bytes:
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        lz4 = _runtime_import("lz4.frame", "lz4")
        return lz4.open(fh, mode="rb")


class _Bzip2Codec(Codec):
    @property
    def name(self) -> str:
        return "bzip2"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.BZ2

    def compress_bytes(self, data: bytes) -> bytes:
        import bz2
        return bz2.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import bz2
        return bz2.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import bz2
        return bz2.BZ2File(fh, mode="rb")


class _XzCodec(Codec):
    @property
    def name(self) -> str:
        return "xz"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.XZ

    def compress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="rb")


class _SnappyCodec(Codec):
    @property
    def name(self) -> str:
        return "snappy"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.SNAPPY

    def compress_bytes(self, data: bytes) -> bytes:
        cj = _runtime_import("cramjam", "cramjam")
        return bytes(cj.snappy.compress(data))

    def decompress_bytes(self, data: bytes) -> bytes:
        cj = _runtime_import("cramjam", "cramjam")
        return bytes(cj.snappy.decompress(data))
    # no streaming reader; base read_start_end falls back (but now from start)


class _BrotliCodec(Codec):
    @property
    def name(self) -> str:
        return "brotli"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.BROTLI

    def compress_bytes(self, data: bytes) -> bytes:
        brotli = _runtime_import("brotli", "brotli")
        return brotli.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        brotli = _runtime_import("brotli", "brotli")
        return brotli.decompress(data)
    # no streaming reader; base read_start_end falls back (but now from start)


class _ZlibCodec(Codec):
    @property
    def name(self) -> str:
        return "zlib"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.ZLIB

    def compress_bytes(self, data: bytes) -> bytes:
        import zlib
        return zlib.compress(data)

    def decompress_bytes(self, data: bytes) -> bytes:
        import zlib
        return zlib.decompress(data)

    def read_start_end(
        self,
        src: "IO[bytes] | BytesIO | bytes | bytearray | memoryview",
        *,
        n_start: int = 64,
        n_end: int = 64,
        chunk_size: int = _CHUNK,
    ) -> tuple[bytes, bytes]:
        import zlib
        import io as _io

        if n_start < 0 or n_end < 0:
            raise ValueError("n_start and n_end must be >= 0")

        is_bytes = isinstance(src, (bytes, bytearray, memoryview))
        if is_bytes:
            fh: IO[bytes] = _io.BytesIO(bytes(src))
            saved = 0
        else:
            from ..buffer.bytes_io import BytesIO
            fh = BytesIO.wrap(src)
            saved = fh.tell()

        try:
            # critical fix: always decode from start of compressed stream
            fh.seek(0)

            decomp = zlib.decompressobj()
            start = bytearray()
            tail = bytearray()

            while True:
                comp = fh.read(chunk_size)
                if not comp:
                    break

                out = decomp.decompress(comp)
                if out:
                    if n_start and len(start) < n_start:
                        need = n_start - len(start)
                        start += out[:need]

                    if n_end:
                        tail += out
                        if len(tail) > n_end:
                            del tail[:-n_end]

            out = decomp.flush()
            if out:
                if n_start and len(start) < n_start:
                    need = n_start - len(start)
                    start += out[:need]
                if n_end:
                    tail += out
                    if len(tail) > n_end:
                        del tail[:-n_end]

            return bytes(start), (bytes(tail) if n_end else b"")
        finally:
            if not is_bytes:
                fh.seek(saved)


class _LzmaCodec(Codec):
    @property
    def name(self) -> str:
        return "lzma"

    @property
    def mime_type(self) -> "MimeType":
        from .mime_type import MimeType
        return MimeType.LZMA

    def compress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.compress(data, format=lzma.FORMAT_ALONE)

    def decompress_bytes(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data, format=lzma.FORMAT_ALONE)

    def _open_decompress_reader(self, fh: "IO[bytes]") -> "IO[bytes] | None":
        import lzma
        return lzma.LZMAFile(fh, mode="rb", format=lzma.FORMAT_ALONE)


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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect(src: "IO[bytes] | BytesIO") -> "Codec | None":
    """
    Detect codec from magic bytes of *src*.
    Detection is independent of current cursor; cursor is preserved.
    """
    from ..buffer.bytes_io import BytesIO
    from .mime_type import MimeType

    fh = BytesIO.wrap(src)
    header = fh.head(64)
    return _codec_from_mime(MimeType.parse_magic(header))


def detect_bytes(b: bytes) -> "Codec | None":
    """Detect codec from leading bytes of *b*."""
    from .mime_type import MimeType
    return _codec_from_mime(MimeType.parse_magic(b))