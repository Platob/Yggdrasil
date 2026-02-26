# yggdrasil/io/enums/codec.py
"""Compression codec detection, compression, and decompression from magic bytes.

This module provides :class:`Codec`, an enum whose members represent the
compression codecs supported by the yggdrasil I/O layer.  Each member exposes
a symmetric compress / decompress API that operates on either raw :class:`bytes`
or seekable binary streams (:class:`~yggdrasil.pyutils.dynamic_buffer.BytesIO`
/ :class:`io.BytesIO`).

Supported codecs
----------------
+----------+------------------------------------------+------------+
| Member   | Magic bytes                              | Dependency |
+==========+==========================================+============+
| GZIP     | ``\\x1f\\x8b``                           | stdlib     |
| ZSTD     | ``\\x28\\xb5\\x2f\\xfd``                 | zstandard  |
| LZ4      | ``\\x04\\x22\\x4d\\x18`` /               | lz4        |
|          | ``\\x02\\x21\\x4c\\x18``                 |            |
| BZIP2    | ``BZh``                                  | stdlib     |
| XZ       | ``\\xfd7zXZ\\x00``                       | stdlib     |
| SNAPPY   | ``\\xff\\x06\\x00\\x00sNaPpY``           | cramjam    |
+----------+------------------------------------------+------------+

Third-party dependencies (``zstandard``, ``lz4``, ``cramjam``) are imported
lazily and installed automatically at runtime via
:func:`~yggdrasil.environ.runtime_import_module` if they are not already
present in the active environment.

Typical usage
-------------
Detection::

    codec = Codec.from_io(stream)          # None if uncompressed
    codec = Codec.from_bytes(raw_bytes)

Compression / decompression (stream API)::

    compressed_buf = Codec.ZSTD.compress(my_bytesio)
    decompressed_buf = Codec.ZSTD.open(compressed_bytesio)

Compression / decompression (bytes API)::

    blob = Codec.ZSTD.compress_bytes(raw)
    raw  = Codec.ZSTD.decompress_bytes(blob)

Conditional helpers (codec may be ``None``)::

    out = Codec.compress_with(raw, user_codec)    # passthrough when None
    out = Codec.decompress_with(blob, user_codec)
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from ..buffer import BytesIO

__all__ = ["Codec", "_peek", "_peek_buf"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _runtime_import(module_name: str, pip_name: str):
    """Import *module_name*, installing *pip_name* on first-time miss.

    Uses :func:`importlib.import_module` as the hot path (no exception on
    cache hit) and delegates to
    :func:`~yggdrasil.environ.runtime_import_module` only when the package is
    genuinely absent.

    Parameters
    ----------
    module_name:
        Fully qualified module name passed to :func:`importlib.import_module`
        (e.g. ``"lz4.frame"``).
    pip_name:
        PyPI distribution name used for the ``pip install`` fallback
        (e.g. ``"lz4"``).

    Returns
    -------
    types.ModuleType
        The imported module.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        from yggdrasil.environ import runtime_import_module
        return runtime_import_module(
            module_name=module_name,
            pip_name=pip_name,
            install=True,
        )


def _peek(src: IO[bytes], n: int) -> bytes:
    """Read up to *n* bytes from *src* at its current position, then seek back.

    The caller's cursor is always restored, even when fewer than *n* bytes are
    available (short stream, EOF, etc.).

    Parameters
    ----------
    src:
        Any seekable binary stream.
    n:
        Maximum number of bytes to read.

    Returns
    -------
    bytes
        Between 0 and *n* bytes starting at the original cursor position.
    """
    pos = src.tell()
    try:
        return src.read(n)
    finally:
        src.seek(pos)


def _peek_buf(buf: "BytesIO", n: int) -> bytes:
    """Cursor-safe peek for a :class:`~dynamic_buffer.BytesIO` instance.

    Delegates to :func:`_peek` on the underlying :meth:`~BytesIO.buffer`.

    Parameters
    ----------
    buf:
        A :class:`~yggdrasil.pyutils.dynamic_buffer.BytesIO` instance.
    n:
        Maximum number of bytes to read.

    Returns
    -------
    bytes
        Between 0 and *n* bytes starting at the current cursor position.
    """
    return _peek(buf.buffer(), n)


def _drain(src: "IO[bytes] | BytesIO") -> bytes:
    """Read the entire contents of *src* from its current position.

    Works for both plain :class:`io.BytesIO` and yggdrasil
    :class:`~dynamic_buffer.BytesIO` (duck-typed via ``.buffer()``).  The
    caller's cursor position is **preserved**.

    Parameters
    ----------
    src:
        Any seekable binary stream or :class:`~dynamic_buffer.BytesIO`.

    Returns
    -------
    bytes
        All bytes from the current position to EOF.
    """
    fh: IO[bytes] = src.buffer() if hasattr(src, "buffer") else src  # type: ignore[union-attr]
    pos = fh.tell()
    try:
        return fh.read()
    finally:
        fh.seek(pos)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

class Codec(str, Enum):
    """Compression codec with symmetric compress / decompress operations.

    Each member's value is a canonical lowercase string usable as a file
    extension or protocol label (e.g. ``"zstd"``, ``"gzip"``).

    The class exposes four levels of API, ordered from highest to lowest
    abstraction:

    1. **Stream API** — :meth:`compress` / :meth:`open`
       Accept and return :class:`~dynamic_buffer.BytesIO` instances.
       Cursor positions are preserved on both input and output.

    2. **Bytes API** — :meth:`compress_bytes` / :meth:`decompress_bytes`
       Accept and return plain :class:`bytes`.  No I/O overhead.

    3. **Conditional classmethods** — :meth:`compress_with` / :meth:`decompress_with`
       Accept an ``Optional[Codec]`` and pass data through unchanged when the
       codec is ``None``.  Ideal for configurable pipelines.

    4. **Detection classmethods** — :meth:`from_io` / :meth:`from_bytes`
       Infer the codec from magic bytes without consuming the stream.

    Members
    -------
    GZIP   : stdlib ``gzip``
    ZSTD   : ``zstandard`` (auto-installed)
    LZ4    : ``lz4`` (auto-installed)
    BZIP2  : stdlib ``bz2``
    XZ     : stdlib ``lzma``
    SNAPPY : ``cramjam`` (auto-installed) — framed format only
    """

    GZIP   = "gzip"
    ZSTD   = "zstd"
    LZ4    = "lz4"
    BZIP2  = "bzip2"
    XZ     = "xz"
    SNAPPY = "snappy"

    # Populated after the class body; defined here for type-checker visibility.
    _MAGIC: "list[tuple[bytes, Codec]]"

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    @classmethod
    def from_io(cls, src: "IO[bytes] | BytesIO") -> "Codec | None":
        """Detect the compression codec by inspecting *src*'s magic bytes.

        Reads up to 16 bytes from the current cursor position without
        consuming the stream (cursor is always restored).

        Parameters
        ----------
        src:
            Any seekable binary stream or
            :class:`~dynamic_buffer.BytesIO` instance.

        Returns
        -------
        Codec | None
            The matched :class:`Codec`, or ``None`` when no known magic
            sequence is present (uncompressed or unknown format).

        Examples
        --------
        >>> import io
        >>> Codec.from_io(io.BytesIO(b"\\x1f\\x8b" + b"\\x00" * 20))
        <Codec.GZIP: 'gzip'>
        >>> Codec.from_io(io.BytesIO(b"PAR1")) is None
        True
        """
        # Duck-type .buffer() to avoid a circular import (BytesIO imports Codec).
        fh: IO[bytes] = src.buffer() if hasattr(src, "buffer") else src  # type: ignore[union-attr]
        header = _peek(fh, 16)
        for magic, codec in cls._MAGIC:
            if header[: len(magic)] == magic:
                return codec
        return None

    @classmethod
    def from_bytes(cls, data: bytes) -> "Codec | None":
        """Detect the compression codec from a raw *data* bytes object.

        No I/O involved; operates purely on the leading bytes of *data*.

        Parameters
        ----------
        data:
            Bytes whose leading bytes may contain a compression magic
            sequence.

        Returns
        -------
        Codec | None
            The matched :class:`Codec`, or ``None`` for uncompressed data.

        Examples
        --------
        >>> import gzip
        >>> Codec.from_bytes(gzip.compress(b"hello"))
        <Codec.GZIP: 'gzip'>
        >>> Codec.from_bytes(b"PAR1") is None
        True
        """
        for magic, codec in cls._MAGIC:
            if data[: len(magic)] == magic:
                return codec
        return None

    # ------------------------------------------------------------------
    # Stream API  (BytesIO ↔ BytesIO)
    # ------------------------------------------------------------------

    def compress(self, src: "IO[bytes] | BytesIO") -> "BytesIO":
        """Compress *src* with this codec and return a new seekable buffer.

        The caller's cursor on *src* is **preserved**.  The returned buffer
        is positioned at offset 0 and fully supports random access.

        Parameters
        ----------
        src:
            Seekable binary stream or :class:`~dynamic_buffer.BytesIO`
            holding uncompressed data.

        Returns
        -------
        BytesIO
            A seekable buffer containing the compressed payload.

        Examples
        --------
        >>> import io
        >>> buf = io.BytesIO(b"TTF front-month price series" * 500)
        >>> out = Codec.ZSTD.compress(buf)
        >>> Codec.from_io(out)
        <Codec.ZSTD: 'zstd'>
        >>> out.tell()
        0
        """
        from ..buffer import BytesIO as _BytesIO
        result = _BytesIO(self.compress_bytes(_drain(src)))
        result.seek(0)
        return result

    def open(self, src: "IO[bytes] | BytesIO") -> "BytesIO":
        """Decompress *src* with this codec and return a new seekable buffer.

        The caller's cursor on *src* is **preserved**.  The returned buffer
        is positioned at offset 0 and fully supports random access.

        Parameters
        ----------
        src:
            Seekable binary stream or :class:`~dynamic_buffer.BytesIO`
            holding compressed data produced by this codec.

        Returns
        -------
        BytesIO
            A seekable buffer containing the decompressed payload.

        Raises
        ------
        ValueError
            If *self* is not a recognised codec (future-proofing for
            subclasses / dynamic enum extension).

        Examples
        --------
        >>> import io, gzip
        >>> compressed = io.BytesIO(gzip.compress(b"Brent prompt close"))
        >>> buf = Codec.GZIP.open(compressed)
        >>> buf.read()
        b'Brent prompt close'
        >>> buf.seek(0); buf.read(5)
        b'Brent'
        """
        from ..buffer import BytesIO as _BytesIO
        result = _BytesIO(self.decompress_bytes(_drain(src)))
        result.seek(0)
        return result

    # ------------------------------------------------------------------
    # Bytes API  (bytes ↔ bytes)
    # ------------------------------------------------------------------

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress *data* with this codec and return the compressed bytes.

        No I/O or cursor management.  Use :meth:`compress` when working with
        streams.

        Parameters
        ----------
        data:
            Raw uncompressed bytes.

        Returns
        -------
        bytes
            Compressed payload whose leading magic bytes identify this codec.

        Examples
        --------
        >>> payload = b"Henry Hub daily settle" * 2000
        >>> blob = Codec.LZ4.compress_bytes(payload)
        >>> Codec.from_bytes(blob)
        <Codec.LZ4: 'lz4'>
        >>> len(blob) < len(payload)
        True
        """
        match self:
            case Codec.GZIP:
                import gzip
                return gzip.compress(data)
            case Codec.ZSTD:
                return _runtime_import("zstandard", "zstandard").ZstdCompressor().compress(data)
            case Codec.LZ4:
                return _runtime_import("lz4.frame", "lz4").compress(data)
            case Codec.BZIP2:
                import bz2
                return bz2.compress(data)
            case Codec.XZ:
                import lzma
                return lzma.compress(data)
            case Codec.SNAPPY:
                return bytes(_runtime_import("cramjam", "cramjam").snappy.compress(data))
            case _:
                raise ValueError(f"compress_bytes: unhandled codec {self!r}")

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress *data* with this codec and return the raw bytes.

        No I/O or cursor management.  Use :meth:`open` when working with
        streams.

        Parameters
        ----------
        data:
            Compressed bytes produced by this codec.

        Returns
        -------
        bytes
            Decompressed payload.

        Examples
        --------
        >>> raw = b"WTI calendar spread"
        >>> Codec.BZIP2.decompress_bytes(Codec.BZIP2.compress_bytes(raw)) == raw
        True
        """
        match self:
            case Codec.GZIP:
                import gzip
                return gzip.decompress(data)
            case Codec.ZSTD:
                return _runtime_import("zstandard", "zstandard").ZstdDecompressor().decompress(data)
            case Codec.LZ4:
                return _runtime_import("lz4.frame", "lz4").decompress(data)
            case Codec.BZIP2:
                import bz2
                return bz2.decompress(data)
            case Codec.XZ:
                import lzma
                return lzma.decompress(data)
            case Codec.SNAPPY:
                return bytes(_runtime_import("cramjam", "cramjam").snappy.decompress(data))
            case _:
                raise ValueError(f"decompress_bytes: unhandled codec {self!r}")

    # ------------------------------------------------------------------
    # Conditional classmethods  (Optional[Codec] passthrough)
    # ------------------------------------------------------------------

    @classmethod
    def compress_with(
        cls,
        data: bytes,
        codec: "Codec | str | None",
    ) -> bytes:
        """Compress *data* with *codec*, or return it unchanged when *codec* is ``None``.

        Eliminates boilerplate ``if codec: ... else: data`` branches in
        configurable serialisation pipelines.

        Parameters
        ----------
        data:
            Raw uncompressed bytes.
        codec:
            A :class:`Codec` member, its string value (e.g. ``"zstd"``), or
            ``None`` to pass through without modification.

        Returns
        -------
        bytes

        Examples
        --------
        >>> Codec.compress_with(b"raw", None) == b"raw"
        True
        >>> Codec.compress_with(b"raw", "gzip") == Codec.GZIP.compress_bytes(b"raw")
        True
        """
        if codec is None:
            return data
        if isinstance(codec, str):
            codec = cls(codec)
        return codec.compress_bytes(data)

    @classmethod
    def decompress_with(
        cls,
        data: bytes,
        codec: "Codec | str | None",
    ) -> bytes:
        """Decompress *data* with *codec*, or return it unchanged when *codec* is ``None``.

        Symmetric counterpart to :meth:`compress_with` for encode / decode
        pipelines that share the same ``Optional[Codec]`` configuration value.

        Parameters
        ----------
        data:
            Compressed bytes, or raw bytes when *codec* is ``None``.
        codec:
            A :class:`Codec` member, its string value, or ``None``.

        Returns
        -------
        bytes

        Examples
        --------
        >>> raw = b"nat gas prompt settle"
        >>> Codec.decompress_with(Codec.compress_with(raw, "zstd"), "zstd") == raw
        True
        >>> Codec.decompress_with(raw, None) == raw
        True
        """
        if codec is None:
            return data
        if isinstance(codec, str):
            codec = cls(codec)
        return codec.decompress_bytes(data)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def roundtrip(self, data: bytes) -> bool:
        """Verify compress → decompress identity for *data*.

        Useful in tests and startup probes to confirm a runtime-installed
        library produces a lossless round-trip before trusting it with real
        tick or settlement data.

        Parameters
        ----------
        data:
            Arbitrary bytes to compress and decompress.

        Returns
        -------
        bool
            ``True`` iff ``decompress(compress(data)) == data``.

        Examples
        --------
        >>> Codec.ZSTD.roundtrip(b"Brent ICE close" * 1000)
        True
        """
        return self.decompress_bytes(self.compress_bytes(data)) == data


# ---------------------------------------------------------------------------
# Magic-byte table
# ---------------------------------------------------------------------------
# Populated after the class body so all Codec members are available.
# Entries are sorted longest-first so more-specific prefixes win over shorter
# ones (e.g. XZ's 6-byte magic beats GZIP's 2-byte magic on any prefix clash).

Codec._MAGIC = [
    (b"\xfd7zXZ\x00",           Codec.XZ),      # 6 bytes — most specific
    (b"\xff\x06\x00\x00sNaPpY", Codec.SNAPPY),  # 10 bytes framed
    (b"\x28\xb5\x2f\xfd",       Codec.ZSTD),    # 4 bytes
    (b"\x04\x22\x4d\x18",       Codec.LZ4),     # LZ4 frame magic v1
    (b"\x02\x21\x4c\x18",       Codec.LZ4),     # LZ4 frame magic v0
    (b"\x1f\x8b",               Codec.GZIP),    # 2 bytes
    (b"BZh",                    Codec.BZIP2),   # 3 bytes
]