# yggdrasil/io/buffer/bytes_io.py
"""Spill-to-disk byte buffer with transparent memory/file backing.

This module provides :class:`BytesIO`, a file-like buffer that begins
in-process (backed by :class:`io.BytesIO`) and automatically migrates to a
temporary file on disk once the buffered data exceeds a configurable
threshold.  After a spill the public API is completely unchanged — callers
interact with the same ``read`` / ``write`` / ``seek`` / ``tell`` interface
regardless of which backing store is active.

The class is designed as the primary I/O primitive for the yggdrasil pipeline,
where payloads range from small JSON blobs (a few KB) to large Parquet shards
(hundreds of MB).  The spill mechanism ensures that memory pressure from large
payloads is bounded while keeping the common fast path (small buffers) fully
in-process.

Typical usage
-------------
Empty buffer, write data, read back::

    buf = BytesIO()
    buf.write(b"TTF 2024-Q3 settlement prices")
    buf.seek(0)
    print(buf.read())

Initialise from bytes or a file::

    buf = BytesIO(b"PAR1..." )          # from raw bytes
    buf = BytesIO(Path("/tmp/data.zst")) # file-backed, no in-memory phase

Polars integration::

    df = buf.read_polars()              # auto-detects format + codec
    buf.write_polars(df, MediaTypes.PARQUET)

Hashing::

    digest = buf.blake3().hexdigest()
    digest = buf.xxh3_64().hexdigest()

Context manager::

    with BytesIO(raw) as buf:
        df = buf.read_polars()
        # spill file (if any) deleted on exit
"""

from __future__ import annotations

import io
import mmap
import os
import shutil
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Any, IO, TYPE_CHECKING, Optional

from yggdrasil.io.config import BufferConfig, DEFAULT_CONFIG
from yggdrasil.io.enums.codec import Codec
from yggdrasil.io.enums.media_type import MediaType, MediaTypes
from yggdrasil.io.path import AbstractDataPath
from yggdrasil.io.types import BytesLike

if TYPE_CHECKING:
    import xxhash
    import blake3
    import polars

__all__ = ["BytesIO"]


class BytesIO(io.RawIOBase):
    """File-like byte buffer that spills to a temp file when it grows large.

    The buffer starts in memory (backed by :class:`io.BytesIO`) and
    automatically migrates to a temporary file on disk once the amount of
    buffered data exceeds :attr:`BufferConfig.spill_bytes`.  After a spill the
    public API is unchanged — ``read``, ``write``, ``seek``, ``tell`` all
    delegate to the underlying file handle.

    Exactly one of two internal states is active at any given time:

    * **In-memory** — ``_mem`` is a live :class:`io.BytesIO`; ``_file`` and
      ``_path`` are ``None``.
    * **Spilled** — ``_file`` is an open ``w+b`` file handle; ``_mem`` is
      ``None``; ``_path`` holds the :class:`Path` to the backing file.

    Parameters
    ----------
    data:
        Optional initial content.  Accepted types:

        ``None`` *(default)*
            Create an empty, in-memory buffer.
        ``bytes | bytearray | memoryview``
            Initialise with a bytes-like object.  Spills immediately if the
            payload exceeds the threshold.
        ``io.BytesIO``
            Snapshot the contents without disturbing the caller's cursor.
            Spills if the snapshot is over-threshold.
        ``Path``
            Open an existing (or new) file as the backing store directly —
            no in-memory phase.
        ``file-like`` (has ``.read()``)
            Stream the contents in.  If seekable, the size is measured first
            to decide memory vs. spill; non-seekable streams are drained
            fully before the decision is made.
        ``BytesIO``
            Shallow-alias the other instance's internal state.
            **Warning**: both objects share the same underlying buffer / file;
            use with care.
    config:
        :class:`BufferConfig` instance.  Falls back to :data:`DEFAULT_CONFIG`.

    Notes
    -----
    * The class is named ``BytesIO`` to be a transparent drop-in for
      :class:`io.BytesIO`; it deliberately shadows the stdlib name within this
      module.
    * :meth:`memoryview` also shadows the builtin, but is kept for API
      compatibility.  Use ``import builtins; builtins.memoryview(...)`` if you
      need the builtin inside this module.

    Examples
    --------
    >>> buf = BytesIO(b"hello")
    >>> buf.read()
    b'hello'

    >>> buf = BytesIO()
    >>> buf.write(b"Brent close")
    11
    >>> buf.seek(0); buf.read()
    b'Brent close'
    """

    def __init__(
        self,
        data: Any = None,
        *,
        config: BufferConfig | None = None,
    ) -> None:
        super().__init__()

        self._cfg: BufferConfig = config or DEFAULT_CONFIG

        # Exactly one of (_mem) or (_file + _path) is active at any time.
        self._mem:    io.BytesIO | None   = None
        self._file:   IO[bytes]  | None   = None
        self._path:   AbstractDataPath | None = None
        self._mmap:   mmap.mmap  | None   = None
        self._closed: bool                = False

        self._init_from(data)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BytesIO":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Initialisation dispatch
    # ------------------------------------------------------------------

    def _init_from(self, data: Any) -> None:
        """Route *data* to the appropriate initialisation helper.

        Dispatch order matters: :class:`io.BytesIO` is checked before the
        generic ``hasattr(data, "read")`` branch because it is seekable and
        has a known size, allowing a cheaper initialisation path.
        """
        if data is None:
            self._mem = io.BytesIO()

        elif isinstance(data, (bytes, bytearray, memoryview)):
            self._load_bytes(bytes(data) if not isinstance(data, bytes) else data)

        elif isinstance(data, io.BytesIO):
            self._init_from_stdlib_bytesio(data)

        elif isinstance(data, Path):
            self._init_from_path(data)

        elif isinstance(data, BytesIO):
            # Shallow alias — both instances share the same backing store.
            self._mem  = data._mem
            self._path = data._path
            self._file = data._file

        elif hasattr(data, "read"):
            self._init_from_filelike(data)

        else:
            raise TypeError(
                f"{type(self).__name__} does not accept data of type {type(data)!r}. "
                "Pass bytes, bytearray, memoryview, BytesIO, "
                "a file-like object, or a Path."
            )

    def _init_from_stdlib_bytesio(self, src: io.BytesIO) -> None:
        """Snapshot a stdlib :class:`io.BytesIO` without mutating its cursor.

        Measures the total size before copying so large payloads are spilled
        directly without an intermediate in-memory allocation.
        """
        saved_pos = src.tell()
        src.seek(0, io.SEEK_END)
        size = src.tell()
        src.seek(0)

        if size > self._cfg.spill_bytes:
            path, fh = self._open_spill_file()
            shutil.copyfileobj(src, fh, length=self._cfg.spill_bytes)
            fh.flush()
            self._path = path
            self._file = fh
        else:
            self._mem = io.BytesIO(src.read())

        src.seek(saved_pos)

    def _init_from_path(self, path: Path) -> None:
        """Use *path* directly as the backing store (no in-memory phase).

        Opens in ``r+b`` mode when the file already exists, ``w+b`` otherwise.
        """
        self._path = path
        self._file = path.open("r+b") if path.exists() else path.open("w+b")

    def _init_from_filelike(self, src: Any) -> None:
        """Initialise from a generic file-like object.

        Seekable sources are sized first to avoid an unnecessary intermediate
        buffer; non-seekable sources (pipes, network streams) are fully
        drained before the memory-vs-spill decision is made.
        """
        if hasattr(src, "seek") and hasattr(src, "tell"):
            saved_pos = src.tell()
            src.seek(0, io.SEEK_END)
            size = src.tell() - saved_pos
            src.seek(saved_pos)

            if size > self._cfg.spill_bytes:
                path, fh = self._open_spill_file()
                shutil.copyfileobj(src, fh, length=self._cfg.spill_bytes)
                fh.flush()
                self._path = path
                self._file = fh
            else:
                self._mem = io.BytesIO(src.read())
        else:
            self._load_bytes(src.read())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_bytes(self, payload: bytes) -> None:
        """Store *payload* in memory or spill immediately based on its size.

        Must only be called during ``__init__`` — assumes no existing
        internal state needs to be torn down.
        """
        if len(payload) > self._cfg.spill_bytes:
            path, fh = self._open_spill_file()
            fh.write(payload)
            fh.flush()
            self._path = path
            self._file = fh
        else:
            self._mem = io.BytesIO(payload)

    def _open_spill_file(self) -> tuple[AbstractDataPath, IO[bytes]]:
        """Create and open a new uniquely-named temporary spill file.

        The filename incorporates a UUID4 hex string to prevent collisions
        between concurrent buffers sharing the same *tmp_dir*.

        Returns
        -------
        tuple[AbstractDataPath, IO[bytes]]
            ``(path, file_handle)`` where the handle is opened in ``w+b``
            mode, supporting both reads and writes on the same descriptor.
        """
        tmp_dir = self._cfg.tmp_dir
        name = f"{self._cfg.prefix}{uuid.uuid4().hex}{self._cfg.suffix}"
        spill_path: AbstractDataPath = (tmp_dir / name) if tmp_dir is not None else (
            Path(tempfile.gettempdir()) / name
        )
        return spill_path, spill_path.open("w+b")

    def _spill_to_file(self) -> None:
        """Migrate in-memory contents to a temporary file.

        Called automatically by :meth:`write` when a pending write would push
        the in-memory buffer over :attr:`BufferConfig.spill_bytes`.  The
        current cursor position is preserved across the migration.

        No-op when the buffer is already spilled (``_mem`` is ``None``).
        """
        if self._mem is None:
            return

        cfg = self._cfg
        tmp_dir = str(cfg.tmp_dir) if cfg.tmp_dir is not None else None
        fd, name = tempfile.mkstemp(prefix=cfg.prefix, suffix=cfg.suffix, dir=tmp_dir)
        path = Path(name)
        fh = os.fdopen(fd, "w+b", buffering=0)

        mem = self._mem
        saved_pos = mem.tell()
        mem.seek(0)
        fh.write(mem.read())
        fh.flush()
        fh.seek(saved_pos)

        try:
            mem.close()
        except Exception:
            pass

        self._mem  = None
        self._file = fh
        self._path = path

    @staticmethod
    def _coerce_to_memoryview(b: Any) -> memoryview:
        """Coerce *b* to a :class:`memoryview`, handling common edge cases.

        Dispatch order:

        1. ``None`` → empty view (no-op write).
        2. Direct :class:`memoryview` construction (covers ``bytes``,
           ``bytearray``, ``memoryview``, ``array.array``, etc.).
        3. File-like objects → drained via ``.read()``.
        4. :class:`str` → UTF-8 encoded.

        Parameters
        ----------
        b:
            Value to coerce.

        Returns
        -------
        memoryview

        Raises
        ------
        TypeError
            When *b* cannot be coerced to a bytes-like object.
        """
        if b is None:
            return memoryview(b"")
        try:
            return memoryview(b)
        except TypeError:
            pass
        if hasattr(b, "read"):
            return memoryview(b.read())
        if isinstance(b, str):
            return memoryview(b.encode("utf-8"))
        raise TypeError(f"Cannot coerce {type(b)!r} to bytes-like")

    def buffer(self) -> IO[bytes]:
        """Return the active backing I/O object (memory buffer or file handle).

        This is the single choke-point through which all delegated I/O flows.
        Prefer the public ``read`` / ``write`` / ``seek`` / ``tell`` methods
        unless you need direct access to the underlying handle.

        Returns
        -------
        IO[bytes]
            :class:`io.BytesIO` when in-memory, otherwise the open spill-file
            handle.

        Raises
        ------
        ValueError
            If the buffer has been closed.
        RuntimeError
            If both ``_mem`` and ``_file`` are ``None`` (corrupted state).
        """
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if self._file is not None:
            return self._file
        if self._mem is not None:
            return self._mem
        raise RuntimeError("BytesIO is in an invalid state (no backing store)")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """``True`` when the buffer contains at least one byte."""
        return self.size > 0

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def __iter__(self):  # type: ignore[override]
        return iter(self.to_bytes())

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        if self._closed:
            return "<BytesIO [closed]>"
        state = "spilled" if self.spilled else "memory"
        return f"<BytesIO [{state}] size={self.size} bytes>"

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def parse_any(cls, obj: Any, config: BufferConfig | None = None) -> "BytesIO":
        """Create a :class:`BytesIO` from a variety of source types.

        Unlike the constructor, this method handles string paths and provides
        a zero-copy fast-path for objects that are already a :class:`BytesIO`.

        Parameters
        ----------
        obj:
            Source data.  Accepted types:

            :class:`BytesIO`
                Returned as-is (no copy).
            :class:`io.BytesIO`
                Aliased directly as the backing memory store (no copy).
            ``str | Path``
                Opened as a file-backed buffer in ``w+b`` mode.
            bytes-like or file-like
                Written into a new empty buffer; cursor is rewound to 0.
        config:
            Optional :class:`BufferConfig` override.

        Returns
        -------
        BytesIO
        """
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, io.BytesIO):
            buf = cls(config=config)
            buf._mem = obj
            return buf

        if isinstance(obj, (str, Path)):
            p = Path(obj)
            buf = cls(config=config)
            buf._path = p
            buf._file = p.open("w+b", buffering=0)
            buf._mem  = None
            return buf

        buf = cls(config=config)
        buf.write_any_bytes(obj)
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def spilled(self) -> bool:
        """``True`` when the buffer has migrated to a temporary file."""
        return self._file is not None

    @property
    def path(self) -> Path | None:
        """Filesystem path of the spill file, or ``None`` when in memory."""
        return self._path

    @property
    def size(self) -> int:
        """Current number of bytes held by the buffer.

        Uses :func:`os.fstat` for spilled buffers (O(1), no seek) and
        :meth:`io.BytesIO.getbuffer` for in-memory buffers (also O(1)).
        Falls back to a seek-based measurement only when ``fstat`` is
        unavailable (e.g. non-file-descriptor handles).
        """
        if self._mem is not None:
            return self._mem.getbuffer().nbytes
        fh = self.buffer()
        try:
            return os.fstat(fh.fileno()).st_size
        except Exception:
            pos = fh.tell()
            fh.seek(0, io.SEEK_END)
            end = fh.tell()
            fh.seek(pos)
            return int(end)

    @property
    def content_type(self) -> MediaType:
        """MIME type inferred from the buffer's magic bytes.

        Delegates to :meth:`MediaTypes.from_io`.  The buffer's cursor is not
        moved.

        Returns
        -------
        MediaType
            A frozen :class:`MediaType` describing the payload format.
            :attr:`MediaType.codec` carries any outer compression codec.

        Examples
        --------
        >>> buf = BytesIO(b"PAR1" + b"\\x00" * 100 + b"PAR1")
        >>> buf.content_type == MediaTypes.PARQUET
        True
        >>> buf2 = BytesIO(b'{"symbol": "TTF", "price": 42.5}')
        >>> buf2.content_type.mime
        'application/json'
        """
        return MediaType.from_io(self)

    # ------------------------------------------------------------------
    # io.RawIOBase interface
    # ------------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def flush(self) -> None:
        try:
            self.buffer().flush()
        except Exception:
            pass

    def tell(self) -> int:
        return self.buffer().tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.buffer().seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        return self.buffer().read(size)

    def write(self, b: Any) -> int:
        """Write *b* to the buffer, spilling to disk if the threshold is crossed.

        Accepts any type supported by :meth:`_coerce_to_memoryview` —
        ``bytes``, ``bytearray``, ``memoryview``, ``str`` (UTF-8 encoded),
        file-likes (drained), or ``None`` (no-op).

        Parameters
        ----------
        b:
            Data to write.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        ValueError
            If the buffer is closed.
        """
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        mv = self._coerce_to_memoryview(b)
        n = len(mv)
        if n == 0:
            return 0

        if self._mem is not None:
            if self._mem.getbuffer().nbytes + n > self._cfg.spill_bytes:
                self._spill_to_file()

        return self.buffer().write(mv.tobytes())

    def write_any_bytes(self, obj: Any) -> int:
        """Write a bytes-like or file-like object into the buffer.

        Thin wrapper over :meth:`write` that also handles file-like objects
        whose ``.read()`` returns bytes.

        Parameters
        ----------
        obj:
            ``bytes | bytearray | memoryview`` written directly, or any
            object with a ``.read()`` method whose result is written.
            ``None`` is a no-op.

        Returns
        -------
        int
            Number of bytes written.

        Raises
        ------
        TypeError
            For unsupported types.
        """
        if obj is None:
            return 0
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return self.write(obj)
        if hasattr(obj, "read"):
            return self.write(obj.read())
        raise TypeError(f"Unsupported type for write_any_bytes: {type(obj)!r}")

    # ------------------------------------------------------------------
    # Structured binary I/O — little-endian
    # ------------------------------------------------------------------
    # Each read_* / write_* pair encodes a single fixed-width scalar value.
    # All integers are little-endian two's complement; floats are IEEE 754.
    # The naming mirrors Rust's byteorder / binrw conventions so that the
    # Python and Rust codec implementations stay in sync.

    def _read_exact(self, n: int) -> bytes:
        """Read exactly *n* bytes or raise :class:`EOFError`.

        Parameters
        ----------
        n:
            Number of bytes required.

        Raises
        ------
        EOFError
            When fewer than *n* bytes remain in the stream.
        """
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    def read_int8(self) -> int:
        return struct.unpack("<b", self._read_exact(1))[0]

    def write_int8(self, value: int) -> int:
        return self.write(struct.pack("<b", int(value)))

    def read_uint8(self) -> int:
        return struct.unpack("<B", self._read_exact(1))[0]

    def write_uint8(self, value: int) -> int:
        return self.write(struct.pack("<B", int(value)))

    def read_int16(self) -> int:
        return struct.unpack("<h", self._read_exact(2))[0]

    def write_int16(self, value: int) -> int:
        return self.write(struct.pack("<h", int(value)))

    def read_uint16(self) -> int:
        return struct.unpack("<H", self._read_exact(2))[0]

    def write_uint16(self, value: int) -> int:
        return self.write(struct.pack("<H", int(value)))

    def read_int32(self) -> int:
        return struct.unpack("<i", self._read_exact(4))[0]

    def write_int32(self, value: int) -> int:
        return self.write(struct.pack("<i", int(value)))

    def read_uint32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def write_uint32(self, value: int) -> int:
        return self.write(struct.pack("<I", int(value)))

    def read_int64(self) -> int:
        return struct.unpack("<q", self._read_exact(8))[0]

    def write_int64(self, value: int) -> int:
        return self.write(struct.pack("<q", int(value)))

    def read_uint64(self) -> int:
        return struct.unpack("<Q", self._read_exact(8))[0]

    def write_uint64(self, value: int) -> int:
        return self.write(struct.pack("<Q", int(value)))

    def read_f32(self) -> float:
        return struct.unpack("<f", self._read_exact(4))[0]

    def write_f32(self, value: float) -> int:
        return self.write(struct.pack("<f", float(value)))

    def read_f64(self) -> float:
        return struct.unpack("<d", self._read_exact(8))[0]

    def write_f64(self, value: float) -> int:
        return self.write(struct.pack("<d", float(value)))

    def read_bool(self) -> bool:
        """Read a single byte and return ``True`` if non-zero."""
        return bool(self.read_uint8())

    def write_bool(self, value: bool) -> int:
        """Write *value* as a single byte (``0x01`` or ``0x00``)."""
        return self.write_uint8(1 if value else 0)

    def read_bytes_u32(self) -> bytes:
        """Read a uint32 length prefix followed by that many raw bytes."""
        return self._read_exact(self.read_uint32())

    def write_bytes_u32(self, data: BytesLike) -> int:
        """Write *data* preceded by its uint32 byte-length.

        Returns
        -------
        int
            Total bytes written (4 header bytes + payload length).
        """
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write(mv)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        """Read a uint32 length-prefixed string."""
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        """Write *s* as a uint32 length-prefixed encoded byte sequence."""
        return self.write_bytes_u32(s.encode(encoding))

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def xxh3_64(self) -> "xxhash.xxh3_64":
        """Return an ``xxh3_64`` hash object seeded with the buffer contents.

        Uses a :meth:`memoryview` snapshot for both in-memory and spilled
        buffers — the mmap path avoids an extra heap copy for large files.

        Returns
        -------
        xxhash.xxh3_64
            A finalised hash object; call ``.hexdigest()`` or ``.intdigest()``.
        """
        from yggdrasil.xxhash import xxh3_64
        h = xxh3_64()
        h.update(self.memoryview())
        return h

    def blake3(self) -> "blake3.blake3":
        """Return a ``blake3`` hash object seeded with the buffer contents.

        Selects the most efficient update strategy based on backing store:

        * **In-memory** — ``h.update(getbuffer())`` — zero-copy view.
        * **Spilled with path** — ``h.update_mmap(path)`` — OS-level mmap,
          avoids loading the file into Python heap.
        * **Spilled without path** — chunked ``h.update()`` in 8 MiB blocks.

        Returns
        -------
        blake3.blake3
            A finalised hash object; call ``.hexdigest()`` or ``.digest()``.
        """
        from yggdrasil.blake3 import blake3
        h = blake3(max_threads=blake3.AUTO)

        if self._mem is not None:
            h.update(self._mem.getbuffer())
            return h

        if self._path is not None:
            h.update_mmap(str(self._path))
            return h

        fh = self.buffer()
        fh.seek(0)
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
        return h

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Decode the buffer's entire contents to a string.

        Parameters
        ----------
        encoding:
            Text encoding.  Defaults to ``"utf-8"``.
        errors:
            Error handler forwarded to :meth:`bytes.decode`.  Defaults to
            ``"replace"`` so corrupt bytes never raise unexpectedly.

        Returns
        -------
        str
            Empty string when the buffer is empty.

        Raises
        ------
        UnicodeDecodeError
            Only when *errors* is ``"strict"`` and the content is not valid
            *encoding*.

        Examples
        --------
        >>> BytesIO(b'{"price": 42.5}').decode()
        '{"price": 42.5}'
        """
        if not self:
            return ""
        return self.to_bytes().decode(encoding, errors)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def getvalue(self) -> bytes:
        """Return the entire buffer as :class:`bytes`.

        For in-memory buffers this is a zero-seek snapshot via
        :meth:`io.BytesIO.getvalue`.  For spilled buffers the file is read
        into memory in full — prefer :meth:`open_reader` to stream large
        spilled payloads instead.
        """
        if self._mem is not None:
            return self._mem.getvalue()
        return self.to_bytes()

    def memoryview(self) -> memoryview:  # noqa: A003 — intentional shadowing for API compat
        """Return a :class:`memoryview` over the buffer's contents.

        In-memory
            A view over :meth:`io.BytesIO.getvalue` — one internal copy, but
            no additional copy relative to the public API.
        Spilled
            An ``mmap``-backed read-only view of the spill file.  The mmap is
            cached in ``_mmap`` and reused across calls until the buffer is
            closed; creating it is O(1) from the OS perspective.

        Notes
        -----
        This method deliberately shadows the builtin :class:`memoryview`.
        Use ``import builtins; builtins.memoryview(...)`` if you need the
        builtin inside this module.
        """
        if self._mem is not None:
            return memoryview(self._mem.getvalue())

        fh = self.buffer()
        size = os.fstat(fh.fileno()).st_size
        if size == 0:
            return memoryview(b"")

        if self._mmap is None or self._mmap.closed:
            self._mmap = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ)
        return memoryview(self._mmap)

    def to_bytes(self) -> bytes:
        """Read the entire buffer from position 0 and return it as :class:`bytes`.

        The caller's cursor position is preserved.
        """
        fh = self.buffer()
        saved = fh.tell()
        try:
            fh.seek(0)
            return fh.read()
        finally:
            try:
                fh.seek(saved)
            except Exception:
                pass

    def open_reader(self) -> IO[bytes]:
        """Return a fresh, independent read-only file-like over the contents.

        The returned handle has its own cursor starting at 0 and must be
        closed by the caller.  It is completely decoupled from the buffer's
        own cursor.

        Returns
        -------
        IO[bytes]
            :class:`io.BytesIO` snapshot for in-memory buffers; an ``rb``
            file handle for spilled buffers.

        Raises
        ------
        RuntimeError
            If the buffer has spilled but ``_path`` is ``None`` (unexpected
            internal state).
        """
        if self._mem is not None:
            return io.BytesIO(self._mem.getvalue())
        if self._path is None:
            raise RuntimeError("Spilled buffer has no path (unexpected state)")
        return self._path.open("rb")

    def open_writer(self) -> IO[bytes]:
        """Return the underlying writable file handle directly.

        Writes through this handle bypass :meth:`write`'s spill-threshold
        check.  Use :meth:`write` for normal appending.
        """
        return self.buffer()

    # ------------------------------------------------------------------
    # Polars integration
    # ------------------------------------------------------------------

    def read_polars(
        self,
        content_type: "MediaType | str | None" = None,
        *,
        raise_error: bool = True,
        lazy: bool = False,
    ) -> "polars.DataFrame | polars.LazyFrame":
        """Deserialise the buffer into a Polars DataFrame (or LazyFrame).

        The format and outer compression codec are inferred automatically from
        the buffer's magic bytes unless *content_type* is supplied explicitly.
        When a codec is detected the payload is transparently decompressed
        before parsing.

        Parameters
        ----------
        content_type:
            Override automatic format detection.  Useful when the buffer
            contains raw CSV or JSON without a recognisable magic header.
        raise_error:
            Raise error on reading content
        lazy:
            When ``True`` return a :class:`polars.LazyFrame` instead of a
            :class:`polars.DataFrame`.  Parquet and IPC/Arrow use
            ``scan_parquet`` / ``scan_ipc`` when a filesystem path is
            available; all other formats are read eagerly and wrapped with
            ``.lazy()``.

        Returns
        -------
        polars.DataFrame | polars.LazyFrame

        Raises
        ------
        ValueError
            For unsupported or unrecognised :class:`MediaType` values.

        Examples
        --------
        >>> buf = BytesIO(b"a,b\\n1,2\\n3,4")
        >>> buf.read_polars(MediaTypes.CSV)
        shape: (2, 2) ...
        """
        from yggdrasil.polars.lib import polars as pl

        ct = MediaType.parse_any(content_type, default=self.content_type)

        # Decompress outer codec wrapper so every format branch gets a plain
        # uncompressed IO[bytes] — codec awareness lives here, not in each branch.
        src: IO[bytes]
        if ct.codec is not None:
            src = ct.codec.open(self)   # returns seekable BytesIO at offset 0
        else:
            src = self.open_reader()    # independent cursor, must be closed

        try:
            fmt = ct.without_codec()   # strip codec for format comparison

            if fmt == MediaTypes.PARQUET:
                if lazy:
                    if self._path is not None and ct.codec is None:
                        return pl.scan_parquet(self._path)
                    return pl.read_parquet(src).lazy()
                return pl.read_parquet(src)

            if fmt in (MediaTypes.IPC, MediaTypes.FEATHER):
                if lazy:
                    if self._path is not None and ct.codec is None:
                        return pl.scan_ipc(self._path)
                    return pl.read_ipc(src).lazy()
                return pl.read_ipc(src)

            if fmt == MediaTypes.CSV:
                df = pl.read_csv(src)
                return df.lazy() if lazy else df

            if fmt == MediaTypes.JSON:
                df = pl.read_json(src)
                return df.lazy() if lazy else df

            if fmt == MediaTypes.NDJSON:
                df = pl.read_ndjson(src)
                return df.lazy() if lazy else df

            if fmt == MediaTypes.AVRO:
                df = pl.read_avro(src)
                return df.lazy() if lazy else df

            if fmt == MediaTypes.ZIP:
                import zipfile
                with zipfile.ZipFile(src) as zf:
                    if not (names := zf.namelist()):
                        if raise_error:
                            raise ValueError("ZIP archive is empty")
                        return pl.DataFrame([], schema={})

                    frames = [
                        BytesIO(zf.read(name)).read_polars(
                            raise_error=raise_error,
                            lazy=lazy
                        )
                        for name in names
                    ]

                    return pl.concat(
                        frames,
                        rechunk=False,
                        how="diagonal_relaxed"
                    )

            raise ValueError(
                f"read_polars: unsupported MediaType {ct!r}. "
                "Pass an explicit content_type= to override detection."
            )
        finally:
            src.close()

    def write_polars(
        self,
        df: "polars.DataFrame | polars.LazyFrame",
        content_type: "MediaType | str | None" = None,
        *,
        codec: Optional[Codec] = None,
        compression: str = "zstd",
        row_group_size: Optional[int] = None,
    ) -> int:
        """Serialise *df* into this buffer using the specified format.

        :class:`polars.LazyFrame` inputs are collected before serialisation.
        The result is always appended at the current cursor position.

        Parameters
        ----------
        df:
            DataFrame or LazyFrame to serialise.
        content_type:
            Target format.  Defaults to :attr:`MediaType.PARQUET`.
        codec:
            Optional outer compression wrapper applied **after** internal
            serialisation.  Distinct from Parquet's built-in *compression*
            argument — prefer Parquet's native ZSTD/Snappy over this for
            columnar data.
        compression:
            Internal compression passed to Polars for Parquet and IPC writes.
            Ignored for CSV / JSON / NDJSON / Avro.
        row_group_size:
            Parquet row-group size hint forwarded to
            :func:`polars.DataFrame.write_parquet`.

        Returns
        -------
        int
            Number of bytes written into this buffer.

        Raises
        ------
        ValueError
            For unsupported :class:`MediaType` values.

        Examples
        --------
        >>> import polars as pl
        >>> buf = BytesIO()
        >>> buf.write_polars(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
        >>> buf.seek(0)
        >>> buf.read_polars().shape
        (2, 2)
        """
        from yggdrasil.polars.lib import polars as pl

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        mt = MediaType.parse_any(content_type, default=MediaTypes.PARQUET)
        fmt = mt.without_codec()
        sink = io.BytesIO()

        if fmt == MediaTypes.PARQUET:
            kw: dict[str, Any] = {"compression": compression}
            if row_group_size is not None:
                kw["row_group_size"] = row_group_size
            df.write_parquet(sink, **kw)

        elif fmt in (MediaTypes.IPC, MediaTypes.FEATHER):
            df.write_ipc(sink, compression=compression)

        elif fmt == MediaTypes.CSV:
            df.write_csv(sink)

        elif fmt == MediaTypes.JSON:
            df.write_json(sink)

        elif fmt == MediaTypes.NDJSON:
            df.write_ndjson(sink)

        elif fmt == MediaTypes.AVRO:
            df.write_avro(sink)

        else:
            raise ValueError(
                f"write_polars: unsupported MediaType {mt!r}. "
                "Supported: PARQUET, ARROW/IPC/FEATHER, CSV, JSON, NDJSON, AVRO."
            )

        payload = Codec.compress_with(sink.getvalue(), codec)
        n = self.write(payload)
        self.flush()
        return n

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Alias for :meth:`close` — provided for explicit resource teardown."""
        self.close()

    def close(self) -> None:
        """Release all resources held by the buffer.

        Teardown order:

        1. Flush the backing store.
        2. Close the mmap (must precede the file close on Windows, where an
           open mmap pins the file handle).
        3. Close the in-memory or file backing store.
        4. Delete the spill file unless
           :attr:`BufferConfig.keep_spilled_file` is ``True``.

        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return

        try:
            self.flush()
        except Exception:
            pass

        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

        if self._mem is not None:
            try:
                self._mem.close()
            except Exception:
                pass
            self._mem = None

        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        if self._path is not None and not self._cfg.keep_spilled_file:
            try:
                self._path.unlink(missing_ok=True)
            except Exception:
                pass

        self._closed = True
        super().close()