"""Spill-to-disk byte buffer with transparent memory/path backing.

Design
------
- Own logical cursor (self._pos), independent from any OS/file cursor
- In-memory backing is bytearray + logical size
- Spilled backing is a path-like object (local ``Path`` or ``DatabricksPath``)
- Uses os.pread/os.pwrite for cursorless file IO
- Keeps optional readonly mmap for zero-copy reads when spilled
- A cached read file-handle avoids open()/close() per pread() call

Semantics
---------
- By default, construction is deep-copying for raw bytes/path/file-like inputs
- ``BytesIO(BytesIO(...), copy=True)`` duplicates content
- ``BytesIO(BytesIO(...), copy=False)`` aliases spilled-path backing only
- ``BytesIO(path, copy=True)`` copies file content into owned backing
- ``BytesIO(path, copy=False)`` aliases the original path and mutates it
- ``BytesIO(io.BytesIO(...))`` and other file-like sources read from the current cursor onward

Compression
-----------
- ``compress`` and ``decompress`` delegate to :class:`~yggdrasil.io.enums.codec.Codec`,
  which streams chunk-by-chunk when the codec supports it (gzip, zstd,
  lz4, bz2, xz, lzma) and falls back to a bytes roundtrip otherwise
  (snappy, brotli). Peak memory on streaming codecs is bounded by the
  codec's internal chunk size, not by the full payload size.
"""

from __future__ import annotations

import base64
import io
import mmap
import os
import shutil
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, IO, Literal, Optional, Union

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.io.config import DEFAULT_CONFIG, BufferConfig
from yggdrasil.io.enums import MediaType, MimeTypes
from yggdrasil.io.types import BytesLike

if TYPE_CHECKING:
    import blake3
    import xxhash

    from .media_io import MediaIO, MediaOptions
    from ..enums import Codec

__all__ = ["BytesIO", "BufferLike"]

BufferLike = Union[
    bytes,
    bytearray,
    memoryview,
    io.BytesIO,
    "BytesIO",
    str,
    Path,
    "DatabricksPath",
    IO[bytes],
]

SpillPath = "Path | DatabricksPath"

_HEAD_DEFAULT = 128
_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_PICKLE_COMPRESS_THRESHOLD_DEFAULT = 1 * 1024 * 1024
_HAS_PREAD = hasattr(os, "pread")
_HAS_PWRITE = hasattr(os, "pwrite")


def _as_contiguous_memoryview(mv: memoryview) -> memoryview:
    """Return a C-contiguous view over *mv*.

    ``memoryview.contiguous`` only exists from Python 3.12 on; use
    ``c_contiguous`` for portability.
    """
    return mv if mv.c_contiguous else memoryview(mv.tobytes())


class BytesIO(io.RawIOBase):
    __slots__ = (
        "_cfg",
        "_buf",
        "_size",
        "_pos",
        "_path",
        "_mmap",
        "_read_fh",
        "_closed",
        "_owns_path",
        "_media_type",
    )

    def __init__(
        self,
        data: IO[bytes] | bytes | bytearray | memoryview | str | SpillPath | None = None,
        *,
        media_type: Optional["MediaType"] = None,
        config: BufferConfig | None = None,
        copy: bool = False,
    ) -> None:
        super().__init__()
        self._cfg: BufferConfig = config or DEFAULT_CONFIG

        self._buf: bytearray | None = None
        self._size: int = 0
        self._pos: int = 0

        self._path: SpillPath | None = None
        self._mmap: mmap.mmap | None = None
        self._read_fh: IO[bytes] | None = None
        self._owns_path: bool = False
        self._media_type: MediaType | None = media_type

        self._closed: bool = False
        self._init_from(data, copy=copy)

    # ------------------------------------------------------------------
    # Dunder protocol
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return self.size > 0

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def __iter__(self):
        return iter(self.to_bytes())

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        if self._closed:
            return "<BytesIO [closed]>"
        state = "spilled" if self.spilled else "memory"
        return f"<BytesIO [{state}] size={self.size} bytes pos={self._pos}>"

    def __enter__(self) -> "BytesIO":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __getstate__(self):
        threshold = getattr(
            self._cfg,
            "pickle_compress_threshold",
            _PICKLE_COMPRESS_THRESHOLD_DEFAULT,
        )
        if self.size > threshold:
            from yggdrasil.io import ZSTD

            # compress(copy=True) now streams via Codec.compress() — the
            # resulting BytesIO may be spilled if large enough. We still
            # have to materialize to bytes for the pickle payload, but
            # at least the *compressed* size is bounded, not the input.
            blob = self.compress(codec=ZSTD, copy=True).to_bytes()
            codec = ZSTD.name
        else:
            blob = self.to_bytes()
            codec = None

        return {
            "data": blob,
            "codec": codec,
            "media_type": self._media_type,
        }

    def __setstate__(self, state):
        blob = state["data"]
        codec = state.get("codec")
        media_type = state.get("media_type")

        if codec is not None:
            from yggdrasil.io import Codec
            codec = Codec.parse(codec)
            blob = codec.decompress_bytes(blob)

        self.__init__(blob, copy=False, media_type=media_type)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_from(self, data: Any, *, copy: bool) -> None:
        if data is None:
            self._buf = bytearray()
            self._size = 0
            self._pos = 0
            self._path = None
            self._owns_path = False
            return

        if isinstance(data, (bytes, bytearray, memoryview)):
            self._init_from_bytes(memoryview(data))
            return

        if isinstance(data, io.BytesIO):
            self._init_from_stdlib_bytesio(data)
            return

        if self._is_pathish(data):
            self._init_from_path(self._coerce_path(data), copy=copy)
            return

        if isinstance(data, BytesIO):
            self._init_from_bytesio(data, copy=copy)
            return

        if hasattr(data, "read"):
            self._init_from_filelike(data)
            return

        raise TypeError(
            f"{type(self).__name__} does not accept data of type {type(data)!r}. "
            "Pass bytes/bytearray/memoryview, io.BytesIO, BytesIO, a file-like object, or a Path."
        )

    def _init_from_bytes(self, mv: memoryview) -> None:
        n = len(mv)
        if n > self._cfg.spill_bytes:
            self._spill_from_bytes(mv)
        else:
            src = _as_contiguous_memoryview(mv)
            self._buf = bytearray(src)
            self._size = n
            self._path = None
            self._owns_path = False
        self._pos = 0

    def _init_from_stdlib_bytesio(self, src: io.BytesIO) -> None:
        """Read from *src*'s current cursor to its end."""
        start_pos = src.tell()
        # getbuffer() avoids copying the whole payload just to probe size.
        total = len(src.getbuffer())
        remaining = max(0, total - start_pos)

        if remaining > self._cfg.spill_bytes:
            # Stream directly to disk — never materialize the full
            # remainder as a bytearray first.
            path = self._cfg.create_spill_path()
            with path.open("w+b") as fh:
                # Use readinto loop against getbuffer() slices instead of
                # copying via src.read() to avoid the interpreter-level
                # bytes copy.
                view = memoryview(src.getbuffer())[start_pos:]
                offset = 0
                chunk = _COPY_CHUNK_SIZE
                while offset < remaining:
                    end = min(offset + chunk, remaining)
                    fh.write(view[offset:end])
                    offset = end
                fh.flush()
            self._buf = None
            self._size = 0
            self._path = path
            self._owns_path = True
        else:
            # Small enough to hold in memory — slice from the current cursor.
            self._buf = bytearray(memoryview(src.getbuffer())[start_pos:])
            self._size = len(self._buf)
            self._path = None
            self._owns_path = False

        self._pos = 0

    def _init_from_bytesio(self, src: BytesIO, *, copy: bool) -> None:
        if copy:
            self._init_from_bytes(memoryview(src.to_bytes()))
            self._pos = 0
            # Propagate media_type on copy — caller expected a faithful
            # duplicate, not a media-stripped one.
            if self._media_type is None:
                self._media_type = src._media_type
            return

        # Aliasing memory-backed instances is unsafe because metadata
        # (_size/_pos) would diverge between aliases. Copy instead.
        if src._buf is not None:
            self._init_from_bytes(memoryview(src.to_bytes()))
            self._pos = src._pos
            if self._media_type is None:
                self._media_type = src._media_type
            return

        self._buf = None
        self._size = 0
        self._path = src._path
        self._owns_path = False
        self._pos = src._pos
        if self._media_type is None:
            self._media_type = src._media_type

    @staticmethod
    def _is_pathish(value: Any) -> bool:
        return isinstance(value, (str, Path)) or (
            hasattr(value, "open")
            and hasattr(value, "exists")
            and hasattr(value, "stat")
        )

    @staticmethod
    def _coerce_path(value: Any):
        if isinstance(value, (str, Path)):
            return Path(value)
        return value

    @staticmethod
    def _is_local_path(path: Any) -> bool:
        return isinstance(path, Path)

    def _init_from_path(self, path, *, copy: bool) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        if copy:
            size = path.stat().st_size
            if size > self._cfg.spill_bytes:
                dst = self._cfg.create_spill_path()
                if hasattr(path, "copy_to"):
                    path.copy_to(dst)
                else:
                    shutil.copyfile(path, dst)
                self._buf = None
                self._size = 0
                self._path = dst
                self._owns_path = True
            else:
                payload = path.read_bytes()
                self._buf = bytearray(payload)
                self._size = len(payload)
                self._path = None
                self._owns_path = False
        else:
            self._buf = None
            self._size = 0
            self._path = path
            self._owns_path = False

        self._pos = 0

    def _init_from_filelike(self, src: Any) -> None:
        """Read from *src*'s current cursor to its end."""
        if hasattr(src, "seek") and hasattr(src, "tell"):
            start_pos = src.tell()
            src.seek(0, io.SEEK_END)
            end = src.tell()
            src.seek(start_pos)
            remaining = max(0, end - start_pos)

            if remaining > self._cfg.spill_bytes:
                path = self._cfg.create_spill_path()
                with path.open("w+b") as fh:
                    shutil.copyfileobj(src, fh, length=_COPY_CHUNK_SIZE)
                    fh.flush()
                self._buf = None
                self._size = 0
                self._path = path
                self._owns_path = True
                self._pos = 0
                return

            payload = src.read()
            self._init_from_bytes(memoryview(payload))
            self._pos = 0
            return

        # Non-seekable: have to drain and decide after.
        payload = src.read()
        self._init_from_bytes(memoryview(payload))
        self._pos = 0

    # ------------------------------------------------------------------
    # Backing helpers
    # ------------------------------------------------------------------

    def _create_spill_path(self):
        if self._path is None:
            self._path = self._cfg.create_spill_path()
        return self._path

    def _invalidate_mmap(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

    def _close_read_fh(self) -> None:
        if self._read_fh is not None:
            try:
                self._read_fh.close()
            except Exception:
                pass
            self._read_fh = None

    def _get_read_fh(self) -> IO[bytes] | None:
        """Return a cached read-only file handle for the spilled path.

        Reopens lazily if the cached handle was invalidated. Returns
        ``None`` when there is no spilled path.
        """
        if self._path is None:
            return None
        if self._read_fh is None:
            self._read_fh = self._path.open("rb")
        return self._read_fh

    def _spill_from_bytes(self, mv: memoryview) -> None:
        path = self._cfg.create_spill_path()
        with path.open("w+b") as fh:
            fh.write(_as_contiguous_memoryview(mv))
            fh.flush()

        self._buf = None
        self._size = 0
        self._path = path
        self._owns_path = True
        self._invalidate_mmap()
        self._close_read_fh()

    def spill_to_file(self) -> None:
        if self._buf is None:
            return
        path = self._cfg.create_spill_path()
        with path.open("w+b") as fh:
            if self._size:
                payload = _as_contiguous_memoryview(memoryview(self._buf)[: self._size])
                fh.write(payload)
            fh.flush()

        self._buf = None
        self._size = 0
        self._path = path
        self._owns_path = True
        self._invalidate_mmap()
        self._close_read_fh()

    def _reset_backing_keep_open(self) -> None:
        self._invalidate_mmap()
        self._close_read_fh()

        old_path = self._path
        old_owns_path = self._owns_path

        self._path = None
        self._owns_path = False

        if old_path is not None and old_owns_path and not self._cfg.keep_spilled_file:
            try:
                old_path.unlink(missing_ok=True)
            except Exception:
                pass

        self._buf = bytearray()
        self._size = 0
        self._pos = 0
        self._media_type = None

    def replace_with_payload(self, payload: Any) -> None:
        """Replace this buffer's backing with *payload*, consuming payload.

        Ownership semantics
        -------------------
        When *payload* is a path-backed :class:`BytesIO` that owns its
        spill file, ownership is TRANSFERRED to ``self``. The source's
        ``_owns_path`` is cleared so its ``close()`` will not unlink
        the file that ``self`` now depends on. Callers should treat
        *payload* as consumed — do not read from or close it after
        this call.
        """
        self._reset_backing_keep_open()
        self._init_from(payload, copy=False)

        # Transfer path ownership when we aliased a path-backed payload.
        # Without this, both self and payload claim no ownership (the
        # alias path in _init_from_bytesio sets self._owns_path=False)
        # or — worse — payload retains ownership and self becomes a
        # zombie when payload is GC'd. See also _init_from_bytesio's
        # alias path, which is used for read-only aliasing where
        # transfer is NOT desired.
        if (
            isinstance(payload, BytesIO)
            and payload._path is not None
            and payload._path is self._path
            and payload._owns_path
            and not self._owns_path
        ):
            self._owns_path = True
            payload._owns_path = False
            # Best-effort: also release payload's handles so its
            # eventual close() is a no-op. We don't null out payload._path
            # because inspection (repr, path property) remains valid.
            payload._invalidate_mmap()
            payload._close_read_fh()

    def open_file(self) -> IO[bytes]:
        """Return a writable file object for the current backing.

        For in-memory buffers this returns a stdlib ``io.BytesIO``
        **snapshot** — a copy of the current bytes. Writes to the
        snapshot do NOT propagate back to this BytesIO.

        For spilled buffers this returns ``self._path.open("r+b")`` —
        writes DO go to the backing file.

        This asymmetry is load-bearing for some callers; prefer
        :meth:`pread` / :meth:`pwrite` when you need consistent
        behavior across backing modes.
        """
        if self._buf is not None:
            return io.BytesIO(bytes(memoryview(self._buf)[: self._size]))
        if self._path is None:
            raise RuntimeError("No backing store available")
        return self._path.open("r+b")

    # Backwards-compatible alias.
    buffer = open_file

    # ------------------------------------------------------------------
    # Cursorless IO primitives
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""

        if pos < 0:
            raise ValueError("pread position must be >= 0")

        if self._buf is not None:
            end = min(pos + n, self._size)
            if pos >= end:
                return b""
            return bytes(memoryview(self._buf)[pos:end])

        if self._path is None:
            return b""

        fh = self._get_read_fh()
        if fh is None:
            return b""

        if self._is_local_path(self._path) and _HAS_PREAD:
            return os.pread(fh.fileno(), n, pos)

        # Non-local path or no os.pread: synchronize via seek/read on the
        # cached handle. Not thread-safe — if you need concurrent reads,
        # open separate handles per reader.
        fh.seek(pos)
        return fh.read(n)

    def pwrite(self, mv: memoryview, pos: int) -> int:
        if len(mv) == 0:
            return 0
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")

        if self._buf is not None:
            need = pos + len(mv)
            if need > len(self._buf):
                new_cap = max(need, int(len(self._buf) * 1.5) + 1)
                self._buf.extend(b"\x00" * (new_cap - len(self._buf)))

            memoryview(self._buf)[pos : pos + len(mv)] = mv
            self._size = max(self._size, need)
            return len(mv)

        if self._path is None:
            raise RuntimeError("No backing store available for write")

        # Writes invalidate both the mmap view and the cached read handle
        # so subsequent reads see fresh bytes (Windows mmap over a written
        # file is undefined; re-open is the portable answer).
        self._invalidate_mmap()
        self._close_read_fh()

        with self._path.open("r+b") as fh:
            if self._is_local_path(self._path) and _HAS_PWRITE:
                written = int(os.pwrite(fh.fileno(), mv, pos))
            else:
                fh.seek(pos)
                out = fh.write(mv)
                fh.flush()
                written = len(mv) if out is None else int(out)

        return written

    def _ensure_spill_for_growth(self, extra: int) -> None:
        if self._buf is None:
            return
        # Conservative projection: the write lands at _pos, so worst-case
        # size after the write is max(current size, _pos + extra).
        projected = max(self._size, self._pos + extra)
        if projected > self._cfg.spill_bytes:
            self.spill_to_file()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def parse(cls, obj: Any, config: BufferConfig | None = None) -> "BytesIO":
        if isinstance(obj, cls):
            return obj
        return cls(obj, config=config)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> BufferConfig:
        return self._cfg

    @property
    def spilled(self) -> bool:
        return self._path is not None

    @property
    def path(self):
        return self._path

    @property
    def size(self) -> int:
        if self._buf is not None:
            return self._size
        if self._path is not None and self._path.exists():
            return self._path.stat().st_size
        return 0

    @property
    def media_type(self) -> MediaType:
        if self._media_type is None:
            if self._path is not None:
                # Path extensions are far more reliable than magic bytes for
                # ZIP-based container formats (XLSX, DOCX, …) that would
                # otherwise be detected as generic ZIP.
                from_path = MediaType.parse(str(self._path), default=None)
                if from_path is not None and from_path.mime_type is not MimeTypes.OCTET_STREAM:
                    self._media_type = from_path
                    return self._media_type
            self._media_type = MediaType.parse(self, default=MediaType(MimeTypes.OCTET_STREAM))
        return self._media_type

    @media_type.setter
    def media_type(self, value: "MediaType"):
        self.set_media_type(value, safe=True)

    def set_media_type(
        self,
        value: MediaType,
        *,
        safe: bool = True,
    ) -> "BytesIO":
        from ..enums import MediaType as _MediaType

        parsed = _MediaType.parse(value)
        if parsed is None and safe:
            raise ValueError(f"Invalid media type: {value!r}")
        self._media_type = parsed
        return self

    def media_io(
        self,
        media: Optional[MediaType] = None,
    ) -> "MediaIO":
        from .media_io import MediaIO

        media = self.media_type if media is None else MediaType.parse(media)
        return MediaIO.make(buffer=self, media=media)

    # ------------------------------------------------------------------
    # RawIOBase-ish API
    # ------------------------------------------------------------------

    def json_load(
        self,
        orient: Optional[Literal["records", "split", "index", "columns", "values"]] = None,
        *,
        media_type: Optional[MediaType] = None,
    ):
        media_type = media_type or self.media_type

        if media_type.codec is None:
            if media_type.is_json:
                with self.view(pos=0) as v:
                    return json_module.load(v)

            mio = self.media_io(media_type)

            if orient == "columns":
                return mio.read_pydict()
            return mio.read_pylist()

        with self.decompress(codec=media_type.codec, copy=True) as decompressed:
            return decompressed.json_load(
                orient=orient,
                media_type=media_type.with_codec(None),
            )

    def exists(self) -> bool:
        return self.size > 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def flush(self) -> None:
        return None

    def head(self, n: int = _HEAD_DEFAULT) -> bytes:
        """Return the first *n* bytes without mmapping the whole file."""
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if n <= 0 or self.size == 0:
            return b""

        # In-memory: zero-copy slice.
        if self._buf is not None:
            return bytes(memoryview(self._buf)[: min(n, self._size)])

        # Spilled: pread avoids creating a full-file mmap for a small peek.
        return self.pread(min(n, self.size), 0)

    def tell(self) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        return int(self._pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if whence == io.SEEK_SET:
            new_pos = int(offset)
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + int(offset)
        elif whence == io.SEEK_END:
            new_pos = self.size + int(offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")

        if new_pos < 0:
            raise ValueError("Negative seek position")

        self._pos = new_pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if size is None or size < 0:
            size = max(0, self.size - self._pos)

        out = self.pread(size, self._pos)
        self._pos += len(out)
        return out

    def readinto(self, b) -> int:
        """Read into a pre-allocated buffer without allocating an intermediate bytes.

        For in-memory backing we copy straight from the bytearray slice.
        For spilled local paths we prefer ``os.pread`` into a temporary
        then memcpy; the extra copy is unavoidable without ctypes.
        """
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        mv = memoryview(b)
        n = len(mv)
        if n == 0:
            return 0

        if self._buf is not None:
            end = min(self._pos + n, self._size)
            if self._pos >= end:
                return 0
            chunk_len = end - self._pos
            mv[:chunk_len] = memoryview(self._buf)[self._pos : end]
            self._pos += chunk_len
            return chunk_len

        chunk = self.pread(n, self._pos)
        got = len(chunk)
        if got:
            mv[:got] = chunk
            self._pos += got
        return got

    def readinto1(self, b) -> int:
        return self.readinto(b)

    def write(self, b: Any, *, batch_size: int = 1024 * 1024) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if b is None:
            return 0

        if isinstance(b, str):
            return self.write_str(b)

        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b)

        if isinstance(b, (io.RawIOBase, io.BufferedIOBase)) or hasattr(b, "read"):
            total = 0
            while True:
                chunk = b.read(batch_size)
                if not chunk:
                    break
                total += self.write_bytes(chunk)
            return total

        return self.write_bytes(bytes(b))

    def write_into(
        self,
        dst: IO[bytes] | str | os.PathLike[str],
        *,
        batch_size: int = _COPY_CHUNK_SIZE,
        overwrite: bool = True,
    ) -> int:
        """
        Write the current payload into another sink.

        Parameters
        ----------
        dst:
            Destination sink. Can be:
            - a writable binary file-like object
            - a ``str`` or ``PathLike`` file path
        batch_size:
            Chunk size used when copying spilled content to file-like sinks.
        overwrite:
            Only applies to path-like destinations. If False and the path exists,
            ``FileExistsError`` is raised.

        Returns
        -------
        int
            Number of bytes written.
        """
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        # --------------------------------------------------------------
        # Path-like destinations: one unified branch.
        # --------------------------------------------------------------
        if isinstance(dst, str) or isinstance(dst, os.PathLike) or self._is_pathish(dst):
            dst_path = self._coerce_path(dst)

            if dst_path.exists() and not overwrite:
                raise FileExistsError(f"Destination already exists: {dst_path}")

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # In-memory → just write out.
            if self._buf is not None:
                payload = _as_contiguous_memoryview(
                    memoryview(self._buf)[: self._size]
                )
                dst_path.write_bytes(payload)
                return self._size

            # No backing or empty spill → write zero bytes.
            if self._path is None or self.size == 0:
                dst_path.write_bytes(b"")
                return 0

            # No-op if source and destination are the same path.
            if self._path == dst_path:
                return self.size

            # Custom (e.g. Databricks) → use its copy_to if available.
            if hasattr(self._path, "copy_to") and not isinstance(self._path, Path):
                self._path.copy_to(dst_path)
                return self.size

            # Local-to-local fast path.
            if self._is_local_path(self._path) and isinstance(dst_path, Path):
                shutil.copyfile(self._path, dst_path)
                return self.size

            # Cross-type fallback: stream.
            with self._path.open("rb") as src:
                with dst_path.open("wb") as out:
                    shutil.copyfileobj(src, out, length=batch_size)
            return self.size

        # --------------------------------------------------------------
        # File-like destinations.
        # --------------------------------------------------------------
        if not hasattr(dst, "write"):
            raise TypeError(
                f"write_into() expected a writable binary IO or path-like destination, "
                f"got {type(dst)!r}"
            )

        writable = getattr(dst, "writable", None)
        if callable(writable) and not writable():
            raise ValueError("Destination IO is not writable")

        total = 0

        if self._buf is not None:
            payload = _as_contiguous_memoryview(
                memoryview(self._buf)[: self._size]
            )
            out = dst.write(payload)
            written = self._size if out is None else int(out)
            if written != self._size:
                raise io.BlockingIOError(
                    f"Short write while writing in-memory payload: "
                    f"expected {self._size}, got {written}"
                )
            total = written
        elif self._path is not None and self.size:
            with self._path.open("rb") as src:
                while True:
                    chunk = src.read(batch_size)
                    if not chunk:
                        break
                    out = dst.write(chunk)
                    written = len(chunk) if out is None else int(out)
                    if written != len(chunk):
                        raise io.BlockingIOError(
                            f"Short write while streaming spilled payload: "
                            f"expected {len(chunk)}, got {written}"
                        )
                    total += written
        else:
            out = dst.write(b"")
            total = 0 if out is None else int(out)

        flush = getattr(dst, "flush", None)
        if callable(flush):
            flush()

        return total

    def write_linebreak(self, newline: str = "\n") -> int:
        return self.write(newline)

    def write_bytes(self, b: bytes | bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        mv = memoryview(b)
        if len(mv) == 0:
            return 0

        self._ensure_spill_for_growth(len(mv))
        n = self.pwrite(mv, self._pos)
        self._pos += n
        return n

    def write_str(self, s: str, encoding: str = "utf-8") -> int:
        if not s:
            return 0
        return self.write_bytes(s.encode(encoding))

    def truncate(self, size: int | None = None) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if size is None:
            size = self._pos
        size = int(size)

        if size < 0:
            raise ValueError("Negative size value")

        if self._buf is not None:
            if size < self._size:
                self._size = size
            else:
                if size > len(self._buf):
                    self._buf.extend(b"\x00" * (size - len(self._buf)))
                self._size = size

            if self._pos > size:
                self._pos = size
            return size

        if self._path is None:
            raise RuntimeError("No backing store available for truncate")

        # truncate(0) on a spilled buffer drops back to in-memory empty —
        # otherwise we'd leak a zero-byte file until close().
        if size == 0:
            self._reset_backing_keep_open()
            return 0

        self._invalidate_mmap()
        self._close_read_fh()
        with self._path.open("r+b") as fh:
            fh.truncate(size)

        if self._pos > size:
            self._pos = size
        return size

    # ------------------------------------------------------------------
    # Structured binary I/O — little-endian
    # ------------------------------------------------------------------

    def _read_exact(self, n: int) -> bytes:
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
        return bool(self.read_uint8())

    def write_bool(self, value: bool) -> int:
        return self.write_uint8(1 if value else 0)

    def read_bytes_u32(self) -> bytes:
        return self._read_exact(self.read_uint32())

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        return self.write_uint32(len(mv)) + self.write_bytes(mv)

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def xxh3_64(self) -> "xxhash.xxh3_64":
        from yggdrasil.xxhash.lib import xxhash

        h = xxhash.xxh3_64()
        h.update(self.memoryview())
        return h

    def xxh3_int64(self) -> int:
        u = self.xxh3_64().intdigest()
        return u if u < 2**63 else u - 2**64

    def blake3(self) -> "blake3.blake3":
        from yggdrasil.blake3.lib import blake3

        h = blake3(max_threads=blake3.AUTO)

        if self._buf is not None:
            if self._size:
                h.update(memoryview(self._buf)[: self._size])
            return h

        if self._path is not None:
            if not self._is_local_path(self._path):
                h.update(self.to_bytes())
                return h
            h.update_mmap(str(self._path))
            return h

        return h

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        if self.size == 0:
            return ""
        return self.to_bytes().decode(encoding, errors)

    def getvalue(self) -> bytes:
        return self.to_bytes()

    def view(
        self,
        *,
        pos: int | None = None,
        size: int | None = None,
        max_size: int | None = None,
    ):
        from .bytes_view import BytesIOView

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if pos is None:
            # Default: start the view at the current parent cursor,
            # except when the cursor is at EOF — treat that as "rewind"
            # so ``for line in view:`` on a just-written buffer works.
            pos = self._pos
            if pos == self.size:
                pos = 0

        pos = int(pos)
        if pos < 0:
            raise ValueError("view pos must be >= 0")

        if size is None:
            size = max(0, self.size - pos)
        else:
            size = int(size)
            if size < 0:
                raise ValueError("view length must be >= 0")

        return BytesIOView(
            parent=self,
            start=pos,
            size=size,
            pos=0,
            max_size=max_size,
        )

    def memoryview(self):
        if self._buf is not None:
            return memoryview(self._buf)[: self._size]

        if self._path is None or not self._path.exists():
            return memoryview(b"")

        if not self._is_local_path(self._path):
            return memoryview(self.to_bytes())

        size = self._path.stat().st_size
        if size == 0:
            return memoryview(b"")

        if self._mmap is None or self._mmap.closed:
            with self._path.open("rb") as fh:
                self._mmap = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ)

        return memoryview(self._mmap)

    def to_bytes(self) -> bytes:
        if self._buf is not None:
            return bytes(memoryview(self._buf)[: self._size])
        if self.size == 0:
            return b""
        return self.pread(self.size, 0)

    def to_base64(
        self,
        urlsafe: bool = True
    ) -> str:
        b = self.to_bytes()

        if urlsafe:
            return base64.urlsafe_b64encode(b).decode("ascii")
        else:
            return base64.b64encode(b).decode("ascii")

    def open_reader(self) -> IO[bytes]:
        if self._buf is not None:
            return io.BytesIO(bytes(memoryview(self._buf)[: self._size]))
        if self._path is None:
            raise RuntimeError("Spilled buffer has no path")
        return self._path.open("rb")

    def to_arrow_io(self, mode: str = "r"):
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if "r" in mode:
            if self.spilled and self._path is not None:
                if not self._is_local_path(self._path):
                    return pa.PythonFile(self.open_reader(), mode="r")
                return pa.memory_map(str(self._path), "r")

            mv = self.memoryview()
            buf = pa.py_buffer(mv) if len(mv) else pa.py_buffer(b"")
            return pa.BufferReader(buf)

        if "w" in mode or "a" in mode:
            if not self.spilled:
                self.spill_to_file()

            if self._path is None:
                raise RuntimeError("Failed to materialize spill file for Arrow IO")

            if not self._is_local_path(self._path):
                return pa.PythonFile(self.open_file(), mode=mode)

            # Writes invalidate any existing read-side state.
            self._invalidate_mmap()
            self._close_read_fh()

            if "w" in mode:
                return pa.OSFile(str(self._path), mode="w")
            return pa.OSFile(str(self._path), mode="a")

        raise ValueError(f"Unsupported mode for to_arrow_io: {mode!r}")

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    #
    # Both methods delegate to the Codec class, which now streams
    # chunk-by-chunk for codecs that support it (gzip, zstd, lz4, bz2,
    # xz, lzma) and falls back to a bytes roundtrip for codecs that
    # don't (snappy, brotli). Peak memory is bounded by the codec's
    # internal chunk size for streaming codecs, regardless of input
    # size.

    def compress(
        self,
        codec: "Codec | str",
        *,
        copy: bool = False,
    ) -> "BytesIO":
        """Compress this buffer with *codec*.

        Delegates to :meth:`Codec.compress`, which streams when the
        codec supports it. The resulting BytesIO is re-wrapped through
        this instance's config so a large compressed output will spill
        if it exceeds the spill threshold.

        When ``copy=True`` returns a new instance; when ``copy=False``
        replaces this instance's backing with the compressed payload
        and returns ``self``.
        """
        from ..enums.codec import Codec as _Codec

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        c = _Codec.parse(codec)
        if c is None:
            raise ValueError(f"Unknown codec: {codec!r}")

        target_mt = self.media_type.with_codec(c) if self._media_type else None

        payload = c.compress(self)
        owned = self._own_payload(payload)
        owned._media_type = target_mt

        if copy:
            return owned

        # Replace-in-place. _reset_backing_keep_open clears _media_type
        # so we re-set it explicitly after.
        self.replace_with_payload(owned)
        self._media_type = target_mt
        return self

    def decompress(
        self,
        codec: "Codec | str | None" = "infer",
        *,
        copy: bool = False,
    ) -> "BytesIO":
        """Decompress this buffer with *codec*.

        ``codec="infer"`` reads the codec from the current media type;
        any other value is parsed via :meth:`Codec.parse`. An explicitly
        named but unparseable codec raises ``ValueError`` — parity with
        :meth:`compress`.

        Delegates to :meth:`Codec.decompress`, which streams when the
        codec supports it.
        """
        from ..enums.codec import Codec as _Codec

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if codec == "infer":
            target_mt = self.media_type.without_codec()
            resolved = self.media_type.codec
        else:
            target_mt = (
                self._media_type.without_codec()
                if self._media_type else None
            )
            resolved = _Codec.parse(codec)
            # Parity with compress(): an unknown codec name is a user
            # error, not a silent no-op. None stays permitted (means
            # "no codec needed").
            if resolved is None and codec is not None:
                raise ValueError(f"Unknown codec: {codec!r}")

        if resolved is None:
            if copy:
                return self.__class__(self, copy=True, config=self._cfg)
            return self

        payload = resolved.decompress(self)
        owned = self._own_payload(payload)
        owned._media_type = target_mt

        if copy:
            return owned

        self.replace_with_payload(owned)
        self._media_type = target_mt
        return self

    def _own_payload(self, payload: "BytesIO") -> "BytesIO":
        """Re-wrap a codec-produced payload under this instance's config.

        :meth:`Codec.compress` / :meth:`Codec.decompress` build their
        output via ``BytesIO()`` without our config, so the result
        uses the default spill threshold. If the caller configured a
        custom threshold (e.g. ``spill_bytes=1`` for a test, or a
        larger value for production), we need the payload to honor it.

        Reuses the payload as-is if it already shares our config,
        otherwise reconstructs through our constructor.
        """
        if type(payload) is BytesIO and payload._cfg is self._cfg:
            return payload

        # Feed the payload through our config-aware constructor. For
        # large outputs this goes through _init_from_bytesio which
        # either aliases the spilled path (zero-copy) or copies the
        # memory-backed bytes — both honor our spill_bytes setting.
        rewrapped = self.__class__(payload, copy=True, config=self._cfg)
        try:
            payload.close()
        except Exception:
            pass
        return rewrapped

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        self._invalidate_mmap()
        self._close_read_fh()

        old_path = self._path
        old_owns_path = self._owns_path

        self._buf = None
        self._size = 0
        self._pos = 0
        self._path = None
        self._owns_path = False
        self._media_type = None
        self._closed = True

        if old_path is not None and old_owns_path and not self._cfg.keep_spilled_file:
            try:
                old_path.unlink(missing_ok=True)
            except Exception:
                pass

        super().close()

    # Data framing
    def to_polars(
        self,
        media: Optional[MediaType] = None,
        options: "MediaOptions | None" = None,
        **option_kwargs,
    ):
        return self.media_io(media=media).read_polars_frame(
            options=options,
            **option_kwargs
        )