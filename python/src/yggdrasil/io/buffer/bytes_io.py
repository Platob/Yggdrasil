# yggdrasil/io/buffer/bytes_io.py
"""Spill-to-disk byte buffer with transparent memory/file backing.

Optimized version:
- Own logical cursor (self._pos) independent of backing store cursor
- In-memory backing is a native bytearray (self._buf) + logical size (self._size)
  (no stdlib io.BytesIO)
- Uses os.pread/os.pwrite when possible for spilled files (true cursorless IO)
- Keeps mmap for zero-copy reads when spilled
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
from typing import Any, IO, Optional, TYPE_CHECKING

from yggdrasil.io.config import BufferConfig, DEFAULT_CONFIG
from yggdrasil.io.enums import MediaType, MimeType
from yggdrasil.io.path import AbstractDataPath
from yggdrasil.io.types import BytesLike

if TYPE_CHECKING:
    from .media_io import MediaIO
    import blake3
    import xxhash

__all__ = ["BytesIO"]

_HEAD_DEFAULT = 128


class BytesIO(io.RawIOBase):
    __slots__ = (
        "_cfg",
        "_buf",
        "_size",
        "_pos",
        "_file",
        "_path",
        "_mmap",
        "auto_close",
        "_closed",
    )

    def __init__(
        self,
        data: IO[bytes] | bytes | bytearray | memoryview | str | Path | None = None,
        *,
        config: BufferConfig | None = None,
        auto_close: bool = True,
    ) -> None:
        super().__init__()
        self._cfg: BufferConfig = config or DEFAULT_CONFIG

        # Memory backing (no io.BytesIO)
        self._buf: bytearray | None = None
        self._size: int = 0  # logical size within _buf

        # Owned cursor
        self._pos: int = 0

        # Spilled backing
        self._file: IO[bytes] | None = None
        self._path: AbstractDataPath | None = None
        self._mmap: mmap.mmap | None = None

        self.auto_close: bool = auto_close
        self._closed: bool = False

        self._init_from(data)

    # ------------------------------------------------------------------
    # Init dispatch
    # ------------------------------------------------------------------

    def _init_from(self, data: Any) -> None:
        if data is None:
            self._buf = bytearray()
            self._size = 0
            self._pos = 0
            return

        if isinstance(data, (bytes, bytearray, memoryview)):
            self._init_from_bytes(memoryview(data))
            return

        if isinstance(data, io.BytesIO):
            self._init_from_stdlib_bytesio(data)
            return

        if isinstance(data, (str, Path)):
            self._init_from_path(Path(data))
            return

        if isinstance(data, BytesIO):
            # Share backing stores (dangerous but intentional), cursor stays owned here.
            self._buf = data._buf
            self._size = data._size
            self._file = data._file
            self._path = data._path
            self._mmap = None  # never share mmaps
            self.auto_close = False
            self._pos = data._pos
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
            # copy into our bytearray
            self._buf = bytearray(mv.tobytes() if not mv.contiguous else mv)
            self._size = n
            self._file = None
            self._path = None
        self._pos = 0

    def _init_from_stdlib_bytesio(self, src: io.BytesIO) -> None:
        saved = src.tell()
        src.seek(0, io.SEEK_END)
        size = src.tell()
        src.seek(0)

        if size > self._cfg.spill_bytes:
            path, fh = self._open_spill_file()
            shutil.copyfileobj(src, fh, length=8 * 1024 * 1024)
            fh.flush()
            self._file, self._path = fh, path
            self._buf, self._size = None, 0
        else:
            payload = src.read()
            self._buf = bytearray(payload)
            self._size = len(payload)
            self._file = None
            self._path = None

        src.seek(saved)
        self._pos = 0

    def _init_from_path(self, path: Path) -> None:
        self._path = path
        self._file = path.open("r+b") if path.exists() else path.open("w+b")
        self._buf, self._size = None, 0
        self._pos = 0

    def _init_from_filelike(self, src: Any) -> None:
        # If seekable, measure remaining size cheaply
        if hasattr(src, "seek") and hasattr(src, "tell"):
            saved = src.tell()
            src.seek(0, io.SEEK_END)
            end = src.tell()
            src.seek(saved)
            remaining = max(0, end - saved)

            if remaining > self._cfg.spill_bytes:
                path, fh = self._open_spill_file()
                shutil.copyfileobj(src, fh, length=8 * 1024 * 1024)
                fh.flush()
                self._file, self._path = fh, path
                self._buf, self._size = None, 0
            else:
                payload = src.read()
                self._buf = bytearray(payload)
                self._size = len(payload)
                self._file = None
                self._path = None
        else:
            payload = src.read()
            self._init_from_bytes(memoryview(payload))

        self._pos = 0

    # --- add these helpers inside BytesIO ---------------------------------

    def _reset_backing_keep_open(self) -> None:
        """
        Drop current backing stores without marking the BytesIO as closed.
        Used for in-place replace operations (copy=False).
        """
        self._invalidate_mmap()

        # close old file if we own it
        if self.auto_close and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

        old_path = self._path
        self._file = None
        self._path = None

        # delete old spill file if configured
        if old_path is not None and not self._cfg.keep_spilled_file:
            try:
                old_path.unlink(missing_ok=True)
            except Exception:
                pass

        # reset memory buffer
        self._buf = bytearray()
        self._size = 0
        self._pos = 0

    def _replace_with_payload(self, payload: bytes) -> None:
        """
        Replace this BytesIO backing with payload (memory or spilled based on spill_bytes).
        Cursor ends at 0.
        """
        self._reset_backing_keep_open()
        # re-init from bytes using the same logic as constructor
        self._init_from_bytes(memoryview(payload))
        self._pos = 0

    def _bytes_from_codec_output(self, out: Any) -> bytes:
        """
        Normalize codec output to raw bytes.
        - Many codec APIs return io.BytesIO-like objects (getvalue)
        - Some may return bytes/bytearray/memoryview
        - Some may return file-like (read)
        """
        if out is None:
            return b""
        if isinstance(out, (bytes, bytearray, memoryview)):
            return bytes(out)
        gv = getattr(out, "getvalue", None)
        if callable(gv):
            return gv()
        rd = getattr(out, "read", None)
        if callable(rd):
            return rd()
        return bytes(out)

    # ------------------------------------------------------------------
    # Backing store helpers
    # ------------------------------------------------------------------

    def _open_spill_file(self) -> tuple[Path, IO[bytes]]:
        tmp_dir = self._cfg.tmp_dir
        name = f"{self._cfg.prefix}{uuid.uuid4().hex}{self._cfg.suffix}"
        spill_path: Path = (tmp_dir / name) if tmp_dir is not None else (Path(tempfile.gettempdir()) / name)
        return spill_path, spill_path.open("w+b")

    def _invalidate_mmap(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

    def _spill_from_bytes(self, mv: memoryview) -> None:
        path, fh = self._open_spill_file()
        fh.write(mv.tobytes() if not mv.contiguous else mv)
        fh.flush()
        self._file, self._path = fh, path
        self._buf, self._size = None, 0
        self._invalidate_mmap()

    def spill_to_file(self) -> None:
        """Move memory buffer to file. Keeps self._pos."""
        if self._buf is None:
            return

        cfg = self._cfg
        tmp_dir = str(cfg.tmp_dir) if cfg.tmp_dir is not None else None
        fd, name = tempfile.mkstemp(prefix=cfg.prefix, suffix=cfg.suffix, dir=tmp_dir)
        path = Path(name)
        fh = os.fdopen(fd, "w+b", buffering=0)

        if self._size:
            fh.write(memoryview(self._buf)[: self._size])
            fh.flush()

        self._file, self._path = fh, path
        self._buf, self._size = None, 0
        self._invalidate_mmap()

    def buffer(self) -> IO[bytes]:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if self._file is None:
            raise RuntimeError("BytesIO is in memory mode; no file handle available")
        return self._file

    # ------------------------------------------------------------------
    # Cursorless IO primitives
    # ------------------------------------------------------------------

    def _pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""

        if self._buf is not None:
            end = min(pos + n, self._size)
            if pos >= end:
                return b""
            # memoryview slice -> bytes copy
            return bytes(memoryview(self._buf)[pos:end])

        fh = self.buffer()
        try:
            return os.pread(fh.fileno(), n, pos)
        except Exception:
            saved = None
            try:
                if hasattr(fh, "tell") and hasattr(fh, "seek"):
                    saved = fh.tell()
                    fh.seek(pos)
                return fh.read(n)
            finally:
                if saved is not None:
                    try:
                        fh.seek(saved)
                    except Exception:
                        pass

    def _pwrite(self, mv: memoryview, pos: int) -> int:
        if len(mv) == 0:
            return 0

        if self._buf is not None:
            need = pos + len(mv)
            if need > len(self._buf):
                # grow with minimal realloc churn
                new_cap = max(need, int(len(self._buf) * 1.5) + 1)
                self._buf.extend(b"\x00" * (new_cap - len(self._buf)))

            # if pos is beyond logical size, fill the gap with zeros (already zeros from growth/extend)
            memoryview(self._buf)[pos : pos + len(mv)] = mv
            self._size = max(self._size, need)
            return len(mv)

        fh = self.buffer()
        try:
            n = os.pwrite(fh.fileno(), mv, pos)
            return int(n)
        except Exception:
            saved = None
            try:
                if hasattr(fh, "tell") and hasattr(fh, "seek"):
                    saved = fh.tell()
                    fh.seek(pos)
                n = fh.write(mv)
                return int(n) if n is not None else len(mv)
            finally:
                if saved is not None:
                    try:
                        fh.seek(saved)
                    except Exception:
                        pass

    def _ensure_spill_for_growth(self, extra: int) -> None:
        # Growth check uses logical position too (overwrite beyond EOF)
        if self._buf is None:
            return
        projected = max(self._size, self._pos) + extra
        if projected > self._cfg.spill_bytes:
            self.spill_to_file()

    # ------------------------------------------------------------------
    # Dunder helpers
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def wrap(
        cls,
        handler: IO[bytes] | io.IOBase | str | Path | "BytesIO",
        *,
        auto_close: bool = False,
        config: BufferConfig | None = None,
    ) -> "BytesIO":
        # Fast path: already BytesIO
        if isinstance(handler, BytesIO):
            if config is not None:
                handler._cfg = config
            handler.auto_close = auto_close
            # Make sure delegation points at itself (or nothing)
            handler._handler = None
            return handler

        # Path-like => use normal parse/init behavior (we own the file)
        if isinstance(handler, (str, Path)):
            b = cls.parse(handler, config=config)
            b.auto_close = auto_close
            b._handler = None
            return b

        # Otherwise: treat as file-ish
        if not hasattr(handler, "read"):
            # last resort: parse (bytes-like etc)
            b = cls.parse(handler, config=config)
            b.auto_close = auto_close
            b._handler = None
            return b

        b = cls(config=config)

        # Keep original handler for API delegation + close correctness
        b._handler = handler
        b.auto_close = auto_close

        # Normalize _file to a binary object where possible
        f: Any = handler

        # If user passed text IO, try to grab its underlying binary buffer
        if isinstance(handler, io.TextIOBase):
            # common: TextIOWrapper has .buffer
            buf = getattr(handler, "buffer", None)
            if buf is None:
                raise TypeError(
                    "BytesIO.wrap() got a text stream without a .buffer; "
                    "wrap a binary stream (rb/wb) or pass handler.buffer."
                )
            f = buf

        # Some wrappers expose .raw which is closer to the OS file
        raw = getattr(f, "raw", None)
        if raw is not None and hasattr(raw, "fileno"):
            # Prefer raw if it looks fileno-capable (best chance for os.pread/pwrite)
            f = raw

        b._file = f
        b._path = None
        b._buf = None
        b._size = 0
        b._mmap = None

        # Logical cursor: start aligned to underlying cursor if seekable/tellable
        try:
            if hasattr(handler, "tell"):
                b._pos = int(handler.tell())
            elif hasattr(f, "tell"):
                b._pos = int(f.tell())
            else:
                b._pos = 0
        except Exception:
            b._pos = 0

        return b

    @classmethod
    def parse(cls, obj: Any, config: BufferConfig | None = None) -> "BytesIO":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, io.BytesIO):
            b = cls(config=config)
            payload = obj.getvalue()
            b._init_from_bytes(memoryview(payload))
            b._pos = 0
            return b

        if hasattr(obj, "read"):
            b = cls(config=config)
            b._init_from_filelike(obj)
            b._pos = 0
            return b

        if isinstance(obj, (str, Path)):
            return cls(Path(obj), config=config)

        # bytes-like fallback
        b = cls(config=config)
        b.write_bytes(obj)
        b.seek(0)
        return b

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def config(self) -> BufferConfig:
        return self._cfg

    @property
    def spilled(self) -> bool:
        return self._path is not None and self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def size(self) -> int:
        if self._buf is not None:
            return self._size
        fh = self.buffer()
        try:
            return os.fstat(fh.fileno()).st_size
        except Exception:
            saved = None
            try:
                if hasattr(fh, "tell") and hasattr(fh, "seek"):
                    saved = fh.tell()
                    fh.seek(0, io.SEEK_END)
                    end = fh.tell()
                    return int(end)
                # ultra fallback (avoid in normal life)
                data = fh.read()
                return len(data)
            finally:
                if saved is not None:
                    try:
                        fh.seek(saved)
                    except Exception:
                        pass

    @property
    def media_type(self) -> MediaType:
        """Infer MediaType from magic bytes (cursor-safe because we own cursor)."""
        return MediaType.parse_io(self, MediaType(MimeType.OCTET_STREAM))

    def media_io(self, media: Optional[MediaType] = None) -> "MediaIO":
        from .media_io import MediaIO

        media = MediaType.parse(media)

        if media.is_octet:
            media = self.media_type

        return MediaIO.make(buffer=self, media=media)

    # ------------------------------------------------------------------
    # RawIOBase-ish API
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.size > 0

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def flush(self) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
        except Exception:
            pass

    def head(self, n: int = _HEAD_DEFAULT) -> memoryview:
        """View of first n bytes without touching self._pos."""
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if n <= 0 or self.size == 0:
            return memoryview(b"")

        if self._buf is not None:
            return memoryview(self._buf)[: min(n, self._size)]

        return memoryview(self._pread(n, 0))

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
            new_pos = int(self._pos) + int(offset)
        elif whence == io.SEEK_END:
            new_pos = int(self.size) + int(offset)
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

        out = self._pread(size, self._pos)
        self._pos += len(out)
        return out

    def write(self, b: Any, *, batch_size: int = 1024 * 1024) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if b is None:
            return 0

        if isinstance(b, str):
            return self.write_str(b)

        if isinstance(b, (bytes, bytearray, memoryview)):
            return self.write_bytes(b)

        # stream-like
        if isinstance(b, (io.RawIOBase, io.BufferedIOBase)) or hasattr(b, "read"):
            total = 0
            while True:
                chunk = b.read(batch_size)
                if not chunk:
                    break
                total += self.write_bytes(chunk)
            return total

        return self.write_bytes(bytes(b))

    def write_bytes(self, b: bytes | bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        mv = memoryview(b)
        if len(mv) == 0:
            return 0

        # spill decision in memory-mode only
        self._ensure_spill_for_growth(len(mv))

        n = self._pwrite(mv, self._pos)
        self._pos += n

        # spilled writes can stale an existing mmap view
        if self._buf is None:
            self._invalidate_mmap()

        return n

    def write_str(self, s: str, encoding: str = "utf-8") -> int:
        if not s:
            return 0
        return self.write_bytes(s.encode(encoding))

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
        from yggdrasil.xxhash import xxh3_64

        h = xxh3_64()
        h.update(self.memoryview())
        return h

    def blake3(self) -> "blake3.blake3":
        from yggdrasil.blake3 import blake3

        h = blake3(max_threads=blake3.AUTO)

        if self._buf is not None:
            if self._size:
                h.update(memoryview(self._buf)[: self._size])
            return h

        if self._path is not None:
            h.update_mmap(str(self._path))
            return h

        # fallback: stream from file cursorlessly
        total = self.size
        off = 0
        step = 8 * 1024 * 1024
        while off < total:
            chunk = self._pread(min(step, total - off), off)
            if not chunk:
                break
            h.update(chunk)
            off += len(chunk)
        return h

    # ------------------------------------------------------------------
    # Decode / convenience
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
        text: bool = False,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: str = "",
        start: int = 0,
        length: int | None = None,
    ):
        """
        File-like, cursor-owned view over this BytesIO.

        - Does NOT touch self._pos
        - start/length carve a window (absolute offsets)
        - text=False returns binary stream; text=True returns TextIOWrapper
        """
        from .bytes_view import open_bytes_view

        return open_bytes_view(
            self,
            text=text,
            encoding=encoding,
            errors=errors,
            newline=newline,
            start=start,
            length=length,
        )

    def memoryview(self) -> memoryview:
        """Zero-copy view for memory mode; mmap view for spilled mode."""
        if self._buf is not None:
            return memoryview(self._buf)[: self._size]

        fh = self.buffer()
        try:
            fh.flush()
        except Exception:
            pass

        size = os.fstat(fh.fileno()).st_size
        if size == 0:
            return memoryview(b"")

        if self._mmap is None or self._mmap.closed:
            self._mmap = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ)
        return memoryview(self._mmap)

    def to_bytes(self) -> bytes:
        if self._buf is not None:
            return bytes(memoryview(self._buf)[: self._size])
        if self.size == 0:
            return b""
        return self._pread(self.size, 0)

    def open_reader(self) -> IO[bytes]:
        if self._buf is not None:
            return io.BytesIO(bytes(memoryview(self._buf)[: self._size]))
        if self._path is None:
            raise RuntimeError("Spilled buffer has no path (unexpected state)")
        return self._path.open("rb")

    def open_writer(self) -> IO[bytes]:
        if self._buf is not None:
            # there is no file writer in memory mode; expose a BytesIO copy for compatibility
            return io.BytesIO(bytes(memoryview(self._buf)[: self._size]))
        return self.buffer()

    def to_arrow_io(self, mode: str = "r"):
        """
        Return a PyArrow-native IO object over this buffer.

        mode:
          - "r"/"rb": readable
          - "w"/"wb": writable (truncate)
          - "a"/"ab": writable (append)
        """
        from yggdrasil.arrow.lib import pyarrow as pa

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if "r" in mode:
            if self.spilled and self._path is not None:
                return pa.memory_map(str(self._path), "r")

            if self._file is not None:
                try:
                    self._file.flush()
                except Exception:
                    pass
                try:
                    return pa.OSFile(self._file.fileno(), mode="r")
                except Exception:
                    return pa.PythonFile(self._file)

            mv = self.memoryview()
            buf = pa.py_buffer(mv) if len(mv) else pa.py_buffer(b"")
            return pa.BufferReader(buf)

        if "w" in mode or "a" in mode:
            if not self.spilled:
                self.spill_to_file()

            if "w" in mode:
                return pa.OSFile(str(self._path), mode="w")
            return pa.OSFile(str(self._path), mode="a")

        raise ValueError(f"Unsupported mode for to_arrow_io: {mode!r}")

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    # --- replace compress() with this -------------------------------------

    def compress(self, codec: "Codec | str", *, copy: bool = False) -> "BytesIO":
        from ..enums.codec import Codec as _Codec

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        c = _Codec.parse(codec)
        if c is None:
            raise ValueError(f"Unknown codec: {codec!r}")

        # codec.compress(self) must preserve self cursor (per your codec system)
        out_std = c.compress(self)
        payload = self._bytes_from_codec_output(out_std)

        if copy:
            out = BytesIO(payload, config=self._cfg)
            out.seek(0)
            return out

        # in-place replace
        self._replace_with_payload(payload)
        return self

    # --- replace decompress() with this -----------------------------------

    def decompress(self, codec: "Codec | str | None" = "infer", *, copy: bool = False) -> "BytesIO":
        from ..enums.codec import Codec as _Codec
        from ..enums.codec import detect as _detect

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        # resolve codec
        if codec is None or (isinstance(codec, str) and codec.strip().lower() == "infer"):
            c = _detect(self)
            if c is None:
                # infer failed => "no-op" semantics:
                # - copy=True  returns new buffer with same bytes
                # - copy=False replaces self with same bytes (still resets cursor to 0)
                payload = self.to_bytes()
                if copy:
                    out = BytesIO(payload, config=self._cfg)
                    out.seek(0)
                    return out
                self._replace_with_payload(payload)
                return self
        else:
            c = _Codec.parse(codec)
            if c is None:
                raise ValueError(f"Unknown codec: {codec!r}")

        out_std = c.open(self)  # streaming open; preserves self cursor by design
        payload = self._bytes_from_codec_output(out_std)

        if copy:
            out = BytesIO(payload, config=self._cfg)
            out.seek(0)
            return out

        # in-place replace
        self._replace_with_payload(payload)
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        try:
            self.flush()
        except Exception:
            pass

        if not self.auto_close:
            return

        self._invalidate_mmap()

        # drop memory
        self._buf = None
        self._size = 0
        self._pos = 0

        # close file
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None

        # remove spill file
        if self._path is not None and not self._cfg.keep_spilled_file:
            try:
                self._path.unlink(missing_ok=True)
            except Exception:
                pass
        self._path = None

        self._closed = True
        super().close()