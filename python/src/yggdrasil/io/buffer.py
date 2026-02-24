# yggdrasil/pyutils/dynamic_buffer.py
from __future__ import annotations

import io
import mmap
import os
import shutil
import struct
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Optional, Union, TYPE_CHECKING

__all__ = ["BytesIO", "BufferConfig"]

from yggdrasil.io.path import AbstractDataPath

if TYPE_CHECKING:
    import xxhash
    import blake3


BytesLike = Union[bytes, bytearray, memoryview]

@dataclass(frozen=True, slots=True)
class BufferConfig:
    """
    Configuration for DynamicBuffer.

    spill_bytes:
        When buffered data exceeds this threshold, spill from in-memory BytesIO
        to a real temporary file on disk.

    tmp_dir:
        Directory where spill files are created. If None, uses system temp dir.

    prefix/suffix:
        Naming for the spill file.

    keep_spilled_file:
        If True, spilled temp files are not deleted on close/cleanup.
    """
    spill_bytes: int = 128 * 1024 * 1024  # 128 MiB
    prefix: str = "tmp-"
    suffix: str = ".bin"
    keep_spilled_file: bool = False
    tmp_dir: Optional[AbstractDataPath] = None

    @classmethod
    def default(cls):
        return DEFAULT_CONFIG


DEFAULT_CONFIG = BufferConfig()


class BytesIO(io.RawIOBase):
    """
    A bytes buffer that starts in memory and spills to a local temp file
    when it grows beyond a threshold.

    - file-like: write/read/seek/tell/flush/close
    - getvalue(): returns bytes (loads if spilled; see docs)
    - to_bytes(): reads entire payload (can be huge)
    - memoryview(): zero-copy view for in-mem, mmap view when spilled
    """

    def __init__(
        self,
        data: Any = None,
        *,
        config: BufferConfig | None = None,
    ) -> None:
        super().__init__()

        self._cfg = config or DEFAULT_CONFIG

        # Internal state — exactly one of (_mem) or (_file + _path) is active at
        # any given time.  _spilled tracks whether we've crossed the threshold.
        self._mem: io.BytesIO | None = None
        self._file: IO[bytes] | None = None
        self._path: AbstractDataPath | None = None

        if data is None:
            # Empty buffer — start in-memory, spill lazily on write overflow
            self._mem = io.BytesIO()

        elif isinstance(data, (bytes, bytearray, memoryview)):
            self._load_bytes(bytes(data) if not isinstance(data, bytes) else data)


        elif isinstance(data, io.BytesIO):
            # Peek at size without dumping the entire buffer into a new bytes object.
            # If it's over the threshold, stream it directly to a spill file.

            pos = data.tell()
            data.seek(0, io.SEEK_END)
            size = data.tell()
            data.seek(0)  # rewind for copyfileobj

            if size > self._cfg.spill_bytes:
                path, fh = self._open_spill_file()
                shutil.copyfileobj(data, fh, length=self._cfg.spill_bytes)
                fh.flush()
                data.seek(pos)  # restore caller's cursor — we don't own this object

                self._path = path
                self._file = fh

            else:
                # Small enough — snapshot into our own BytesIO, restore caller cursor
                self._mem = io.BytesIO(data.read())
                data.seek(pos)

        elif isinstance(data, Path):
            self._path = data
            self._file = data.open("r+b") if data.exists() else data.open("w+b")


        elif hasattr(data, "read"):
            # Generic file-like: can't cheaply size it, so stream into a temp
            # BytesIO first and then let _load_bytes decide memory vs spill.
            # For seekable streams we can avoid the intermediate buffer entirely.
            if hasattr(data, "seek") and hasattr(data, "tell"):
                pos = data.tell()
                data.seek(0, io.SEEK_END)
                size = data.tell() - pos
                data.seek(pos)

                if size > self._cfg.spill_bytes:
                    path, fh = self._open_spill_file()
                    shutil.copyfileobj(data, fh, length=self._cfg.spill_bytes)
                    fh.flush()
                    self._path = path
                    self._file = fh
                else:
                    self._mem = io.BytesIO(data.read())

            else:
                # Non-seekable (pipes, network streams) — no choice but to drain first
                self._load_bytes(data.read())

        elif isinstance(data, BytesIO):
            self._mem = data._mem
            self._path = data._path
            self._file = data._file
        else:
            raise TypeError(
                f"{type(self).__name__} does not accept data of type {type(data)!r}. "
                "Pass bytes, bytearray, memoryview, BytesIO, "
                "a file-like object, or an AbstractDataPath."
            )

        self._mmap: mmap.mmap | None = None
        self._closed: bool = False

    # ──────────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────────

    def _load_bytes(self, payload: bytes) -> None:
        """
        Decide whether *payload* fits in memory or must be spilled immediately.
        Called only during __init__ so no existing state needs to be torn down.
        """
        if len(payload) > self._cfg.spill_bytes:
            path, fh = self._open_spill_file()
            fh.write(payload)
            fh.flush()
            # Leave cursor at the end — callers that want to read must seek(0)
            self._path = path
            self._file = fh
        else:
            self._mem = io.BytesIO(payload)

    def _open_spill_file(self) -> tuple[AbstractDataPath, IO[bytes]]:
        """
        Create and open a new temporary spill file according to *_cfg*.

        Returns the (path, writable-binary-file-handle) pair.
        The handle is opened in ``w+b`` mode so the same descriptor can later
        be used for both reading and writing without reopening.
        """
        tmp_dir: AbstractDataPath | None = self._cfg.tmp_dir

        # Build a unique filename using uuid4 so concurrent buffers never clash
        name = f"{self._cfg.prefix}{uuid.uuid4().hex}{self._cfg.suffix}"

        if tmp_dir is not None:
            spill_path: AbstractDataPath = tmp_dir / name
        else:
            # Fall back to the OS temp directory, represented as our path type
            spill_path = Path(tempfile.gettempdir()) / name

        fh: IO[bytes] = spill_path.open("w+b")
        return spill_path, fh

    # ---------------------------------------------------------------------
    # Dunder
    # ---------------------------------------------------------------------
    def __bool__(self):
        return self.size > 0

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def __iter__(self):
        return self.to_bytes().__iter__()

    def __len__(self) -> int:
        return self.size

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Factories
    # ---------------------------------------------------------------------
    @classmethod
    def parse_any(cls, obj: Any, config: BufferConfig | None = None) -> "BytesIO":
        """
        Create a DynamicBuffer from:
          - DynamicBuffer: returns as-is
          - io.BytesIO: wraps as in-mem
          - str/Path: opens as file-backed (w+b)
          - bytes-like / file-like: reads and stores
        """
        if isinstance(obj, cls):
            return obj

        buf = cls(config=config)

        if isinstance(obj, io.BytesIO):
            buf._mem = obj
            return buf

        if isinstance(obj, (str, Path)):
            # local path (no fancy LocalDataPath dependency here)
            p = Path(obj)
            buf._path = p
            buf._file = p.open("w+b", buffering=0)
            buf._mem = None
            return buf

        buf.write_any_bytes(obj)
        buf.seek(0)
        return buf

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------
    @property
    def spilled(self):
        return self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def size(self) -> int:
        if self._mem is not None:
            return self._mem.getbuffer().nbytes
        fh = self.buffer()
        try:
            return os.fstat(fh.fileno()).st_size
        except Exception:
            pos = fh.tell()
            fh.seek(0, io.SEEK_END)
            end = fh.tell()
            fh.seek(pos, io.SEEK_SET)
            return int(end)

    # ---------------------------------------------------------------------
    # io.RawIOBase interface
    # ---------------------------------------------------------------------
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

    def write(self, b: BytesLike) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed DynamicBuffer")

        try:
            mv = memoryview(b)
        except TypeError:
            if b is None:
                return 0

            if hasattr(b, "read"):
                b = b.read()

            if isinstance(b, str):
                b = b.encode("utf-8")

            mv = memoryview(b)

        n = len(mv)

        if self._mem is not None:
            current = self._mem.getbuffer().nbytes
            if current + n > self._cfg.spill_bytes:
                self._spill_to_file()

        return self.buffer().write(mv.tobytes())

    def write_any_bytes(self, obj: Any) -> int:
        """
        Accept bytes-like or file-like (must have .read()).
        """
        if obj is None:
            return 0
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return self.write(obj)
        if hasattr(obj, "read"):
            return self.write(obj.read())
        raise TypeError(f"Unsupported object for write_any_bytes: {type(obj)!r}")

    # ---------------------------------------------------------------------
    # Structured binary read/write (little-endian)
    # ---------------------------------------------------------------------
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
        n = self.read_uint32()
        return self._read_exact(n)

    def write_bytes_u32(self, data: BytesLike) -> int:
        mv = memoryview(data)
        total = self.write_uint32(len(mv))
        total += self.write(mv)
        return total

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        return self.write_bytes_u32(s.encode(encoding))

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------
    def xxh3_64(self) -> "xxhash.xxh3_64":
        from yggdrasil.xxhash import xxh3_64
        h = xxh3_64()
        h.update(self.memoryview())
        return h

    def blake3(self) -> "blake3.blake3":
        from yggdrasil.blake3 import blake3
        h = blake3(max_threads=blake3.AUTO)

        if self._mem is not None:
            # in-memory: just hash bytes
            h.update(self._mem.getbuffer())
            return h

        # spilled: mmap by path (fast)
        if self._path is None:
            # fallback: stream from fh
            self.buffer().seek(0)
            while chunk := self.buffer().read(8 * 1024 * 1024):
                h.update(chunk)
            return h

        h.update_mmap(str(self._path))
        return h

    def getvalue(self) -> bytes:
        """
        Return bytes. If spilled, reads the full file into memory.
        """
        if self._mem is not None:
            return self._mem.getvalue()
        return self.to_bytes()

    def memoryview(self) -> memoryview:
        """
        - In-memory: returns a view of the BytesIO bytes (copies once via getvalue()).
          If you want *true* zero-copy, store a bytearray instead of BytesIO.
        - Spilled: returns memoryview(mmap(file)).
        """
        if self._mem is not None:
            return memoryview(self._mem.getvalue())

        fh = self.buffer()
        fileno = fh.fileno()
        size = os.fstat(fileno).st_size
        if size == 0:
            return memoryview(b"")

        # refresh mmap if needed
        if self._mmap is None or self._mmap.closed:
            self._mmap = mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ)
        return memoryview(self._mmap)

    def to_bytes(self) -> bytes:
        fh = self.buffer()
        pos = fh.tell()
        try:
            fh.seek(0)
            return fh.read()
        finally:
            try:
                fh.seek(pos)
            except Exception:
                pass

    def open_reader(self) -> IO[bytes]:
        if self._mem is not None:
            return io.BytesIO(self._mem.getvalue())
        if self._path is None:
            raise RuntimeError("Spilled buffer missing path (unexpected)")
        return self._path.open("rb")

    def open_writer(self) -> IO[bytes]:
        return self.buffer()

    def cleanup(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return

        try:
            self.flush()
        except Exception:
            pass

        # Close mmap first (it pins the file on Windows)
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

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def buffer(self) -> IO[bytes]:
        if self._closed:
            raise ValueError("I/O operation on closed DynamicBuffer")
        if self._file is not None:
            return self._file
        if self._mem is not None:
            return self._mem
        raise RuntimeError("DynamicBuffer in invalid state")

    def _spill_to_file(self) -> None:
        """
        Create a temp file, copy existing mem contents, and switch to disk.
        Preserves current cursor position.
        """
        if self._mem is None:
            return

        cfg = self._cfg

        tmp_dir = str(cfg.tmp_dir) if cfg.tmp_dir is not None else None
        fd, name = tempfile.mkstemp(prefix=cfg.prefix, suffix=cfg.suffix, dir=tmp_dir)
        path = Path(name)

        f = os.fdopen(fd, "w+b", buffering=0)

        mem = self._mem
        mem_pos = mem.tell()
        mem.seek(0)
        f.write(mem.read())
        f.flush()
        f.seek(mem_pos)

        try:
            mem.close()
        except Exception:
            pass

        self._mem = None
        self._file = f
        self._path = path
