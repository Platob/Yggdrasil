# yggdrasil/pyutils/dynamic_buffer.py
from __future__ import annotations

import io
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, IO, Any, TYPE_CHECKING


__all__ = ["DynamicBuffer", "DynamicBufferConfig"]


if TYPE_CHECKING:
    import xxhash


@dataclass(frozen=False, slots=True)
class DynamicBufferConfig:
    """
    Configuration for DynamicBuffer.

    spill_bytes:
        When buffered data size exceeds this threshold, switch from in-memory BytesIO
        to a real temporary file on disk.

    tmp_dir:
        Directory where spill files are created. If None, uses system temp dir.

    prefix/suffix:
        Naming for the temp file when spilling to disk.

    keep_spilled_file:
        If True, spilled temp files are not deleted on close (you can call cleanup()).
        If False, we delete the temp file on close/cleanup().
    """
    spill_bytes: int = 128 * 1024 * 1024  # 64 MiB
    tmp_dir: Optional[Path] = None
    prefix: str = "tmp-"
    suffix: str = ".bin"
    keep_spilled_file: bool = False


class DynamicBuffer(io.RawIOBase):
    """
    A bytes buffer that starts in memory and spills to a local temp file
    when it grows beyond a threshold.

    Practical uses:
      - serialization buffers (dill/pickle/parquet writes)
      - staging upload payloads without blowing RAM
      - "read as bytes" semantics while supporting large payloads

    Key properties:
      - file-like: supports write(), read(), seek(), tell(), flush(), close()
      - getvalue() works only if still in-memory (else raises)
      - to_bytes() returns bytes (loads entire payload; be careful for huge buffers)
      - path returns the spill file path if spilled, else None
      - underlying IO always positioned at end after write()

    Notes:
      - This is a byte buffer (binary mode).
      - Not thread-safe (like most IO objects).
    """

    def __init__(
        self,
        config: DynamicBufferConfig | None = None
    ) -> None:
        super().__init__()
        self._cfg = config or DynamicBufferConfig()
        if self._cfg.spill_bytes < 1:
            raise ValueError("spill_bytes must be >= 1")

        self._mem: io.BytesIO | None = io.BytesIO()
        self._file: IO[bytes] | None = None
        self._path: Path | None = None
        self._closed: bool = False

    def __bytes__(self) -> bytes:
        return self.to_bytes()

    def __len__(self) -> int:
        """
        Total payload size in bytes (does NOT change the current cursor).
        """
        return self.size

    def __del__(self):
        self.close()

    @classmethod
    def parse_any(
        cls,
        obj: Any,
        config: DynamicBufferConfig | None = None
    ):
        if isinstance(obj, cls):
            return obj

        buffer = DynamicBuffer(config=config)

        if isinstance(obj, io.BytesIO):
            obj._mem = obj
        elif isinstance(obj, (str, Path)):
            from .path import LocalDataPath

            obj._path = LocalDataPath(obj)
            obj._file = obj._path.open("w+b", buffering=0)
        else:
            buffer.write_any_bytes(obj)
            buffer.seek(0)

        return buffer

    # -----------------------
    # Introspection
    # -----------------------
    @property
    def spilled(self) -> bool:
        return self._file is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def size(self) -> int:
        if self._mem is not None:
            # BytesIO: safest is buffer length; avoids messing with cursor.
            return self._mem.getbuffer().nbytes

        try:
            return os.fstat(self._file.fileno()).st_size
        except Exception:
            # fallback: preserve cursor, use seek/tell
            pos = self._file.tell()
            self._file.seek(0, io.SEEK_END)
            end = self._file.tell()
            self._file.seek(pos, io.SEEK_SET)
            return int(end)

    # -----------------------
    # io.RawIOBase interface
    # -----------------------
    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
        except Exception:
            pass

        try:
            if self._mem is not None:
                self._mem.close()
        finally:
            self._mem = None

        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None

        if self._path is not None and not self._cfg.keep_spilled_file:
            try:
                self._path.unlink(missing_ok=True)  # py3.8+ ok
            except Exception:
                pass

        self._closed = True
        super().close()

    def flush(self) -> None:
        fh = self._fh()
        try:
            fh.flush()
        except Exception:
            pass

    def tell(self) -> int:
        return self._fh().tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._fh().seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        return self._fh().read(size)

    def write(self, b: Union[bytes, bytearray, memoryview]) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed DynamicBuffer")

        try:
            mv = memoryview(b)
        except TypeError:
            if isinstance(b, str):
                mv = memoryview(b.encode("utf-8"))
            else:
                raise

        n = len(mv)

        # If we're still in memory, check whether this write pushes us over.
        if self._mem is not None:
            current_size = self._mem.getbuffer().nbytes
            if current_size + n > self._cfg.spill_bytes:
                self._spill_to_file()

        fh = self._fh()
        written = fh.write(mv.tobytes())
        return written

    def write_any_bytes(self, obj: Any):
        if obj is None:
            return 0

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return self.write(obj)

        blob = obj.read()

        return self.write(blob)

    # -----------------------
    # Structured binary read/write (little-endian)
    # -----------------------

    def _read_exact(self, n: int) -> bytes:
        data = self.read(n)
        if len(data) != n:
            raise EOFError(f"expected {n} bytes, got {len(data)}")
        return data

    # ---- ints ----

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

    # ---- floats ----

    def read_f32(self) -> float:
        return struct.unpack("<f", self._read_exact(4))[0]

    def write_f32(self, value: float) -> int:
        return self.write(struct.pack("<f", float(value)))

    def read_f64(self) -> float:
        return struct.unpack("<d", self._read_exact(8))[0]

    def write_f64(self, value: float) -> int:
        return self.write(struct.pack("<d", float(value)))

    # ---- bool ----

    def read_bool(self) -> bool:
        return bool(self.read_uint8())

    def write_bool(self, value: bool) -> int:
        return self.write_uint8(1 if value else 0)

    # ---- bytes / strings (length-prefixed) ----

    def read_bytes_u32(self) -> bytes:
        """
        Read: u32 length (little-endian) + payload bytes.
        """
        n = self.read_uint32()
        return self._read_exact(n)

    def write_bytes_u32(self, data: bytes | bytearray | memoryview) -> int:
        mv = memoryview(data)
        total = self.write_uint32(len(mv))
        total += self.write(mv)
        return total

    def read_str_u32(self, encoding: str = "utf-8") -> str:
        return self.read_bytes_u32().decode(encoding)

    def write_str_u32(self, s: str, encoding: str = "utf-8") -> int:
        b = s.encode(encoding)
        return self.write_bytes_u32(b)

    # -----------------------
    # Convenience APIs
    # -----------------------
    def xxh3_64(self) -> "xxhash.xxh3_64":
        from ..xxhash import xxhash

        h = xxhash.xxh3_64()
        h.update(self.to_bytes())
        return h

    def getvalue(self) -> bytes:
        """
        Return bytes only if still in memory.
        Raises if spilled to disk (to avoid accidental huge loads).
        """
        if self._mem is None:
            return self.to_bytes()
        return self._mem.getvalue()

    def to_bytes(self) -> bytes:
        """
        Read the entire buffer into memory.
        Be careful: if you spilled, this could be huge.
        """
        fh = self._fh()
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
        """
        Return a new readable handle positioned at start.
        - If in-memory: returns a new BytesIO copy
        - If spilled: opens the spill file in 'rb'
        """
        if self._mem is not None:
            return io.BytesIO(self._mem.getvalue())
        if self._path is None:
            raise RuntimeError("Spilled buffer missing path (unexpected)")
        return open(self._path, "rb")

    def open_writer(self) -> IO[bytes]:
        """
        Return the underlying writable handle (position preserved).
        Mostly for advanced scenarios.
        """
        return self._fh()

    def cleanup(self) -> None:
        """
        Explicit cleanup (same behavior as close regarding file deletion).
        Safe to call multiple times.
        """
        self.close()

    # -----------------------
    # Internals
    # -----------------------
    def _fh(self) -> IO[bytes]:
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

        # We want a binary file handle with random access.
        # Use os.fdopen so the fd is managed correctly.
        f = os.fdopen(fd, "w+b", buffering=0)

        # Copy bytes + preserve cursor position
        mem = self._mem
        mem_pos = mem.tell()
        mem.seek(0)
        f.write(mem.read())
        f.flush()
        f.seek(mem_pos)

        # Switch backing store
        mem.close()
        self._mem = None
        self._file = f
        self._path = path
