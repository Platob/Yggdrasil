# yggdrasil/pyutils/dynamic_buffer.py
from __future__ import annotations

import io
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, IO

__all__ = ["DynamicBuffer", "DynamicBufferConfig"]


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
    spill_bytes: int = 64 * 1024 * 1024  # 64 MiB
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
        fh = self._fh()
        pos = fh.tell()
        fh.seek(0, io.SEEK_END)
        end = fh.tell()
        fh.seek(pos, io.SEEK_SET)
        return end

    @property
    def keep_spilled_file(self):
        return self._cfg.keep_spilled_file

    @keep_spilled_file.setter
    def keep_spilled_file(self, value: bool):
        self._cfg.keep_spilled_file = value

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

        mv = memoryview(b)
        n = len(mv)

        # If we're still in memory, check whether this write pushes us over.
        if self._mem is not None:
            current_size = self._mem.getbuffer().nbytes
            if current_size + n > self._cfg.spill_bytes:
                self._spill_to_file()

        fh = self._fh()
        written = fh.write(mv.tobytes())
        return written

    def read_int64(self) -> int:
        data = self.read(8)
        return struct.unpack("<q", data)[0]

    def write_int64(self, value: int) -> int:
        # Arrow / most binary formats: little-endian signed 64-bit
        data = struct.pack("<q", value)
        return self.write(data)

    # -----------------------
    # Convenience APIs
    # -----------------------
    def getvalue(self) -> bytes:
        """
        Return bytes only if still in memory.
        Raises if spilled to disk (to avoid accidental huge loads).
        """
        if self._mem is None:
            raise RuntimeError("Buffer spilled to disk; use to_bytes() or open_reader()")
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
