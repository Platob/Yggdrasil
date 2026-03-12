# yggdrasil/io/buffer/bytes_io.py
"""Spill-to-disk byte buffer with transparent memory/path backing.

Design:
- Own logical cursor (self._pos), independent from any OS/file cursor
- In-memory backing is bytearray + logical size
- Spilled backing is a filesystem path only (no persistent Python file handle)
- Uses os.pread/os.pwrite for cursorless file IO
- Keeps optional readonly mmap for zero-copy reads when spilled

Semantics:
- By default, construction is deep-copying (`copy=True`)
- `BytesIO(BytesIO(...), copy=True)` duplicates backing/content
- `BytesIO(path, copy=True)` copies file content into owned backing
- `BytesIO(path, copy=False)` aliases the original path and mutates it
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
from typing import Any, IO, Optional, TYPE_CHECKING, Union

from yggdrasil.io.config import BufferConfig, DEFAULT_CONFIG
from yggdrasil.io.enums import MediaType, MimeType
from yggdrasil.io.types import BytesLike

if TYPE_CHECKING:
    import blake3
    import xxhash

    from .media_io import MediaIO
    from ..enums import Codec

__all__ = ["BytesIO", "BufferLike"]

BufferLike = Union[
    bytes, bytearray, memoryview, io.BytesIO, "BytesIO", str, Path, IO[bytes]
]
_HEAD_DEFAULT = 128
_COPY_CHUNK_SIZE = 8 * 1024 * 1024
_HAS_PREAD = hasattr(os, "pread")
_HAS_PWRITE = hasattr(os, "pwrite")

class BytesIO(io.RawIOBase):
    __slots__ = (
        "_cfg",
        "_buf",
        "_size",
        "_pos",
        "_path",
        "_mmap",
        "_closed",
        "_owns_path",
    )

    def __init__(
        self,
        data: IO[bytes] | bytes | bytearray | memoryview | str | Path | None = None,
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

        self._path: Path | None = None
        self._mmap: mmap.mmap | None = None
        self._owns_path: bool = False

        self._media_type = media_type

        self._closed: bool = False
        self._init_from(data, copy=copy)

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

        if isinstance(data, (str, Path)):
            self._init_from_path(Path(data), copy=copy)
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

    def _init_from_bytes(self, mv) -> None:
        n = len(mv)
        if n > self._cfg.spill_bytes:
            self._spill_from_bytes(mv)
        else:
            self._buf = bytearray(mv.tobytes() if not mv.contiguous else mv)
            self._size = n
            self._path = None
            self._owns_path = False
        self._pos = 0

    def _init_from_stdlib_bytesio(self, src: io.BytesIO) -> None:
        saved = src.tell()
        src.seek(0, io.SEEK_END)
        size = src.tell()
        src.seek(0)

        if size > self._cfg.spill_bytes:
            path = self._create_spill_path()
            with path.open("w+b") as fh:
                shutil.copyfileobj(src, fh, length=_COPY_CHUNK_SIZE)
                fh.flush()
            self._buf = None
            self._size = 0
            self._path = path
            self._owns_path = True
        else:
            payload = src.read()
            self._buf = bytearray(payload)
            self._size = len(payload)
            self._path = None
            self._owns_path = False

        src.seek(saved)
        self._pos = 0

    def _init_from_bytesio(self, src: BytesIO, *, copy: bool) -> None:
        if copy:
            self._init_from_bytes(memoryview(src.to_bytes()))
            self._pos = 0
            return

        if src._buf is not None:
            self._buf = src._buf
            self._size = src._size
            self._path = None
            self._owns_path = False
        else:
            self._buf = None
            self._size = 0
            self._path = src._path
            self._owns_path = False

        self._pos = src._pos

    def _init_from_path(self, path: Path, *, copy: bool) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        if copy:
            size = path.stat().st_size
            if size > self._cfg.spill_bytes:
                dst = self._create_spill_path()
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
        if hasattr(src, "seek") and hasattr(src, "tell"):
            saved = src.tell()
            src.seek(0, io.SEEK_END)
            end = src.tell()
            src.seek(saved)
            remaining = max(0, end - saved)

            if remaining > self._cfg.spill_bytes:
                path = self._create_spill_path()
                with path.open("w+b") as fh:
                    shutil.copyfileobj(src, fh, length=_COPY_CHUNK_SIZE)
                    fh.flush()
                self._buf = None
                self._size = 0
                self._path = path
                self._owns_path = True
            else:
                payload = src.read()
                self._buf = bytearray(payload)
                self._size = len(payload)
                self._path = None
                self._owns_path = False
        else:
            payload = src.read()
            self._init_from_bytes(memoryview(payload))
            return

        self._pos = 0

    # ------------------------------------------------------------------
    # Backing helpers
    # ------------------------------------------------------------------

    def _create_spill_path(self) -> Path:
        if self._path is None:
            tmp_dir = self._cfg.tmp_dir
            name = f"{self._cfg.prefix}{uuid.uuid4().hex}{self._cfg.suffix}"
            self._path = (tmp_dir / name) if tmp_dir is not None else (Path(tempfile.gettempdir()) / name)
        return self._path

    def _invalidate_mmap(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

    def _spill_from_bytes(self, mv) -> None:
        path = self._create_spill_path()
        with path.open("w+b") as fh:
            fh.write(mv.tobytes() if not mv.contiguous else mv)
            fh.flush()

        self._buf = None
        self._size = 0
        self._path = path
        self._owns_path = True
        self._invalidate_mmap()

    def spill_to_file(self) -> None:
        if self._buf is None:
            return

        cfg = self._cfg
        tmp_dir = str(cfg.tmp_dir) if cfg.tmp_dir is not None else None
        fd, name = tempfile.mkstemp(prefix=cfg.prefix, suffix=cfg.suffix, dir=tmp_dir)
        path = Path(name)

        try:
            if self._size:
                os.write(fd, memoryview(self._buf)[: self._size])
        finally:
            os.close(fd)

        self._buf = None
        self._size = 0
        self._path = path
        self._owns_path = True
        self._invalidate_mmap()

    def _reset_backing_keep_open(self) -> None:
        self._invalidate_mmap()

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

    def _replace_with_payload(self, payload: bytes) -> None:
        self._reset_backing_keep_open()
        self._init_from_bytes(memoryview(payload))
        self._pos = 0

    @staticmethod
    def _bytes_from_codec_output(out: Any) -> bytes:
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
    # Cursorless IO primitives
    # ------------------------------------------------------------------

    def pread(self, n: int, pos: int) -> bytes:
        if n <= 0:
            return b""

        if self._buf is not None:
            end = min(pos + n, self._size)
            if pos >= end:
                return b""
            return bytes(memoryview(self._buf)[pos:end])

        if self._path is None:
            return b""

        with self._path.open("rb") as fh:
            if _HAS_PREAD:
                return os.pread(fh.fileno(), n, pos)

            fh.seek(pos)
            return fh.read(n)

    def pwrite(self, mv: memoryview, pos: int) -> int:
        if len(mv) == 0:
            return 0

        if self._buf is not None:
            need = pos + len(mv)
            if need > len(self._buf):
                new_cap = max(need, int(len(self._buf) * 1.5) + 1)
                self._buf.extend(b"\x00" * (new_cap - len(self._buf)))

            memoryview(self._buf)[pos: pos + len(mv)] = mv
            self._size = max(self._size, need)
            return len(mv)

        if self._path is None:
            raise RuntimeError("No backing store available for write")

        with self._path.open("r+b") as fh:
            if _HAS_PWRITE:
                return int(os.pwrite(fh.fileno(), mv, pos))

            fh.seek(pos)
            written = fh.write(mv)
            fh.flush()
            return len(mv) if written is None else int(written)

    def _ensure_spill_for_growth(self, extra: int) -> None:
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
    def path(self) -> Path | None:
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
            self._media_type = MediaType.parse_io(self, MediaType(MimeType.OCTET_STREAM))
        return self._media_type

    @media_type.setter
    def media_type(self, value: "MediaType"):
        self.set_media_type(value, safe=True)

    def set_media_type(
        self,
        value: MediaType,
        *,
        safe: bool
    ) -> "BytesIO":
        from ..enums import MediaType
        self._media_type = MediaType.parse(value)

    def media_io(
        self,
        media: Optional[MediaType] = None
    ) -> "MediaIO":
        from .media_io import MediaIO

        media = MediaType.parse(media)
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
        return None

    def head(self, n: int = _HEAD_DEFAULT):
        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")
        if n <= 0 or self.size == 0:
            return memoryview(b"")

        if self._buf is not None:
            return memoryview(self._buf)[: min(n, self._size)]

        return memoryview(self.pread(n, 0))

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

        if self._buf is None:
            self._invalidate_mmap()

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

        with self._path.open("r+b") as fh:
            fh.truncate(size)

        if self._pos > size:
            self._pos = size
        self._invalidate_mmap()
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
        from yggdrasil.xxhash import xxh3_64

        h = xxh3_64()
        h.update(self.memoryview())
        return h

    def xxh3_int64(self) -> int:
        u = self.xxh3_64().intdigest()
        return u if u < 2 ** 63 else u - 2 ** 64

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
        text: bool = False,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: str = "",
        start: int = 0,
        length: int | None = None,
    ):
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

    def memoryview(self):
        if self._buf is not None:
            return memoryview(self._buf)[: self._size]

        if self._path is None or not self._path.exists():
            return memoryview(b"")

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

    def open_reader(self) -> IO[bytes]:
        if self._buf is not None:
            return io.BytesIO(bytes(memoryview(self._buf)[: self._size]))
        if self._path is None:
            raise RuntimeError("Spilled buffer has no path")
        return self._path.open("rb")

    def to_arrow_io(self, mode: str = "r"):
        from yggdrasil.arrow.lib import pyarrow as pa

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if "r" in mode:
            if self.spilled and self._path is not None:
                return pa.memory_map(str(self._path), "r")

            mv = self.memoryview()
            buf = pa.py_buffer(mv) if len(mv) else pa.py_buffer(b"")
            return pa.BufferReader(buf)

        if "w" in mode or "a" in mode:
            if not self.spilled:
                self.spill_to_file()

            if self._path is None:
                raise RuntimeError("Failed to materialize spill file for Arrow IO")

            if "w" in mode:
                return pa.OSFile(str(self._path), mode="w")
            return pa.OSFile(str(self._path), mode="a")

        raise ValueError(f"Unsupported mode for to_arrow_io: {mode!r}")

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    def compress(self, codec: "Codec | str", *, copy: bool = False) -> "BytesIO":
        from ..enums.codec import Codec as _Codec

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        c = _Codec.parse(codec)
        if c is None:
            raise ValueError(f"Unknown codec: {codec!r}")

        out_std = c.compress(self)
        payload = self._bytes_from_codec_output(out_std)

        if copy:
            out = BytesIO(payload, config=self._cfg)
            out.seek(0)
            return out

        self._replace_with_payload(payload)
        return self

    def decompress(self, codec: "Codec | str | None" = "infer", *, copy: bool = False) -> "BytesIO":
        from ..enums.codec import Codec as _Codec
        from ..enums.codec import detect as _detect

        if self._closed:
            raise ValueError("I/O operation on closed BytesIO")

        if codec is None or (isinstance(codec, str) and codec.strip().lower() == "infer"):
            c = _detect(self)
            if c is None:
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

        out_std = c.open(self)
        payload = self._bytes_from_codec_output(out_std)

        if copy:
            out = BytesIO(payload, config=self._cfg)
            out.seek(0)
            return out

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

        self._invalidate_mmap()

        old_path = self._path
        old_owns_path = self._owns_path

        self._buf = None
        self._size = 0
        self._pos = 0
        self._path = None
        self._owns_path = False
        self._closed = True

        if old_path is not None and old_owns_path and not self._cfg.keep_spilled_file:
            try:
                old_path.unlink(missing_ok=True)
            except Exception:
                pass

        super().close()