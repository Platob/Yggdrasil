"""File-like IO abstractions for Databricks paths.

Each subclass of :class:`DatabricksIO` implements ``read_byte_range`` and
``write_all_bytes`` for exactly one Databricks namespace (DBFS, Workspace,
Volumes).  The base class provides the full buffered read/write/seek/flush
layer on top.

Buffers use :class:`yggdrasil.io.BytesIO` (``_Buffer``) which spills to
disk transparently for large payloads.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import TYPE_CHECKING, Optional, IO, AnyStr, Union, BinaryIO

from yggdrasil.io.buffer import BytesIO as _Buffer

from databricks.sdk.errors import InternalError
from databricks.sdk.errors.platform import (
    NotFound,
    ResourceDoesNotExist,
    BadRequest,
)
from databricks.sdk.service.workspace import ImportFormat, ExportFormat

from .path_kind import DatabricksPathKind
from ...pyutils.retry import retry

if TYPE_CHECKING:
    from .path import DatabricksPath

__all__ = ["DatabricksIO"]

LOGGER = logging.getLogger(__name__)
_COPY_CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helper: normalise arbitrary data into (_Buffer, size, owns_buffer)
# ---------------------------------------------------------------------------

def _prepare_buffer(
    data: Union[bytes, bytearray, memoryview, BinaryIO],
) -> tuple[int, _Buffer, bool]:
    """Wrap *data* in a :class:`_Buffer` and return ``(size, buf, True)``."""
    buf = _Buffer(data)
    return buf.size, buf, True


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class DatabricksIO(ABC, IO):
    """Buffered file-like interface for Databricks workspace/volume/DBFS paths.

    The buffer is a :class:`_Buffer` (``yggdrasil.io.BytesIO``) which spills
    transparently to disk for large payloads — no OOM risk.
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        path: "DatabricksPath",
        mode: str,
        encoding: str | None = None,
        position: int = 0,
        buffer: Optional[_Buffer] = None,
    ):
        super().__init__()
        self.encoding = encoding
        self.mode = mode          # triggers the property setter below
        self.path = path
        self._buffer: Optional[_Buffer] = buffer
        self.position = position
        self._write_flag = False

    # ── Factory: dispatch on path.kind ────────────────────────────────

    @classmethod
    def create_instance(
        cls,
        path: "DatabricksPath",
        mode: str,
        encoding: str | None = None,
        position: int = 0,
        buffer: Optional[_Buffer] = None,
    ) -> "DatabricksIO":
        """Return the right IO subclass for *path.kind*."""
        _map = {
            DatabricksPathKind.VOLUME:    DatabricksVolumeIO,
            DatabricksPathKind.DBFS:      DatabricksDBFSIO,
            DatabricksPathKind.WORKSPACE: DatabricksWorkspaceIO,
        }
        klass = _map.get(path.kind)
        if klass is None:
            raise ValueError(f"Unsupported path kind: {path.kind}")
        return klass(
            path=path, mode=mode, encoding=encoding,
            position=position, buffer=buffer,
        )

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> "DatabricksIO":
        return self.connect(clone=False)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ── Dunder helpers ────────────────────────────────────────────────

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __len__(self):
        return self.content_length or 0

    def __iter__(self):
        return iter(self.read_all_bytes())

    def __hash__(self):
        return hash(self.path)

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path!r})"

    __repr__ = __str__

    # ── Properties ────────────────────────────────────────────────────

    @property
    def workspace(self):
        return self.path.workspace

    @property
    def name(self):
        return self.path.name

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        self._mode = value
        if "b" in value:
            self.encoding = None
        elif self.encoding is None:
            self.encoding = "utf-8"

    @property
    def content_length(self) -> int:
        return self.path.content_length

    @content_length.setter
    def content_length(self, value: int):
        self.path.content_length = value

    # ── Buffer management ─────────────────────────────────────────────

    @property
    def buffer(self) -> _Buffer:
        """Lazily create or return the spill-to-disk buffer."""
        if self._buffer is None:
            self._buffer = _Buffer()
            self._buffer.seek(self.position, io.SEEK_SET)
        return self._buffer

    @buffer.setter
    def buffer(self, value: Optional[_Buffer]) -> None:
        self._buffer = value

    def clear_buffer(self):
        self._buffer = None

    # ── Clone / connect ───────────────────────────────────────────────

    def clone_instance(self, **kwargs) -> "DatabricksIO":
        return self.__class__(
            path=kwargs.get("path", self.path),
            mode=kwargs.get("mode", self.mode),
            encoding=kwargs.get("encoding", self.encoding),
            position=kwargs.get("position", self.position),
            buffer=kwargs.get("buffer", self._buffer),
        )

    @property
    def connected(self) -> bool:
        return self.path.connected

    def connect(self, clone: bool = False) -> "DatabricksIO":
        path = self.path.connect()
        if clone:
            return self.clone_instance(path=path)
        self.path = path
        return self

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self, flush: bool = True):
        if flush:
            self.flush()
        self.clear_buffer()

    @property
    def closed(self) -> bool:
        return False

    def fileno(self) -> int:
        return hash(self)

    def isatty(self) -> bool:
        return False

    # ── Seek / tell ───────────────────────────────────────────────────

    def tell(self) -> int:
        return self.position

    def seekable(self) -> bool:
        return True

    def seek(self, offset, whence=0, /):
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self.position + offset
        elif whence == io.SEEK_END:
            new = self.content_length + offset
        else:
            raise ValueError("Invalid whence value")
        if new < 0:
            raise ValueError("Negative seek position")
        if self._buffer is not None:
            self._buffer.seek(new, io.SEEK_SET)
        self.position = new
        return self.position

    # ── Read ──────────────────────────────────────────────────────────

    def readable(self) -> bool:
        return True

    @abstractmethod
    def read_byte_range(
        self, start: int, length: int, allow_not_found: bool = False
    ) -> bytes:
        """Read *length* bytes starting at *start* from the remote file."""

    def read_all_bytes(
        self, use_cache: bool = True, allow_not_found: bool = False
    ) -> bytes:
        """Return the full file contents, optionally caching in the buffer."""
        if use_cache and self._buffer is not None:
            cached = self._buffer.to_bytes()
            if len(cached) == self.content_length:
                return cached
            self._buffer.close()
            self._buffer = None

        data = self.read_byte_range(
            0, self.content_length, allow_not_found=allow_not_found,
        )
        self.content_length = len(data)

        if use_cache and self._buffer is None:
            self._buffer = _Buffer(data)
            self._buffer.seek(self.position, io.SEEK_SET)
        return data

    def getvalue(self) -> bytes:
        if self._buffer is not None:
            return self._buffer.to_bytes()
        return self.read_all_bytes()

    def getbuffer(self) -> _Buffer:
        return self.buffer

    def read(self, n: int = -1, use_cache: bool = True):
        cur = self.position
        blob = self.read_all_bytes(use_cache=use_cache)
        if n == -1:
            n = self.content_length - cur
        chunk = blob[cur:cur + n]
        self.position += len(chunk)
        return chunk.decode(self.encoding) if self.encoding else chunk

    def readline(self, limit: int = -1, use_cache: bool = True):
        if self.encoding:
            chars = []
            read = 0
            while limit == -1 or read < limit:
                ch = self.read(1, use_cache=use_cache)
                if not ch:
                    break
                chars.append(ch)
                read += 1
                if ch == "\n":
                    break
            return "".join(chars)

        buf = bytearray()
        read = 0
        while limit == -1 or read < limit:
            b = self.read(1, use_cache=use_cache)
            if not b:
                break
            buf.extend(b)
            read += 1
            if b == b"\n":
                break
        return bytes(buf)

    def readlines(self, hint: int = -1, use_cache: bool = True):
        lines, total = [], 0
        while True:
            line = self.readline(use_cache=use_cache)
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint != -1 and total >= hint:
                break
        return lines

    # ── Write ─────────────────────────────────────────────────────────

    def writable(self) -> bool:
        return True

    @abstractmethod
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        """Write *data* to the remote path (replaces content)."""

    def truncate(self, size=None, /):
        if size is None:
            size = self.position
        if self._buffer is None:
            return self.write_all_bytes(data=b"\x00" * size)
        self._buffer.truncate(size)
        self.content_length = size
        self._write_flag = True
        return size

    def _need_flush(self) -> bool:
        return self._write_flag and self._buffer is not None

    def flush(self):
        if self._need_flush():
            self.write_all_bytes(data=self._buffer.to_bytes())
            self._write_flag = False

    def write(self, data: AnyStr) -> int:
        if isinstance(data, str):
            data = data.encode(self.encoding or "utf-8")
        written = self.buffer.write(data)
        self.position += written
        self.content_length = self.position
        self._write_flag = True
        return written

    def writelines(self, lines) -> None:
        for line in lines:
            if isinstance(line, str):
                line = line.encode(self.encoding or "utf-8")
            elif not isinstance(line, (bytes, bytearray)):
                raise TypeError(f"expected bytes-like or str, not {type(line).__name__!r}")
            payload = line if line.endswith(b"\n") else line + b"\n"
            self.write(payload)

    # ── Misc ──────────────────────────────────────────────────────────

    def size(self) -> int:
        return self.content_length

    def get_output_stream(self, *args, **kwargs):
        return self

    def copy_to(self, dest: Union["DatabricksIO", "DatabricksPath", str]) -> None:
        data = self.read_all_bytes(use_cache=False)
        if isinstance(dest, DatabricksIO):
            dest.write_all_bytes(data=data)
        elif hasattr(dest, "write"):
            dest.write(data)
        else:
            from .path import DatabricksPath
            dest_path = DatabricksPath.parse(dest, client=self.workspace)
            with dest_path.open(mode="wb") as d:
                self.copy_to(dest=d)

    def _reset_for_write(self):
        if self._buffer is not None:
            self._buffer.seek(0, io.SEEK_SET)
            self._buffer.truncate(0)
        else:
            self._buffer = _Buffer()
        self.position = 0
        self.content_length = 0
        self._write_flag = True


# ═══════════════════════════════════════════════════════════════════════════
# Workspace  (/Workspace/…)
# ═══════════════════════════════════════════════════════════════════════════

class DatabricksWorkspaceIO(DatabricksIO):
    """IO for Databricks Workspace files — Workspace API."""

    @retry(exceptions=(InternalError,))
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        if length == 0:
            return b""
        sdk = self.workspace.workspace_client()
        result = sdk.workspace.download(
            path=self.path.full_path(),
            format=ExportFormat.AUTO,
        )
        if result is None:
            return b""
        data = result.read()
        return data[start:start + length]

    @retry(exceptions=(InternalError,))
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        sdk = self.workspace.workspace_client()
        full_path = self.path.full_path()

        if isinstance(data, bytes):
            bsize = len(data)
        elif isinstance(data, _Buffer):
            bsize = data.size
        elif isinstance(data, io.BytesIO):
            bsize = len(data.getvalue())
        else:
            bsize = None

        LOGGER.debug("Writing %s (size=%s) to %s", type(data).__name__, bsize, self)

        try:
            sdk.workspace.upload(full_path, data, format=ImportFormat.AUTO, overwrite=True)
        except (NotFound, ResourceDoesNotExist, BadRequest):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            sdk.workspace.upload(full_path, data, format=ImportFormat.AUTO, overwrite=True)

        self.path.reset_metadata(is_file=True, is_dir=False, size=bsize, mtime=time.time())
        LOGGER.info("Written %s bytes to %s", bsize, self)
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Volume  (/Volumes/…)
# ═══════════════════════════════════════════════════════════════════════════

class DatabricksVolumeIO(DatabricksIO):
    """IO for Unity Catalog volume files — Files API."""

    @retry(exceptions=(InternalError,))
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        if length <= 0:
            return b""
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")

        sdk = self.workspace.workspace_client()
        try:
            resp = sdk.files.download(self.path.full_path())
        except (NotFound, ResourceDoesNotExist, BadRequest, InternalError):
            if allow_not_found:
                return b""
            raise
        data = resp.contents.read()
        if start >= len(data):
            return b""
        return data[start:start + length]

    def write_all_bytes(
        self,
        data: Union[bytes, bytearray, memoryview, BinaryIO],
        *,
        overwrite: bool = True,
        part_size: int | None = None,
        use_parallel: bool = True,
        parallelism: int | None = None,
    ):
        sdk = self.workspace.workspace_client()
        full_path = self.path.full_path()

        LOGGER.debug("Writing all bytes to %s", self)
        size, bio, should_close = _prepare_buffer(data)

        def _upload():
            return sdk.files.upload(
                full_path, bio, overwrite=overwrite,
                part_size=part_size, use_parallel=use_parallel,
                parallelism=parallelism,
            )

        try:
            _upload()
        except (NotFound, ResourceDoesNotExist, BadRequest, InternalError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                bio.seek(0)
            except Exception:
                pass
            _upload()
        finally:
            if should_close:
                try:
                    bio.close()
                except Exception:
                    pass

        self.path.reset_metadata(is_file=True, is_dir=False, size=size, mtime=time.time())
        LOGGER.info("Written %s bytes to %s", size or "all", self.path)
        return self


# ═══════════════════════════════════════════════════════════════════════════
# DBFS  (/dbfs/…)
# ═══════════════════════════════════════════════════════════════════════════

class DatabricksDBFSIO(DatabricksIO):
    """IO for DBFS files — DBFS API with base-64 chunked reads."""

    @retry(exceptions=(InternalError,))
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        if length == 0:
            return b""
        sdk = self.workspace.workspace_client()
        full_path = self.path.full_path()

        result = bytearray()
        remaining = length
        pos = start
        while remaining > 0:
            chunk_size = min(remaining, 2 * 1024 * 1024)
            resp = sdk.dbfs.read(path=full_path, offset=pos, length=chunk_size)
            if not resp.data:
                break
            decoded = base64.b64decode(resp.data)
            result.extend(decoded)
            pos += len(decoded)
            remaining -= len(decoded)
        return bytes(result)

    @retry(exceptions=(InternalError,))
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        sdk = self.workspace.workspace_client()
        full_path = self.path.full_path()

        LOGGER.debug("Writing all bytes to %s", self)
        try:
            with sdk.dbfs.open(path=full_path, read=False, write=True, overwrite=True) as f:
                f.write(data)
        except (NotFound, ResourceDoesNotExist, BadRequest):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sdk.dbfs.open(path=full_path, read=False, write=True, overwrite=True) as f:
                f.write(data)

        LOGGER.info("Written all bytes to %s", self)
        self.path.reset_metadata(is_file=True, is_dir=False, size=len(data), mtime=time.time())

