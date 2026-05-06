""":class:`DBFSPath` — ``/dbfs/...`` paths via the DBFS REST API.

DBFS has two access modes:

- **FUSE mount** — on a Databricks cluster, ``/dbfs`` is a real
  kernel-mounted filesystem. ``open("/dbfs/foo", ...)`` works as
  a regular OS file. We expose this via :attr:`is_local` (becomes
  True when the FUSE mount is detected) and override ``pread`` /
  ``pwrite`` / ``read_bytes`` / ``write_bytes`` to use direct
  ``os.*`` syscalls.
- **REST API** — off-cluster (developer laptop, generic VM) or
  when the FUSE mount isn't present, all I/O goes through chunked
  ``sdk.dbfs.read`` / ``sdk.dbfs.open(write=True)``.

Whether FUSE is available is a runtime probe that fires once per
process and caches on the class.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import ClassVar, Optional, Union

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from ._errors import (
    ALREADY_EXISTS_ERRORS,
    NOT_FOUND_ERRORS,
    SDK_ERRORS,
    retry_sdk_call,
)
from .path import DatabricksPath
from .path_kind import DatabricksPathKind

__all__ = ["DBFSPath"]


LOGGER = logging.getLogger(__name__)


# Hard cap on a single ``dbfs.read`` — requesting more returns
# ``BadRequest``. Drives the chunk loop in :meth:`_remote_download`
# (and the chunked ``pread`` for non-FUSE mode).
_DBFS_CHUNK = 1 * 1024 * 1024


class DBFSPath(DatabricksPath):
    """Path under ``/dbfs/...`` via the DBFS REST API, optionally FUSE-fast."""

    scheme: ClassVar[str] = "dbfs+fuse"
    _NAMESPACE_PREFIX: ClassVar[str] = "/dbfs/"

    # Cached FUSE-mount probe. ``None`` = not yet probed.
    _DBFS_FUSE_CACHED: ClassVar[Optional[bool]] = None

    @property
    def kind(self) -> DatabricksPathKind:
        return DatabricksPathKind.DBFS

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = self.url.path.lstrip("/")
        return "/dbfs/" + p if p else "/dbfs"

    # ==================================================================
    # FUSE detection — runtime probe, cached per-process
    # ==================================================================

    @classmethod
    def _probe_dbfs_fuse_mount(cls) -> bool:
        """Probe whether ``/dbfs`` is the DBFS FUSE mount."""
        if cls._DBFS_FUSE_CACHED is not None:
            return cls._DBFS_FUSE_CACHED

        try:
            present = os.path.isdir("/dbfs")
            if present:
                # Standard Databricks layout: /dbfs/tmp and
                # /dbfs/FileStore are always there on a real mount.
                present = (
                    os.path.isdir("/dbfs/tmp")
                    or os.path.isdir("/dbfs/FileStore")
                )
        except Exception:
            present = False

        cls._DBFS_FUSE_CACHED = present
        return present

    @property
    def local_os_path(self) -> Optional[str]:
        """Local-FS path when ``/dbfs`` is FUSE-mounted, else ``None``."""
        if not self._probe_dbfs_fuse_mount():
            return None
        return self.full_path()

    @property
    def is_local(self) -> bool:
        # Toggles with the FUSE mount. When True, BytesIO routes
        # through the local-fd fast path instead of the
        # transaction-buffer mode.
        return self.local_os_path is not None

    # ==================================================================
    # Bytes I/O — FUSE fast path or chunked REST
    # ==================================================================

    def read_bytes(self, *, raise_error: bool = True) -> bytes:
        """FUSE: stdlib ``open().read()``. Else: chunked REST."""
        local = self.local_os_path
        if local is not None:
            try:
                with open(local, "rb") as fh:
                    return fh.read()
            except (OSError, ValueError):
                if raise_error:
                    raise
                return b""

        return super().read_bytes(raise_error=raise_error)

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview],
        *,
        mode: str = "wb",
        parents: bool = True,
    ) -> int:
        """FUSE: stdlib ``open().write()``. Else: SDK upload."""
        del mode

        local = self.local_os_path
        if local is not None:
            payload = bytes(data)
            if parents:
                parent = os.path.dirname(local)
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
            with open(local, "wb") as fh:
                fh.write(payload)
            return len(payload)

        return super().write_bytes(data, parents=parents)

    # ==================================================================
    # Positional IO — FUSE fast paths via os.pread/pwrite
    # ==================================================================

    def pread(self, n: int, pos: int, *, default=...) -> bytes:
        """FUSE: ``os.pread`` on a transient fd. Else: chunked REST.

        Non-FUSE pread is the rare case where the SDK actually
        gives us range-IO — chunk via ``offset`` / ``length`` on
        ``dbfs.read`` rather than downloading the whole object.
        """
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        if n == 0:
            return b""

        local = self.local_os_path
        if local is not None and hasattr(os, "pread"):
            try:
                fd = os.open(local, os.O_RDONLY)
            except OSError:
                if default is ...:
                    raise
                return default
            try:
                if n < 0:
                    # Read to EOF.
                    chunks = []
                    offset = pos
                    while True:
                        chunk = os.pread(fd, _DBFS_CHUNK, offset)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        offset += len(chunk)
                    return b"".join(chunks)
                return os.pread(fd, n, pos)
            finally:
                os.close(fd)

        # Non-FUSE: range-read via chunked REST.
        return self._sdk_chunked_read(n, pos, default=default)

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        """FUSE: ``os.pwrite`` on a transient fd. Else: read-modify-write."""
        mv = memoryview(data)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        n = len(mv)
        if n == 0:
            return 0
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")

        local = self.local_os_path
        if local is not None and hasattr(os, "pwrite"):
            if parents:
                parent = os.path.dirname(local)
                if parent and not os.path.isdir(parent):
                    os.makedirs(parent, exist_ok=True)
            # O_CREAT so a fresh path can grow from a positional
            # write; no O_TRUNC since we patch in place.
            flags = os.O_WRONLY | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            fd = os.open(local, flags, 0o644)
            try:
                total = 0
                while total < n:
                    written = os.pwrite(fd, bytes(mv[total:]), pos + total)
                    if written == 0:
                        break
                    total += written
                return total
            finally:
                os.close(fd)

        # Non-FUSE: fall back to RMW via the base class (uses our
        # SDK-direct read_bytes / write_bytes).
        return super().pwrite(data, pos, parents=parents)

    # ==================================================================
    # SDK transport
    # ==================================================================

    def _remote_download(self, allow_not_found: bool = False) -> BytesIO:
        """Drain the full object via base64-chunked ``dbfs.read``.

        Hits the 1 MiB cap per call. Stops when content_length is
        reached (if known) or when a short page signals EOF.
        Streams into a project :class:`BytesIO` so very large
        objects spill to disk instead of accumulating in RAM.
        """
        try:
            size = self.size
        except Exception:
            size = 0

        sdk = self._sdk()
        full_path = self.full_path()
        out = BytesIO()
        pos = 0
        target = size if size > 0 else None

        try:
            while True:
                chunk_size = _DBFS_CHUNK
                if target is not None:
                    remaining = target - pos
                    if remaining <= 0:
                        break
                    chunk_size = min(_DBFS_CHUNK, remaining)
                resp = retry_sdk_call(
                    sdk.dbfs.read,
                    path=full_path, offset=pos, length=chunk_size,
                )
                if not resp.data:
                    break
                decoded = base64.b64decode(resp.data)
                if not decoded:
                    break
                out.write(decoded)
                pos += len(decoded)
                if target is None and len(decoded) < chunk_size:
                    break
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise FileNotFoundError(self.full_path())
        except SDK_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise

        out.seek(0)
        return out

    def _sdk_chunked_read(self, n: int, pos: int, *, default) -> bytes:
        """Range read via DBFS' chunked REST. Used by non-FUSE pread.

        Honors the 1 MiB-per-call cap. ``n=-1`` reads to EOF (uses
        :meth:`content_length` to know when to stop, falls back to
        short-page detection).
        """
        try:
            total_size = self.size
        except Exception:
            total_size = 0

        if n < 0:
            target = (total_size - pos) if total_size > 0 else None
        else:
            target = n
            if total_size > 0:
                target = min(target, max(0, total_size - pos))

        if target is not None and target <= 0:
            return b""

        sdk = self._sdk()
        full_path = self.full_path()
        result = bytearray()
        offset = pos
        remaining = target  # may be None

        try:
            while True:
                chunk_size = _DBFS_CHUNK
                if remaining is not None:
                    if remaining <= 0:
                        break
                    chunk_size = min(_DBFS_CHUNK, remaining)
                resp = retry_sdk_call(
                    sdk.dbfs.read,
                    path=full_path, offset=offset, length=chunk_size,
                )
                if not resp.data:
                    break
                decoded = base64.b64decode(resp.data)
                if not decoded:
                    break
                result.extend(decoded)
                offset += len(decoded)
                if remaining is not None:
                    remaining -= len(decoded)
                # Short page = end-of-file.
                if len(decoded) < chunk_size:
                    break
        except NOT_FOUND_ERRORS:
            if default is ...:
                raise FileNotFoundError(self.full_path())
            return default
        except SDK_ERRORS:
            if default is ...:
                raise
            return default

        return bytes(result)

    def _remote_upload(self, payload: BytesIO) -> None:
        """Push the full payload via streaming ``sdk.dbfs.open(write=True)``.

        ``payload`` is the project :class:`BytesIO`. The base
        ``write_bytes`` records its position on entry and seeks it
        back before each retry, so we just stream from the current
        cursor here. The DBFS SDK's chunked writer doesn't expose
        per-chunk retries; the outer :func:`retry_sdk_call` replays
        the whole upload on transport failure.
        """
        sdk = self._sdk()
        full_path = self.full_path()

        size_hint = getattr(payload, "size", None)
        LOGGER.debug("Uploading %r bytes to %s", size_hint, self)

        try:
            with sdk.dbfs.open(
                path=full_path, read=False, write=True, overwrite=True,
            ) as f:
                while True:
                    chunk = payload.read(_DBFS_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
        except NOT_FOUND_ERRORS:
            raise FileNotFoundError(full_path)
        LOGGER.info("Wrote %r bytes to %s", size_hint, self)

    # ==================================================================
    # SDK hooks — stat / ls / mkdir / remove
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        try:
            info = retry_sdk_call(
                self._sdk().dbfs.get_status, self.full_path(),
            )
        except NOT_FOUND_ERRORS:
            # Probe via list — bare prefixes don't have stat entries.
            found = next(self._ls(recursive=False, allow_not_found=True), None)
            if found is None:
                return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)
            return IOStats(
                kind=IOKind.DIRECTORY, size=0,
                mtime=float(found.mtime or 0.0),
            )

        return IOStats(
            kind=IOKind.DIRECTORY if info.is_dir else IOKind.FILE,
            size=int(info.file_size or 0),
            mtime=(
                info.modification_time / 1000.0
                if info.modification_time else 0.0
            ),
        )

    def _ls(self, recursive=False, allow_not_found=True):
        try:
            for info in self._sdk().dbfs.list(self.full_path()):
                api_path = info.path
                url_path = (
                    api_path[len("/dbfs"):]
                    if api_path.startswith("/dbfs")
                    else api_path
                )
                child = DBFSPath(
                    url=URL(scheme="dbfs", host=self.url.host, path=url_path),
                    client=self._client,
                )
                if recursive and info.is_dir:
                    yield from child._ls(
                        recursive=True, allow_not_found=allow_not_found,
                    )
                else:
                    yield child
        except NOT_FOUND_ERRORS:
            if not allow_not_found:
                raise

    def _mkdir(self, parents=True, exist_ok=True):
        try:
            retry_sdk_call(self._sdk().dbfs.mkdirs, self.full_path())
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise

    def _remove_file(self, allow_not_found=True):
        try:
            retry_sdk_call(
                self._sdk().dbfs.delete, self.full_path(), recursive=False,
            )
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        path = self.full_path()
        try:
            retry_sdk_call(self._sdk().dbfs.delete, path, recursive=recursive)
            if not with_root:
                retry_sdk_call(self._sdk().dbfs.mkdirs, path)
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        self._invalidate_stat_cache()