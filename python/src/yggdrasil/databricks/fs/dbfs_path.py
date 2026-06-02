""":class:`DBFSPath` — Databricks DBFS via the ``dbfs.*`` SDK API.

DBFS is the legacy cluster-attached filesystem. Reads chunk via
``dbfs.read`` (1 MiB max per call, base64-encoded payload); writes
stream via ``dbfs.open(write=True)`` which the SDK chunk-uploads
under the hood. The new design folds those calls into the
:class:`Holder` byte primitives so :class:`IO` over a DBFS
path Just Works.

Cluster-mount fast path
-----------------------

On a Databricks runtime, ``/dbfs/...`` is the FUSE mount that
exposes DBFS as a regular Linux filesystem. Reads, stats, listdirs,
mkdirs, and removes all run at filesystem speed off the kernel; the
``dbfs.*`` REST API (with its 1 MiB chunked reads and base64
encoding) is only used off-cluster. Note ``self.api_path`` strips
the ``/dbfs/`` prefix for the SDK — the kernel mount uses
``self.full_path()`` (which keeps the prefix).
"""

from __future__ import annotations

import base64
import logging
import os
import stat as _stat
import time
from typing import Any, ClassVar, Iterator

from databricks.sdk.errors import InvalidParameterValue

from yggdrasil.dataclasses import ExpiringDict, WaitingConfig
from yggdrasil.enums import Scheme, MediaTypes
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.path.remote_path import _STAT_CACHE_TTL
from yggdrasil.url import URL

from ..path import DatabricksPath
from .service import DBFSService

__all__ = ["DBFSPath"]

logger = logging.getLogger(__name__)

_DBFS_CHUNK = 1 * 1024 * 1024


# ---------------------------------------------------------------------------
# Local /dbfs FUSE mount fast path
# ---------------------------------------------------------------------------

_LOCAL_DBFS_MOUNT_PROBED: bool = False
_LOCAL_DBFS_MOUNT_AVAILABLE: bool = False


def _dbfs_mount_available() -> bool:
    """``True`` when ``/dbfs/...`` is reachable via the kernel mount.

    Conjunction of "this process runs inside a Databricks runtime"
    (``DATABRICKS_RUNTIME_VERSION`` env var) and "``/dbfs`` exists on
    disk". Cached after the first probe; the result is logged once at
    INFO so an operator can confirm the fast path is engaged.
    """
    global _LOCAL_DBFS_MOUNT_PROBED, _LOCAL_DBFS_MOUNT_AVAILABLE
    if _LOCAL_DBFS_MOUNT_PROBED:
        return _LOCAL_DBFS_MOUNT_AVAILABLE
    try:
        from yggdrasil.databricks.client import DatabricksClient
        in_runtime = DatabricksClient.is_in_databricks_environment()
    except Exception:
        in_runtime = False
    has_mount = bool(in_runtime) and os.path.isdir("/dbfs")
    _LOCAL_DBFS_MOUNT_AVAILABLE = has_mount
    _LOCAL_DBFS_MOUNT_PROBED = True
    if has_mount:
        logger.info(
            "DBFSPath: /dbfs kernel mount detected — short-circuiting "
            "stat/read/ls/mkdir/remove/upload off the DBFS REST API.",
        )
    else:
        logger.debug(
            "DBFSPath: /dbfs kernel mount unavailable "
            "(in_runtime=%s, /dbfs exists=%s) — routing through DBFS API.",
            in_runtime, os.path.isdir("/dbfs"),
        )
    return _LOCAL_DBFS_MOUNT_AVAILABLE


def _reset_dbfs_mount_probe() -> None:
    """Test hook — drop the cached probe result."""
    global _LOCAL_DBFS_MOUNT_PROBED, _LOCAL_DBFS_MOUNT_AVAILABLE
    _LOCAL_DBFS_MOUNT_PROBED = False
    _LOCAL_DBFS_MOUNT_AVAILABLE = False


class DBFSPath(DatabricksPath):
    """Path under ``/dbfs/...`` via the DBFS SDK API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_DBFS
    NAMESPACE_PREFIX: ClassVar[str] = "/dbfs/"
    _SERVICE_CLASS: ClassVar[type] = DBFSService

    # Per-class singleton cache — partitioned away from VolumePath /
    # WorkspacePath. No companion lock —
    # :class:`ExpiringDict.get_or_set` is GIL-atomic.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = (self.url.path or "").lstrip("/")
        return "/dbfs/" + p if p else "/dbfs"

    @property
    def api_path(self) -> str:
        """Path as the DBFS SDK expects it — leading slash, no
        ``/dbfs/`` prefix."""
        p = (self.url.path or "").lstrip("/")
        return "/" + p if p else "/"

    # ==================================================================
    # Stat — uncached probe; caching lives on :class:`RemotePath`
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        if _dbfs_mount_available():
            mount_path = self.full_path()
            try:
                st = os.stat(mount_path)
            except FileNotFoundError:
                logger.debug(
                    "stat via /dbfs mount: %r -> MISSING", mount_path,
                )
                return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)
            except OSError as exc:
                logger.debug(
                    "stat via /dbfs mount: %r -> OSError %r, "
                    "falling back to DBFS API", mount_path, exc,
                )
            else:
                if _stat.S_ISDIR(st.st_mode):
                    logger.debug(
                        "stat via /dbfs mount: %r -> DIRECTORY",
                        mount_path,
                    )
                    return IOStats(
                        kind=IOKind.DIRECTORY,
                        size=0,
                        mtime=st.st_mtime,
                    )
                logger.debug(
                    "stat via /dbfs mount: %r -> FILE size=%d",
                    mount_path, st.st_size,
                )
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(st.st_size),
                    mtime=st.st_mtime,
                )
        try:
            info = self._call(
                self.client.workspace_client().dbfs.get_status, self.api_path
            )
        except Exception:
            return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

        kind = IOKind.DIRECTORY if getattr(info, "is_dir", False) else IOKind.FILE
        size = int(getattr(info, "file_size", 0) or 0)
        mtime_ms = getattr(info, "modification_time", None) or 0
        return IOStats(
            kind=kind,
            size=size,
            mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
        )

    @property
    def size(self) -> int:
        return int(self._stat().size)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["DBFSPath"]:
        if _dbfs_mount_available():
            scan_root = self.full_path()
            # Build children from the URL path (the SDK-shape, ``/tmp/x``)
            # not from ``full_path`` — ``full_path`` includes the
            # ``/dbfs/`` prefix and round-tripping that string through
            # the URL parser would treat the leading segment as a host.
            url_root = (self.url.path or "/").rstrip("/") or "/"
            try:
                scan = os.scandir(scan_root)
            except FileNotFoundError:
                logger.debug(
                    "ls via /dbfs mount: %r -> not found", scan_root,
                )
                return
            except (NotADirectoryError, PermissionError) as exc:
                logger.warning(
                    "Cannot scan DBFS directory %r: %r", self, exc,
                )
                return
            yielded = 0
            with scan as it:
                for entry in it:
                    child_url_path = (
                        f"{url_root}/{entry.name}" if url_root != "/"
                        else f"/{entry.name}"
                    )
                    child = type(self)(
                        url=URL(scheme=self.scheme, path=child_url_path),
                        service=self.service,
                        singleton_ttl=singleton_ttl,
                    )
                    try:
                        st = entry.stat(follow_symlinks=False)
                        is_dir = _stat.S_ISDIR(st.st_mode)
                        child._persist_stat_cache(
                            IOStats(
                                kind=(
                                    IOKind.DIRECTORY
                                    if is_dir
                                    else IOKind.FILE
                                ),
                                size=0 if is_dir else int(st.st_size),
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        is_dir = entry.is_dir(follow_symlinks=False)
                    yielded += 1
                    yield child
                    if recursive and is_dir:
                        yield from child._ls(
                            recursive=True,
                            singleton_ttl=singleton_ttl,
                        )
            logger.debug(
                "ls via /dbfs mount: %r -> %d entries (recursive=%s)",
                scan_root, yielded, recursive,
            )
            return
        try:
            entries = list(
                self._call(self.client.workspace_client().dbfs.list, self.api_path)
            )
        except Exception:
            return
        logger.debug(
            "Listing DBFS directory %r -> %d entries (recursive=%s)",
            self,
            len(entries),
            recursive,
        )
        for info in entries:
            api_path = getattr(info, "path", None)
            if not api_path:
                continue
            # ``singleton_ttl`` defaults to ``False`` so listing
            # children stay out of ``DatabricksPath._INSTANCES``;
            # callers wanting cached children pass it through ``ls``.
            child = type(self)(
                url=URL(scheme=self.scheme, path=api_path),
                service=self.service,
                singleton_ttl=singleton_ttl,
            )
            # ``dbfs.list`` returns ``is_dir`` / ``file_size`` /
            # ``modification_time`` per entry — seed the child so
            # downstream ``is_file()`` / ``size`` / ``exists()`` don't
            # each fire a follow-up ``dbfs.get_status`` round trip.
            is_dir = bool(getattr(info, "is_dir", False))
            mtime_ms = getattr(info, "modification_time", None) or 0
            child._persist_stat_cache(
                IOStats(
                    kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
                    size=0 if is_dir else int(getattr(info, "file_size", 0) or 0),
                    mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
                )
            )
            yield child
            if recursive and is_dir:
                yield from child._ls(recursive=True, singleton_ttl=singleton_ttl)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        logger.debug(
            "Creating DBFS directory %r (parents=%s, exist_ok=%s)",
            self, parents, exist_ok,
        )
        if _dbfs_mount_available():
            mount_path = self.full_path()
            try:
                if parents:
                    os.makedirs(mount_path, exist_ok=exist_ok)
                else:
                    os.mkdir(mount_path)
            except FileExistsError:
                if not exist_ok:
                    raise
            self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))
            logger.debug("mkdir via /dbfs mount: %r", mount_path)
            return
        del parents
        try:
            self._call(self.client.workspace_client().dbfs.mkdirs, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        logger.debug(
            "Deleting DBFS file %r (missing_ok=%s)", self, missing_ok,
        )
        if _dbfs_mount_available():
            try:
                os.remove(self.full_path())
                logger.debug("rm via /dbfs mount: %r", self.full_path())
            except FileNotFoundError:
                if not missing_ok:
                    raise
            except IsADirectoryError:
                raise
            self.invalidate_singleton()
            return
        try:
            self._call(
                self.client.workspace_client().dbfs.delete,
                self.api_path,
                recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        logger.info("Deleted DBFS file %r", self)
        self.invalidate_singleton()

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        del wait
        logger.debug(
            "Deleting DBFS directory %r (recursive=%s, missing_ok=%s)",
            self, recursive, missing_ok,
        )
        if _dbfs_mount_available():
            import shutil
            mount_path = self.full_path()
            try:
                if recursive:
                    shutil.rmtree(mount_path)
                else:
                    os.rmdir(mount_path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            self.invalidate_singleton()
            logger.debug(
                "rmdir via /dbfs mount: %r (recursive=%s)",
                mount_path, recursive,
            )
            return
        try:
            self._call(
                self.client.workspace_client().dbfs.delete,
                self.api_path,
                recursive=recursive,
            )
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()

    # ==================================================================
    # Holder I/O — chunked DBFS read; streaming write
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Range read via chunked ``dbfs.read``.

        Loops the SDK's 1 MiB cap until the window is filled, or a
        short page signals EOF. ``n < 0`` means "read to EOF" — the
        loop keeps issuing chunk-sized requests with no upper bound
        and stops on the first short page. The whole-file read fast
        path leans on this, so a read costs one
        ``ceil(size / 1 MiB)``-chunk round-trip burst, no preceding
        ``get_status`` probe.
        """
        if n == 0:
            return memoryview(b"")

        # Cluster fast path — read off the /dbfs FUSE mount, skipping
        # the 1 MiB chunked + base64-encoded REST loop entirely.
        if _dbfs_mount_available():
            mount_path = self.full_path()
            try:
                with open(mount_path, "rb") as fh:
                    if pos:
                        fh.seek(pos)
                    data = fh.read() if n < 0 else fh.read(n)
            except FileNotFoundError as exc:
                logger.debug(
                    "read via /dbfs mount: %r -> NOT FOUND", mount_path,
                )
                raise FileNotFoundError(self.full_path()) from exc
            except OSError as exc:
                logger.debug(
                    "read via /dbfs mount: %r -> OSError %r, "
                    "falling back to DBFS API", mount_path, exc,
                )
            else:
                logger.debug(
                    "read via /dbfs mount: %r -> %d bytes "
                    "(pos=%d, n=%s)",
                    mount_path, len(data), pos, "EOF" if n < 0 else n,
                )
                if not self._stat_cached:
                    try:
                        st = os.stat(mount_path)
                        self._persist_stat_cache(
                            IOStats(
                                size=int(st.st_size),
                                kind=IOKind.FILE,
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        pass
                return memoryview(data)

        out = bytearray()
        offset = pos
        to_eof = n < 0
        hit_eof = False
        while True:
            if to_eof:
                chunk_size = _DBFS_CHUNK
            else:
                remaining = n - len(out)
                if remaining <= 0:
                    break
                chunk_size = min(_DBFS_CHUNK, remaining)
            try:
                resp = self._call(
                    self.client.workspace_client().dbfs.read,
                    path=self.api_path,
                    offset=offset,
                    length=chunk_size,
                )
            except Exception as exc:
                if isinstance(exc, InvalidParameterValue):
                    msg = str(exc)
                    if "Found directory on path: " in msg:
                        # Stamp via the helper so the entry is marked
                        # fresh (``_stat_cached_at``) — a raw assignment
                        # leaves it cold and ``_stat_cached_fresh`` would
                        # ignore it under TTL gating.
                        self._persist_stat_cache(IOStats(
                            kind=IOKind.DIRECTORY,
                            media_type=MediaTypes.DIRECTORY,
                        ))

                if _looks_like_not_found(exc):
                    raise FileNotFoundError(self.full_path()) from exc

                raise
            data = getattr(resp, "data", None)
            if not data:
                hit_eof = True
                break
            decoded = base64.b64decode(data)
            if not decoded:
                hit_eof = True
                break
            out.extend(decoded)
            offset += len(decoded)
            if len(decoded) < chunk_size:
                # Short page = EOF.
                hit_eof = True
                break
        # When we started at pos 0 and rode the read past EOF, the
        # final offset IS the file size — seed the stat cache so the
        # next ``size`` / ``exists`` / ``is_file`` lookup is local.
        if pos == 0 and hit_eof:
            if self._stat_cached is None:
                self._persist_stat_cache(
                    IOStats(
                        size=offset,
                        kind=IOKind.FILE,
                        media_type=self.media_type,
                    )
                )
            else:
                self._stat_cached.size = offset
                # Re-stamp the TTL — the data we just folded in is
                # fresh, so the entry deserves a full window before
                # the next backend probe.
                self._persist_stat_cache(self._stat_cached)
        logger.debug(
            "Read DBFS file %r pos=%d n=%s -> %d bytes",
            self,
            pos,
            "EOF" if to_eof else n,
            len(out),
        )
        return memoryview(bytes(out))

    def _write_stream(
        self,
        src: Any,
        *,
        offset: int,
        size: int = -1,
        **kwargs: Any,
    ) -> int:
        """Override the base chunked stream — one ``dbfs.open`` session.

        :meth:`_stream_upload` already pipes the live
        :class:`IO[bytes]` through a single
        ``dbfs.open(write=True)`` handle with ``_DBFS_CHUNK``-sized
        writes, so the base :meth:`Holder._write_stream` (which
        opens a new DBFS session per chunk) is a strict loss
        here. ``size>=0`` (capped read) or non-zero ``offset``
        fall back to the chunked base path because DBFS can't
        splice at a range. ``batch_size`` only matters for the
        fallback path; ``_stream_upload`` uses ``_DBFS_CHUNK``.
        """
        if offset != 0 or size >= 0:
            return super()._write_stream(src, offset=offset, size=size, **kwargs)
        return self._upload(src)

    def _upload(self, content: Any) -> int:
        """Write *content* to ``self.api_path`` via the streaming SDK.

        Accepts either a bytes-like payload or a seekable binary
        stream. The stream is rewound to origin on every attempt so
        ``retry_sdk_call`` retries re-stream the full body. Bytes
        inputs are sliced into ``_DBFS_CHUNK`` chunks; stream
        inputs are pulled lazily with ``read(_DBFS_CHUNK)``, so a
        large source never lands as a single :class:`bytes` buffer
        in Python.

        Returns the byte count when known (bytes-like input) or
        ``-1`` when the input is a stream of unknown length.
        """
        size = len(content) if hasattr(content, "__len__") else -1
        logger.debug(
            "Uploading DBFS file %r (%s bytes)",
            self,
            size if size >= 0 else "?",
        )
        # Cluster fast path — write straight to the /dbfs kernel mount.
        if _dbfs_mount_available():
            mount_path = self.full_path()
            parent = os.path.dirname(mount_path)
            if parent and not os.path.isdir(parent):
                logger.debug(
                    "upload via /dbfs mount: auto-creating parent %r",
                    parent,
                )
                os.makedirs(parent, exist_ok=True)
            if hasattr(content, "seek"):
                stream = content
                try:
                    stream.seek(0)
                except Exception:
                    pass
                bytes_written = 0
                with open(mount_path, "wb") as fh:
                    while True:
                        chunk = stream.read(_DBFS_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        bytes_written += len(chunk)
                if size == -1:
                    size = bytes_written
            else:
                payload = bytes(content)
                size = len(payload)
                with open(mount_path, "wb") as fh:
                    fh.write(payload)
            logger.debug(
                "upload via /dbfs mount: %r -> %d bytes",
                mount_path, size,
            )
            self._persist_stat_cache(
                IOStats(
                    kind=IOKind.FILE,
                    size=int(max(size, 0)),
                    mtime=time.time(),
                )
            )
            return int(max(size, -1))
        api_path = self.api_path
        dbfs = self.client.workspace_client().dbfs

        if hasattr(content, "seek"):
            stream = content

            def _do_upload() -> None:
                stream.seek(0)
                with dbfs.open(
                    path=api_path,
                    read=False,
                    write=True,
                    overwrite=True,
                ) as fh:
                    while True:
                        chunk = stream.read(_DBFS_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)

        else:
            payload = content

            def _do_upload() -> None:
                with dbfs.open(
                    path=api_path,
                    read=False,
                    write=True,
                    overwrite=True,
                ) as fh:
                    offset = 0
                    n = len(payload)
                    while offset < n:
                        fh.write(payload[offset : offset + _DBFS_CHUNK])
                        offset += _DBFS_CHUNK

        self._call(_do_upload)
        if size >= 0:
            self._persist_stat_cache(
                IOStats(
                    size=size,
                    kind=IOKind.FILE,
                    mtime=time.time(),
                    media_type=self.media_type,
                )
            )
            logger.info("Uploaded DBFS file %r (size=%d)", self, size)
        else:
            logger.info("Uploaded DBFS file %r (size=stream)", self)
        return size

    def _clear(self) -> None:
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))


# ---------------------------------------------------------------------------
# SDK error duck-typing
# ---------------------------------------------------------------------------


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound",
        "ResourceDoesNotExist",
        "FileNotFoundError",
    ) or isinstance(exc, FileNotFoundError)


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError")
