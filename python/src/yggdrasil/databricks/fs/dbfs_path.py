""":class:`DBFSPath` — Databricks DBFS via the ``dbfs.*`` SDK API.

DBFS is the legacy cluster-attached filesystem. Reads chunk via
``dbfs.read`` (1 MiB max per call, base64-encoded payload); writes
stream via ``dbfs.open(write=True)`` which the SDK chunk-uploads
under the hood. The new design folds those calls into the
:class:`Holder` byte primitives so :class:`BytesIO` over a DBFS
path Just Works.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, ClassVar, Iterator

from yggdrasil.data.enums import Scheme
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from ..path import DatabricksPath
from .service import DBFSService

__all__ = ["DBFSPath"]
from ...dataclasses import WaitingConfig

logger = logging.getLogger(__name__)

_DBFS_CHUNK = 1 * 1024 * 1024


class DBFSPath(DatabricksPath):
    """Path under ``/dbfs/...`` via the DBFS SDK API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_DBFS
    NAMESPACE_PREFIX: ClassVar[str] = "/dbfs/"
    _SERVICE_CLASS: ClassVar[type] = DBFSService

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
        del parents
        logger.debug("Creating DBFS directory %r", self)
        try:
            self._call(self.client.workspace_client().dbfs.mkdirs, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        logger.debug("Deleting DBFS file %r", self)
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
            "Deleting DBFS directory %r (recursive=%s)",
            self,
            recursive,
        )
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
        and stops on the first short page. The aggressive ``_bread``
        path leans on this, so a whole-file read costs one
        ``ceil(size / 1 MiB)``-chunk round-trip burst, no preceding
        ``get_status`` probe.
        """
        if n == 0:
            return memoryview(b"")

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
        # The upload just established the object's full size; seed
        # the cache so the next ``size`` / ``exists`` lookup is local
        # and any concurrent reader on the singleton path sees the
        # post-write metadata without a fresh ``dbfs.get_status``.
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
