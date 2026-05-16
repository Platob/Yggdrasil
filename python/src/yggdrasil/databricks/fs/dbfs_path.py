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
from typing import ClassVar, Iterator

from yggdrasil.data.enums import Scheme
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from ..path import DatabricksPath


__all__ = ["DBFSPath"]
from ...dataclasses import WaitingConfig


logger = logging.getLogger(__name__)

_DBFS_CHUNK = 1 * 1024 * 1024


class DBFSPath(DatabricksPath):
    """Path under ``/dbfs/...`` via the DBFS SDK API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_DBFS
    namespace_prefix: ClassVar[str] = "/dbfs/"

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
            info = self._call(self.client.workspace_client().dbfs.get_status, self.api_path)
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

    def _ls(self, recursive: bool = False) -> Iterator["DBFSPath"]:
        try:
            entries = list(self._call(self.client.workspace_client().dbfs.list, self.api_path))
        except Exception:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dbfs.list %s -> %d entries (recursive=%s)",
                self.api_path, len(entries), recursive,
            )
        for info in entries:
            api_path = getattr(info, "path", None)
            if not api_path:
                continue
            # Listing children skip the ``DatabricksPath._INSTANCES``
            # cache — see ``Singleton.to_singleton`` for the opt-in.
            child = type(self)(
                url=URL(scheme=self.scheme, path=api_path),
                client=self._client,
                singleton_ttl=False,
            )
            # ``dbfs.list`` returns ``is_dir`` / ``file_size`` /
            # ``modification_time`` per entry — seed the child so
            # downstream ``is_file()`` / ``size`` / ``exists()`` don't
            # each fire a follow-up ``dbfs.get_status`` round trip.
            is_dir = bool(getattr(info, "is_dir", False))
            mtime_ms = getattr(info, "modification_time", None) or 0
            child._seed_stat_cache(IOStats(
                kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
                size=0 if is_dir else int(getattr(info, "file_size", 0) or 0),
                mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
            ))
            yield child
            if recursive and is_dir:
                yield from child._ls(recursive=True)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("dbfs.mkdirs %s", self.api_path)
        try:
            self._call(self.client.workspace_client().dbfs.mkdirs, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool = True, wait: WaitingConfig = True) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("dbfs.delete %s (file)", self.api_path)
        try:
            self._call(
                self.client.workspace_client().dbfs.delete, self.api_path, recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True, wait: WaitingConfig = True
    ) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dbfs.delete %s (dir, recursive=%s)", self.api_path, recursive,
            )
        try:
            self._call(
                self.client.workspace_client().dbfs.delete, self.api_path, recursive=recursive,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

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
                self._seed_stat_cache(IOStats(
                    size=offset,
                    kind=IOKind.FILE,
                    media_type=self.media_type,
                ))
            else:
                self._stat_cached.size = offset
                # Re-stamp the TTL — the data we just folded in is
                # fresh, so the entry deserves a full window before
                # the next backend probe.
                self._seed_stat_cache(self._stat_cached)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dbfs.read %s pos=%d n=%s -> %d bytes",
                self.api_path, pos, "EOF" if to_eof else n, len(out),
            )
        return memoryview(bytes(out))

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice via download → in-memory splice → re-upload.

        DBFS has no positional-write API; a positional write has to be
        a read-modify-write at the file granularity. The hot path
        (``pos == 0``) skips the download entirely. For ``pos > 0`` we
        issue a single read-to-EOF (no preceding ``get_status``) and
        let :class:`FileNotFoundError` translate to "no bytes here yet".
        """
        n = len(data)
        if n == 0:
            return 0

        if pos == 0:
            payload = bytes(data)
        else:
            try:
                existing = bytes(self._read_mv(-1, 0))
            except FileNotFoundError:
                existing = b""
            if pos > len(existing):
                existing = existing + b"\x00" * (pos - len(existing))
            payload = existing[:pos] + bytes(data) + existing[pos + n:]

        self._stream_upload(payload)
        return n

    def _stream_upload(self, payload: bytes) -> None:
        """Write *payload* to ``self.api_path`` via the streaming SDK."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "dbfs.upload %s -> %d bytes", self.api_path, len(payload),
            )
        def _do_upload() -> None:
            with self.client.workspace_client().dbfs.open(
                path=self.api_path, read=False, write=True, overwrite=True,
            ) as fh:
                offset = 0
                n = len(payload)
                while offset < n:
                    chunk = payload[offset : offset + _DBFS_CHUNK]
                    fh.write(chunk)
                    offset += len(chunk)
        self._call(_do_upload)
        # The upload just established the object's full size; seed
        # the cache so the next ``size`` / ``exists`` lookup is local
        # and any concurrent reader on the singleton path sees the
        # post-write metadata without a fresh ``dbfs.get_status``.
        self._seed_stat_cache(IOStats(
            size=len(payload),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=self.media_type,
        ))

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")

        if n == 0:
            self._stream_upload(b"")
            return 0

        # Single read-to-EOF — no preceding ``get_status`` probe. A
        # missing object is the natural "nothing to truncate" case;
        # let it surface as zero existing bytes.
        try:
            existing = bytes(self._read_mv(-1, 0))
        except FileNotFoundError:
            existing = b""
        if n <= len(existing):
            head = existing[:n]
        else:
            head = existing + b"\x00" * (n - len(existing))
        self._stream_upload(head)
        return n

    def _clear(self) -> None:
        self._remove_file(missing_ok=True)


# ---------------------------------------------------------------------------
# SDK error duck-typing
# ---------------------------------------------------------------------------


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound", "ResourceDoesNotExist", "FileNotFoundError",
    ) or isinstance(exc, FileNotFoundError)


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError")
