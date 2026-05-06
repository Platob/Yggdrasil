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
from typing import ClassVar, Iterator

from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


__all__ = ["DBFSPath"]


#: Hard cap on a single ``dbfs.read`` — requesting more returns
#: ``BadRequest``. Drives the chunk loop in :meth:`_read_mv`.
_DBFS_CHUNK = 1 * 1024 * 1024


class DBFSPath(DatabricksPath):
    """Path under ``/dbfs/...`` via the DBFS SDK API."""

    scheme: ClassVar[str] = "dbfs"
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
            info = self._call(self.workspace.dbfs.get_status, self.api_path)
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
            entries = list(self._call(self.workspace.dbfs.list, self.api_path))
        except Exception:
            return
        for info in entries:
            api_path = getattr(info, "path", None)
            if not api_path:
                continue
            child = type(self)(
                url=URL(scheme=self.scheme, path=api_path),
                workspace=self._workspace,
            )
            yield child
            if recursive and getattr(info, "is_dir", False):
                yield from child._ls(recursive=True)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        try:
            self._call(self.workspace.dbfs.mkdirs, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise

    def _remove_file(self, missing_ok: bool = True) -> None:
        try:
            self._call(
                self.workspace.dbfs.delete, self.api_path, recursive=False,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True,
    ) -> None:
        try:
            self._call(
                self.workspace.dbfs.delete, self.api_path, recursive=recursive,
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

        ``Holder.read_mv`` already normalized ``(n, pos)`` into a
        non-negative window. We loop the SDK's 1 MiB cap until the
        window is filled (or the SDK returns short, signalling EOF).
        """
        if n == 0:
            return memoryview(b"")

        out = bytearray()
        offset = pos
        remaining = n
        while remaining > 0:
            chunk_size = min(_DBFS_CHUNK, remaining)
            try:
                resp = self._call(
                    self.workspace.dbfs.read,
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
                break
            decoded = base64.b64decode(data)
            if not decoded:
                break
            out.extend(decoded)
            offset += len(decoded)
            remaining -= len(decoded)
            if len(decoded) < chunk_size:
                # Short page = EOF.
                break
        return memoryview(bytes(out))

    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice via download → in-memory splice → re-upload.

        DBFS has no positional-write API; the only honest answer is
        a read-modify-write at the file granularity. The hot path
        is ``pos == 0`` with whole-file writes, where we skip the
        download.
        """
        n = len(data)
        if n == 0:
            return 0

        if pos == 0:
            payload = bytes(data)
        else:
            try:
                existing_size = int(self._stat().size)
            except Exception:
                existing_size = 0
            if existing_size:
                existing = bytes(self._read_mv(existing_size, 0))
            else:
                existing = b""
            if pos > len(existing):
                existing = existing + b"\x00" * (pos - len(existing))
            payload = existing[:pos] + bytes(data) + existing[pos + n:]

        self._stream_upload(payload)
        self._invalidate_stat_cache()
        return n

    def _stream_upload(self, payload: bytes) -> None:
        """Write *payload* to ``self.api_path`` via the streaming SDK."""
        def _do_upload() -> None:
            with self.workspace.dbfs.open(
                path=self.api_path, read=False, write=True, overwrite=True,
            ) as fh:
                offset = 0
                n = len(payload)
                while offset < n:
                    chunk = payload[offset : offset + _DBFS_CHUNK]
                    fh.write(chunk)
                    offset += len(chunk)
        self._call(_do_upload)

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        try:
            existing_size = int(self._stat().size)
        except Exception:
            existing_size = 0

        if n == 0:
            self._stream_upload(b"")
            self._invalidate_stat_cache()
            return 0
        if n <= existing_size:
            head = bytes(self._read_mv(n, 0))
        else:
            existing = bytes(self._read_mv(existing_size, 0)) if existing_size else b""
            head = existing + b"\x00" * (n - existing_size)
        self._stream_upload(head)
        self._invalidate_stat_cache()
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
