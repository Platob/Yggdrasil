""":class:`WorkspacePath` — ``/Workspace/...`` via the Workspace API.

Workspace objects (notebooks, files, directories) are managed
through ``workspace.workspace.*``. There's no FUSE counterpart;
every read / write is SDK-mediated.

The Workspace API has a single download / upload pair (no range
reads, no positional writes), so the byte primitives map onto a
read-modify-rewrite scheme.
"""

from __future__ import annotations

import io as _stdio
from typing import ClassVar, Iterator

from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


__all__ = ["WorkspacePath"]


class WorkspacePath(DatabricksPath):
    """Path under ``/Workspace/...`` via the Workspace API."""

    scheme: ClassVar[str] = "workspace"
    namespace_prefix: ClassVar[str] = "/Workspace/"

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = (self.url.path or "").lstrip("/")
        return "/Workspace/" + p if p else "/Workspace"

    @property
    def api_path(self) -> str:
        return self.full_path()

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        try:
            info = self._call(
                self.workspace.workspace.get_status, self.api_path,
            )
        except Exception:
            return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

        ot = getattr(info, "object_type", None)
        is_dir = (
            getattr(ot, "name", None) == "DIRECTORY"
            or str(ot).upper().endswith("DIRECTORY")
        )
        size = int(getattr(info, "size", 0) or 0)
        mtime_ms = getattr(info, "modified_at", None) or 0
        return IOStats(
            kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
            size=size,
            mtime=float(mtime_ms) / 1000.0 if mtime_ms else 0.0,
        )

    @property
    def size(self) -> int:
        return int(self._stat().size)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(self, recursive: bool = False) -> Iterator["WorkspacePath"]:
        try:
            entries = list(
                self._call(self.workspace.workspace.list, self.api_path)
            )
        except Exception:
            return
        for info in entries:
            child_path = getattr(info, "path", None)
            if not child_path:
                continue
            # Workspace.list returns paths under ``/Workspace/...``;
            # the URL path is the namespace-stripped suffix.
            url_path = child_path
            if url_path.startswith("/Workspace/"):
                url_path = url_path[len("/Workspace"):]
            elif url_path.startswith("/Workspace"):
                url_path = url_path[len("/Workspace"):] or "/"
            child = type(self)(
                url=URL(scheme=self.scheme, path=url_path),
                workspace=self._workspace,
            )
            yield child
            ot = getattr(info, "object_type", None)
            is_dir = (
                getattr(ot, "name", None) == "DIRECTORY"
                or str(ot).upper().endswith("DIRECTORY")
            )
            if recursive and is_dir:
                yield from child._ls(recursive=True)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        try:
            self._call(self.workspace.workspace.mkdirs, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise

    def _remove_file(self, missing_ok: bool = True) -> None:
        try:
            self._call(
                self.workspace.workspace.delete,
                self.api_path, recursive=False,
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
                self.workspace.workspace.delete,
                self.api_path, recursive=recursive,
            )
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    # ==================================================================
    # Holder I/O — download / upload
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        try:
            response = self._call(
                self.workspace.workspace.download, self.api_path,
            )
        except Exception as exc:
            if _looks_like_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            raise
        body = getattr(response, "contents", None) or response
        try:
            data = body.read()
        except AttributeError:
            data = bytes(body)
        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

    def _write_mv(self, data: memoryview, pos: int) -> int:
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
            existing = (
                bytes(self._read_mv(existing_size, 0)) if existing_size else b""
            )
            if pos > len(existing):
                existing = existing + b"\x00" * (pos - len(existing))
            payload = existing[:pos] + bytes(data) + existing[pos + n:]
        self._upload(payload)
        return n

    def _upload(self, payload: bytes) -> None:
        self._call(
            self.workspace.workspace.upload,
            path=self.api_path,
            content=_stdio.BytesIO(payload),
            overwrite=True,
        )
        self._invalidate_stat_cache()

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        try:
            existing_size = int(self._stat().size)
        except Exception:
            existing_size = 0
        if n == 0:
            self._upload(b"")
            return 0
        if n <= existing_size:
            head = bytes(self._read_mv(n, 0))
        else:
            existing = bytes(self._read_mv(existing_size, 0)) if existing_size else b""
            head = existing + b"\x00" * (n - existing_size)
        self._upload(head)
        return n

    def _clear(self) -> None:
        self._remove_file(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound", "ResourceDoesNotExist", "FileNotFoundError",
    ) or isinstance(exc, FileNotFoundError)


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError")
