""":class:`VolumePath` — Databricks Unity Catalog Volume via Files API.

Volumes carry a Unity Catalog hierarchy (catalog → schema → volume →
path) and are the SQL engine's preferred staging surface. Reads /
writes go through ``workspace.files.*``: ``download``, ``upload``,
``list_directory_contents``, ``create_directory``, ``delete``.

The :class:`Holder` byte primitives map onto these:

- :meth:`_read_mv` — ``files.download`` returns a streaming body;
  we slice into the requested range. (Files API doesn't expose
  range reads.)
- :meth:`_write_mv` — read-modify-rewrite via ``files.upload``.
- :meth:`truncate` — ``files.upload`` of the head N bytes.
- :meth:`_clear` — ``files.delete``.

The catalog-management surface (grants, volume metadata, staging
factories) lives in dedicated modules; this class covers the
filesystem contract.
"""

from __future__ import annotations

import io as _stdio
import os
import time
from typing import Any, ClassVar, Iterator, Optional

from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


__all__ = ["VolumePath"]


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/<cat>/<sch>/<vol>/...`` via the Files API."""

    scheme: ClassVar[str] = "volumes"
    namespace_prefix: ClassVar[str] = "/Volumes/"

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = (self.url.path or "").lstrip("/")
        return "/Volumes/" + p if p else "/Volumes"

    @property
    def api_path(self) -> str:
        return self.full_path()

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        files = self.workspace.files
        try:
            info = self._call(files.get_metadata, self.api_path)
        except Exception:
            info = None
        if info is not None:
            return IOStats(
                kind=IOKind.FILE,
                size=int(getattr(info, "content_length", 0) or 0),
                mtime=_mtime(info),
            )
        try:
            dir_info = self._call(files.get_directory_metadata, self.api_path)
        except Exception:
            dir_info = None
        if dir_info is not None:
            return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=0.0)
        return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

    @property
    def _size(self) -> int:
        return int(self._stat().size)

    # ==================================================================
    # SQL staging factory
    # ==================================================================

    @classmethod
    def staging_path(
        cls,
        *,
        catalog_name: str,
        schema_name: str,
        resource_name: Optional[str] = None,
        temporary: bool = True,
        client: Any = None,
        workspace: Any = None,
        max_lifetime: Optional[float] = None,
    ) -> "VolumePath":
        """Mint a fresh Parquet staging file under
        ``/Volumes/<cat>/<sch>/tmp/.sql/<cat>/<sch>/<resource>/part-...``.

        The leaf filename is unique per call (epoch-ms + 8 bytes of
        randomness). Pass ``temporary=False`` to keep the file past
        process exit; otherwise it is unlinked when the holder is
        released.

        Either ``workspace`` (a workspace client) or ``client`` (a
        :class:`DatabricksClient`-shaped aggregator with a
        ``workspace_client()`` method) may be supplied; if both are
        omitted the path lazy-resolves through the active aggregator
        on first use.

        ``max_lifetime`` is accepted for backwards compatibility —
        external sweepers honour it via the ``part-{epoch_ms}-...``
        filename convention.
        """
        del max_lifetime  # filename carries the timestamp; unused here

        cat = _staging_clean_part(catalog_name)
        sch = _staging_clean_part(schema_name)
        tbl = _staging_clean_part(resource_name or "default")

        if workspace is None and client is not None:
            workspace = client.workspace_client()

        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(8).hex()
        leaf = f"part-{epoch_ms}-{seed}.parquet"
        path = f"/{cat}/{sch}/tmp/.sql/{cat}/{sch}/{tbl}/{leaf}"

        return cls(
            url=URL(scheme=cls.scheme, path=path),
            workspace=workspace,
            temporary=temporary,
        )

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(self, recursive: bool = False) -> Iterator["VolumePath"]:
        files = self.workspace.files
        try:
            entries = self._call(files.list_directory_contents, self.api_path)
        except Exception:
            return
        for info in entries:
            child_path = getattr(info, "path", None)
            if not child_path:
                continue
            child = type(self)(
                url=URL.from_(f"volumes://{child_path.lstrip('/Volumes').lstrip('/')}"),
                workspace=self._workspace,
            )
            yield child
            if recursive and getattr(info, "is_directory", False):
                yield from child._ls(recursive=True)

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        try:
            self._call(self.workspace.files.create_directory, self.api_path)
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise

    def _remove_file(self, missing_ok: bool = True) -> None:
        try:
            self._call(self.workspace.files.delete, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True,
    ) -> None:
        try:
            self._call(self.workspace.files.delete_directory, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    # ==================================================================
    # Holder I/O
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        try:
            response = self._call(self.workspace.files.download, self.api_path)
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
            self.workspace.files.upload,
            file_path=self.api_path,
            contents=_stdio.BytesIO(payload),
            overwrite=True,
        )
        self._invalidate_stat_cache()

    def _truncate(self, n: int) -> int:
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


def _mtime(info) -> float:
    val = getattr(info, "last_modified", None) or getattr(info, "modification_time", None)
    if val is None:
        return 0.0
    try:
        return float(val.timestamp())
    except Exception:
        try:
            return float(val) / 1000.0
        except Exception:
            return 0.0


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in (
        "NotFound", "ResourceDoesNotExist", "FileNotFoundError",
    ) or isinstance(exc, FileNotFoundError)


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError")


def _staging_clean_part(value: str) -> str:
    """Strip backticks/whitespace and forbid ``/`` in path segments."""
    return str(value).strip().strip("`").replace("/", "_")
