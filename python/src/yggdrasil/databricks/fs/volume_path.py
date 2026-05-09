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

from yggdrasil.data.enums import Scheme
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


__all__ = ["VolumePath"]


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/<cat>/<sch>/<vol>/...`` via the Files API."""

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_VOLUME
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
    def size(self) -> int:
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
    # Parent / volume auto-creation
    # ==================================================================

    def _split_volume(self) -> Optional[tuple[str, str, str]]:
        """``/cat/sch/vol/...`` → ``("cat", "sch", "vol")`` or ``None``.

        Returns ``None`` when the URL path has fewer than three
        segments (i.e. it doesn't address a volume at all — typically
        a stat probe at ``/Volumes`` itself or a malformed path).
        """
        parts = (self.url.path or "/").lstrip("/").split("/")
        parts = [p for p in parts if p]
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2]

    def _ensure_parents(self) -> bool:
        """Recovery hook for ``_call_ensuring_parents`` after NotFound.

        Cheap-path first: if *self* lives strictly below the volume
        root, try a single ``files.create_directory`` on the parent
        — that's the common case where only a sub-directory was
        missing. Only if that call also fails NotFound (which
        indicates the volume itself doesn't exist) do we fall back
        to :meth:`_ensure_volume` and a parent ``mkdir`` retry. No
        upfront ``catalogs.get`` / ``schemas.get`` / ``volumes.read``
        probes — blind creates swallow ``AlreadyExists`` so the
        idempotent path costs at most three SDK calls.
        """
        triple = self._split_volume()
        if triple is None:
            return False

        parent = self.parent
        pparts = [p for p in (parent.url.path or "/").lstrip("/").split("/") if p]
        has_subdir = len(pparts) > 3  # parent strictly below ``/cat/sch/vol``

        if has_subdir:
            try:
                self._call(
                    self.workspace.files.create_directory, parent.api_path,
                )
                return True
            except Exception as exc:
                if _looks_like_already_exists(exc):
                    return True
                if not _looks_like_not_found(exc):
                    raise
                # Parent missing because volume itself is missing —
                # fall through to volume creation.

        self._ensure_volume()

        if has_subdir:
            try:
                self._call(
                    self.workspace.files.create_directory, parent.api_path,
                )
            except Exception as exc:
                if not _looks_like_already_exists(exc):
                    raise
        return True

    def _ensure_volume(self) -> bool:
        """Blind-create catalog / schema / managed volume.

        Each step swallows ``AlreadyExists`` so re-runs are free.
        Returns ``True`` if at least one ``create`` actually landed
        (so the caller knows something changed).
        """
        triple = self._split_volume()
        if triple is None:
            return False
        catalog, schema, volume = triple
        ws = self.workspace

        created = _safe_create(lambda: ws.catalogs.create(name=catalog))
        created = _safe_create(
            lambda: ws.schemas.create(name=schema, catalog_name=catalog),
        ) or created
        created = _safe_create(
            lambda: ws.volumes.create(
                catalog_name=catalog,
                schema_name=schema,
                name=volume,
                volume_type=_managed_volume_type(),
            ),
        ) or created
        return created

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        try:
            self._call_ensuring_parents(
                self.workspace.files.create_directory, self.api_path,
            )
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
        self._call_ensuring_parents(
            self.workspace.files.upload,
            file_path=self.api_path,
            contents=_stdio.BytesIO(payload),
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
    if name in ("NotFound", "ResourceDoesNotExist", "FileNotFoundError"):
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    return "does not exist" in str(exc).lower()


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError"):
        return True
    return "already exists" in str(exc).lower()


def _safe_create(create: Any) -> bool:
    """Run *create()*; treat ``AlreadyExists`` as success (idempotent)."""
    try:
        create()
    except Exception as exc:
        if _looks_like_already_exists(exc):
            return False
        raise
    return True


def _managed_volume_type() -> Any:
    """Resolve the SDK's ``VolumeType.MANAGED`` enum, falling back to a string.

    The Databricks SDK accepts the enum or the literal ``"MANAGED"``;
    the string fallback keeps the helper usable in test environments
    that mock the workspace client without the SDK installed.
    """
    try:
        from databricks.sdk.service.catalog import VolumeType
        return VolumeType.MANAGED
    except Exception:
        return "MANAGED"


def _staging_clean_part(value: str) -> str:
    """Strip backticks/whitespace and forbid ``/`` in path segments."""
    return str(value).strip().strip("`").replace("/", "_")
