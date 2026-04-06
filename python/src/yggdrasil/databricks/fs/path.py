"""Databricks path abstraction — ``pathlib.Path``-like API.

Hierarchy
---------
::

    DatabricksPath  (abstract base — mirrors ``pathlib.PurePosixPath`` + I/O)
    ├── DBFSPath        — /dbfs/…
    ├── WorkspacePath   — /Workspace/…
    ├── VolumePath      — /Volumes/catalog/schema/volume/…  (Unity Catalog)
    └── TablePath       — /Tables/catalog/schema/table

All namespace-specific logic lives in the concrete subclasses; the base class
provides shared fields, ``parse()`` factory, and the ``pathlib``-compatible API.

The class does **not** inherit from ``AbstractDataPath`` (removed) — it is
self-contained.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import fnmatch
import logging
import stat as stat_mod
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import (
    Any, ClassVar, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING,
)
from urllib.parse import urlparse

from yggdrasil.environ import shutdown as yg_shutdown

from databricks.sdk.errors import InternalError
from databricks.sdk.errors.platform import (
    NotFound,
    ResourceDoesNotExist,
    BadRequest,
    PermissionDenied,
    AlreadyExists,
    ResourceAlreadyExists,
)
from databricks.sdk.service.catalog import VolumeType, VolumeInfo, PathOperation
from databricks.sdk.service.workspace import ObjectType
from pyarrow.fs import FileInfo, FileType

from .io import DatabricksIO
from .path_kind import DatabricksPathKind
from .volumes_path import get_volume_status, get_volume_metadata

if TYPE_CHECKING:
    from ..client import DatabricksClient

logger = logging.getLogger(__name__)

__all__ = [
    "DatabricksPath",
    "DBFSPath",
    "WorkspacePath",
    "VolumePath",
    "TablePath",
    "DatabricksPathKind",
    "DatabricksStatResult",
]

# ---------------------------------------------------------------------------
# Error groups
# ---------------------------------------------------------------------------

NOT_FOUND_ERRORS = (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied)
ALREADY_EXISTS_ERRORS = (AlreadyExists, ResourceAlreadyExists, BadRequest)
SDK_ERRORS = (
    NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied,
    AlreadyExists, ResourceAlreadyExists, InternalError,
)


# ---------------------------------------------------------------------------
# stat_result stand-in
# ---------------------------------------------------------------------------

@dc.dataclass(frozen=True, slots=True)
class DatabricksStatResult:
    """Minimal ``os.stat_result``-compatible object for Databricks paths."""
    st_size: int = 0
    st_mtime: float = 0.0
    st_mode: int = 0

    def __getitem__(self, idx):          # os.stat_result is subscript-able
        return dc.astuple(self)[idx]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _split(path: str) -> List[str]:
    return [p for p in path.split("/") if p]


def _parse_string(s: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """Return ``(namespace, tail_parts, hostname)``."""
    s = s.strip()
    if not s:
        return None, [], None

    if s.startswith(("dbfs://", "http://", "https://")):
        u = urlparse(s)
        host = u.netloc or None
        for parts in (_split(u.fragment), _split(u.path)):
            if parts and parts[0].lower() in _NAMESPACES:
                return parts[0].lower(), parts[1:], host
        return None, _split(u.path), host

    if s.startswith("dbfs:/"):
        return "dbfs", _split(s[len("dbfs:/"):]), None

    parts = _split(s)
    if parts and parts[0].lower() in _NAMESPACES:
        return parts[0].lower(), parts[1:], None

    return None, parts, None


_NAMESPACES = {"workspace", "volumes", "dbfs", "tables"}


def _flatten(parts) -> List[str]:
    if isinstance(parts, DatabricksPath):
        return list(parts.parts)
    if isinstance(parts, (set, tuple)):
        parts = list(parts)
    if not isinstance(parts, list):
        parts = [str(parts).replace("\\", "/")]
    if any("/" in p for p in parts):
        flat: List[str] = []
        for p in parts:
            flat.extend(seg for seg in p.split("/") if seg)
        return flat
    return list(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

@dc.dataclass(eq=False)
class DatabricksPath(ABC):
    """``pathlib.PurePosixPath``-like API for Databricks remote paths.

    Concrete subclasses implement the abstract hooks for each namespace.
    Use :meth:`parse` to build an instance from any string / list / path.
    """

    kind: ClassVar[DatabricksPathKind]

    # ── Shared fields ─────────────────────────────────────────────────
    parts: List[str] = dc.field(default_factory=list)
    temporary: bool = dc.field(default=False)

    # Cached metadata (excluded from eq/hash)
    _is_file: Optional[bool]  = dc.field(repr=False, hash=False, compare=False, default=None)
    _is_dir:  Optional[bool]  = dc.field(repr=False, hash=False, compare=False, default=None)
    _size:    Optional[int]   = dc.field(repr=False, hash=False, compare=False, default=None)
    _mtime:   Optional[float] = dc.field(repr=False, hash=False, compare=False, default=None)
    _client:  Optional["DatabricksClient"] = dc.field(
        repr=False, hash=False, compare=False, default=None,
    )

    # Shutdown-hook handle — populated when temporary=True
    _shutdown_hook: Any = dc.field(
        default=None, init=False, repr=False, hash=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.temporary:
            self._register_shutdown_remove()

    # ================================================================== #
    # Abstract — implement per namespace                                  #
    # ================================================================== #

    @abstractmethod
    def full_path(self) -> str:
        """Absolute path string (e.g. ``/Volumes/cat/schema/vol/file``)."""

    @abstractmethod
    def _refresh_metadata(self) -> None:
        """Fetch remote status and cache in ``_is_file / _is_dir / _size / _mtime``."""

    @abstractmethod
    def _ls_impl(self, recursive: bool, fetch_size: Optional[int],
                 allow_not_found: bool) -> Iterator["DatabricksPath"]: ...

    @abstractmethod
    def _mkdir_impl(self, parents: bool, exist_ok: bool) -> None: ...

    @abstractmethod
    def _remove_file_impl(self, allow_not_found: bool) -> None: ...

    @abstractmethod
    def _remove_dir_impl(self, recursive: bool, allow_not_found: bool,
                         with_root: bool) -> None: ...

    # ================================================================== #
    # Factory                                                             #
    # ================================================================== #

    @classmethod
    def parse(
        cls,
        obj: Union["DatabricksPath", str, List[str]],
        client: Optional["DatabricksClient"] = None,
        temporary: bool = False,
    ) -> "DatabricksPath":
        """Build a concrete path from *obj*."""
        if not obj:
            return _empty(client=client)

        if isinstance(obj, DatabricksPath):
            if client is not None and obj._client is None:
                obj._client = client
            if temporary and not obj.temporary:
                obj.temporary = True
                obj._register_shutdown_remove()
            return obj

        hostname: Optional[str] = None
        if isinstance(obj, str):
            head, tail, hostname = _parse_string(obj)
            if head is None:
                raise ValueError(
                    f"Unrecognized Databricks path {obj!r}.  "
                    "Expected dbfs:/…, /Workspace/…, /Volumes/…, or an https://…/ URL."
                )
            raw = [head, *tail]
        elif isinstance(obj, list):
            raw = _flatten(obj)
        else:
            raise TypeError(f"Cannot parse {type(obj).__name__!r} as a DatabricksPath.")

        flat = [p for p in _flatten(raw) if p]
        if not flat:
            return _empty(client=client)

        ns, *tail = flat
        klass = _KIND_MAP.get(ns.lower())
        if klass is None:
            raise ValueError(f"Unknown namespace {ns!r}.")

        if hostname:
            from ..client import DatabricksClient
            client = DatabricksClient(host=hostname)

        return klass(parts=tail, temporary=temporary, _client=client)

    # ================================================================== #
    # pathlib.PurePosixPath — pure path properties                        #
    # ================================================================== #

    @property
    def name(self) -> str:
        """Last component — ``pathlib.PurePosixPath.name``."""
        return self.parts[-1] if self.parts else ""

    @property
    def suffix(self) -> str:
        """Last extension with dot — ``pathlib.PurePosixPath.suffix``."""
        name = self.name
        i = name.rfind(".")
        return name[i:] if i > 0 else ""

    @property
    def suffixes(self) -> List[str]:
        """All extensions — ``pathlib.PurePosixPath.suffixes``."""
        name = self.name
        if not name or name.startswith("."):
            return []
        parts = name.split(".")
        return [f".{s}" for s in parts[1:]] if len(parts) > 1 else []

    @property
    def stem(self) -> str:
        """Name without the last suffix — ``pathlib.PurePosixPath.stem``."""
        name = self.name
        i = name.rfind(".")
        return name[:i] if i > 0 else name

    @property
    def parent(self) -> "DatabricksPath":
        """Parent path — ``pathlib.PurePosixPath.parent``."""
        if not self.parts:
            return self
        return self._copy_with(
            parts=self.parts[:-1], temporary=False,
            _is_file=False, _is_dir=True, _size=0,
        )

    @property
    def parents(self) -> Tuple["DatabricksPath", ...]:
        """All ancestors — ``pathlib.PurePosixPath.parents``."""
        result = []
        p = self
        while p.parts:
            p = p.parent
            result.append(p)
        return tuple(result)

    def joinpath(self, *others: Union[str, "DatabricksPath"]) -> "DatabricksPath":
        """Join one or more components — ``pathlib.PurePosixPath.joinpath``."""
        result = self
        for o in others:
            result = result / o
        return result

    def with_name(self, name: str) -> "DatabricksPath":
        """Return sibling with *name* — ``pathlib.PurePosixPath.with_name``."""
        if not self.parts:
            raise ValueError("Cannot replace name on an empty path.")
        if not name or "/" in name:
            raise ValueError(f"Invalid name: {name!r}")
        return self._copy_with(parts=self.parts[:-1] + [name])

    def with_suffix(self, suffix: str) -> "DatabricksPath":
        """Return path with *suffix* — ``pathlib.PurePosixPath.with_suffix``."""
        if suffix and not suffix.startswith("."):
            raise ValueError(f"Invalid suffix: {suffix!r}")
        return self.with_name(self.stem + suffix)

    def with_stem(self, stem: str) -> "DatabricksPath":
        """Return path with *stem* — ``pathlib.PurePosixPath.with_stem`` (3.12+)."""
        return self.with_name(stem + self.suffix)

    def match(self, pattern: str) -> bool:
        """Glob-match against the path — ``pathlib.PurePosixPath.match``."""
        return fnmatch.fnmatch(self.name, pattern) or fnmatch.fnmatch(
            str(self), pattern
        )

    def is_relative_to(self, other: Union["DatabricksPath", str]) -> bool:
        """Check prefix relationship — ``pathlib.PurePosixPath.is_relative_to``."""
        if isinstance(other, str):
            other = self.parse(other, client=self._client)
        if type(self) is not type(other):
            return False
        return self.parts[:len(other.parts)] == other.parts

    def relative_to(self, other: Union["DatabricksPath", str]) -> "DatabricksPath":
        """Compute relative path — ``pathlib.PurePosixPath.relative_to``."""
        if isinstance(other, str):
            other = self.parse(other, client=self._client)
        if not self.is_relative_to(other):
            raise ValueError(f"{self} is not relative to {other}")
        return self._copy_with(parts=self.parts[len(other.parts):])

    # ================================================================== #
    # pathlib.Path — concrete I/O operations                              #
    # ================================================================== #

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        """``pathlib.Path.exists``."""
        return bool(self.is_file() or self.is_dir())

    def is_file(self) -> Optional[bool]:
        """``pathlib.Path.is_file``."""
        if self._is_file is None:
            self._refresh_metadata()
        return self._is_file

    def is_dir(self) -> Optional[bool]:
        """``pathlib.Path.is_dir``."""
        if self._is_dir is None:
            self._refresh_metadata()
        return self._is_dir

    def stat(self) -> DatabricksStatResult:
        """``pathlib.Path.stat`` — returns :class:`DatabricksStatResult`."""
        self._refresh_metadata()
        mode = stat_mod.S_IFREG if self._is_file else (stat_mod.S_IFDIR if self._is_dir else 0)
        return DatabricksStatResult(
            st_size=self._size or 0,
            st_mtime=self._mtime or 0.0,
            st_mode=mode | 0o755,
        )

    def iterdir(self) -> Iterator["DatabricksPath"]:
        """``pathlib.Path.iterdir`` — non-recursive listing."""
        yield from self._ls_impl(recursive=False, fetch_size=None, allow_not_found=True)

    def glob(self, pattern: str) -> Iterator["DatabricksPath"]:
        """``pathlib.Path.glob`` — recursive listing filtered by *pattern*."""
        for child in self._ls_impl(recursive=True, fetch_size=None, allow_not_found=True):
            if fnmatch.fnmatch(child.name, pattern):
                yield child

    def rglob(self, pattern: str) -> Iterator["DatabricksPath"]:
        """``pathlib.Path.rglob`` — alias for ``glob`` (always recursive)."""
        yield from self.glob(pattern)

    def mkdir(
        self,
        mode: int = 0o777,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> "DatabricksPath":
        """``pathlib.Path.mkdir``."""
        _ = mode  # Databricks doesn't support POSIX mode bits
        self._mkdir_impl(parents=parents, exist_ok=exist_ok)
        return self

    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> "DatabricksPath":
        """Remove directory (recursive by default)."""
        self._remove_dir_impl(
            recursive=recursive, allow_not_found=allow_not_found, with_root=with_root,
        )
        return self

    def unlink(self, missing_ok: bool = True) -> None:
        """``pathlib.Path.unlink``."""
        self.remove(recursive=True, allow_not_found=missing_ok)

    def remove(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
    ) -> "DatabricksPath":
        self._unregister_shutdown_remove()
        if self.is_file():
            self._remove_file_impl(allow_not_found=allow_not_found)
        elif self.is_dir():
            self._remove_dir_impl(
                recursive=recursive, allow_not_found=allow_not_found, with_root=True,
            )
        return self

    def rmfile(self, allow_not_found: bool = True) -> "DatabricksPath":
        self._remove_file_impl(allow_not_found=allow_not_found)
        return self

    def rename(self, target: Union["DatabricksPath", str]) -> "DatabricksPath":
        """``pathlib.Path.rename`` — implemented as copy + remove."""
        if isinstance(target, str):
            target = self.parse(target, client=self._client)
        self.copy_to(target)
        self.remove(recursive=True)
        return target

    # ── File I/O ──────────────────────────────────────────────────────

    def open(
        self,
        mode: str = "rb",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        clone: bool = False,
    ) -> DatabricksIO:
        """``pathlib.Path.open`` — returns a :class:`DatabricksIO` handle."""
        path = self.connect()
        return (
            DatabricksIO
            .create_instance(path=path, mode=mode, encoding=encoding)
            .connect(clone=False)
        )

    def read_bytes(self, use_cache: bool = False) -> bytes:
        """``pathlib.Path.read_bytes``."""
        with self.open("rb") as f:
            return f.read_all_bytes(use_cache=use_cache)

    def write_bytes(self, data) -> None:
        """``pathlib.Path.write_bytes``."""
        with self.open("wb") as f:
            f.write_all_bytes(data=data)

    def read_text(self, encoding: str = "utf-8", errors: str | None = None) -> str:
        """``pathlib.Path.read_text``."""
        return self.read_bytes().decode(encoding, errors=errors or "strict")

    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        """``pathlib.Path.write_text``."""
        encoded = data.encode(encoding, errors=errors or "strict")
        self.write_bytes(encoded)
        return len(encoded)

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """``pathlib.Path.touch``."""
        if self.exists():
            if not exist_ok:
                raise FileExistsError(f"Path already exists: {self}")
            return
        self.write_bytes(b"")

    # ── Copy ──────────────────────────────────────────────────────────

    def copy_to(
        self,
        dest: Union["DatabricksIO", "DatabricksPath", str],
        allow_not_found: bool = True,
    ) -> None:
        if self.is_file():
            with self.open(mode="rb") as src:
                src.copy_to(dest=dest)
        elif self.is_dir():
            dest_path = self.parse(
                obj=dest,
                client=(
                    self.workspace
                    if isinstance(dest, DatabricksPath) and dest._client is None
                    else None
                ),
            )
            dest_path.mkdir(parents=True, exist_ok=True)
            skip = len(self.parts)
            for child in self.ls(recursive=True, allow_not_found=True):
                child.copy_to(
                    dest=dest_path._copy_with(parts=dest_path.parts + child.parts[skip:]),
                    allow_not_found=allow_not_found,
                )
        elif not allow_not_found:
            raise FileNotFoundError(f"Path {self} does not exist.")

    # ================================================================== #
    # Legacy-compatible helpers                                           #
    # ================================================================== #

    def ls(
        self,
        recursive: bool = False,
        fetch_size: Optional[int] = None,
        allow_not_found: bool = True,
    ) -> Iterator["DatabricksPath"]:
        yield from self._ls_impl(
            recursive=recursive, fetch_size=fetch_size, allow_not_found=allow_not_found,
        )

    def path_parts(self) -> List[str]:
        return list(self.parts)

    @property
    def extension(self) -> str:
        return self.suffix.lstrip(".")

    @property
    def file_type(self) -> FileType:
        if self.is_file():
            return FileType.File
        if self.is_dir():
            return FileType.Directory
        return FileType.NotFound

    # ================================================================== #
    # Metadata                                                            #
    # ================================================================== #

    @property
    def content_length(self) -> int:
        if self._size is None:
            self._refresh_metadata()
        return self._size or 0

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        self._size = value

    @property
    def mtime(self) -> Optional[float]:
        if self._mtime is None:
            self._refresh_metadata()
        return self._mtime

    @mtime.setter
    def mtime(self, value) -> None:
        if isinstance(value, dt.datetime):
            value = value.timestamp()
        elif isinstance(value, str):
            value = dt.datetime.fromisoformat(value).timestamp()
        else:
            value = float(value)
        self._mtime = value

    @property
    def file_info(self) -> FileInfo:
        return FileInfo(
            path=self.full_path(), type=self.file_type,
            mtime=self.mtime, size=self.content_length,
        )

    def refresh_status(self) -> "DatabricksPath":
        self._refresh_metadata()
        return self

    def reset_metadata(
        self,
        is_file: Optional[bool] = None,
        is_dir:  Optional[bool] = None,
        size:    Optional[int]  = None,
        mtime:   Optional[float]= None,
    ) -> "DatabricksPath":
        self._is_file = is_file
        self._is_dir  = is_dir
        self._size    = size
        self._mtime   = mtime
        return self

    # ================================================================== #
    # Client / workspace                                                  #
    # ================================================================== #

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.connected

    @property
    def client(self) -> "DatabricksClient":
        if self._client is None:
            from ..client import DatabricksClient
            self._client = DatabricksClient.current()
        return self._client

    @property
    def workspace(self):
        return self.client.workspace

    @workspace.setter
    def workspace(self, value) -> None:
        self._client = value

    def sql_engine(self):
        catalog, schema, _, _ = self.sql_volume_or_table_parts()
        return self.workspace.sql(catalog_name=catalog, schema_name=schema)

    def connect(self) -> "DatabricksPath":
        self._client = self.workspace.connect()
        return self

    def _register_shutdown_remove(self) -> None:
        """Register a process-exit hook that removes this temporary path."""
        if self._shutdown_hook is not None or not self.temporary:
            return
        try:
            self._shutdown_hook = yg_shutdown.register(self._unsafe_remove)
        except Exception:
            logger.debug(
                "Failed to register shutdown handler for temporary path %s",
                self.full_path(),
                exc_info=True,
            )

    def _unregister_shutdown_remove(self) -> None:
        """Remove the process-exit hook registered by :meth:`_register_shutdown_remove`."""
        hook = self._shutdown_hook
        self._shutdown_hook = None
        if hook is None:
            return
        try:
            try:
                yg_shutdown.unregister(hook)
            except Exception:
                yg_shutdown.unregister(self._unsafe_remove)
        except Exception:
            logger.debug(
                "Failed to unregister shutdown handler for path %s",
                self.full_path(),
                exc_info=True,
            )

    def _unsafe_remove(self) -> None:
        """Best-effort removal used as the atexit / signal shutdown callback."""
        try:
            self.remove(recursive=True, allow_not_found=True)
        except Exception:
            logger.debug(
                "Shutdown cleanup of temporary path %s failed",
                self.full_path(),
                exc_info=True,
            )

    def close(self, wait: bool = True) -> None:
        if self.temporary:
            if wait:
                self.remove(recursive=True)
            else:
                self._unregister_shutdown_remove()
                Thread(target=self.remove, kwargs={"recursive": True}).start()

    # ================================================================== #
    # UC helpers (stubs — overridden by VolumePath / TablePath)           #
    # ================================================================== #

    def sql_volume_or_table_parts(
        self,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        return None, None, None, []

    # ================================================================== #
    # Dunder protocol                                                     #
    # ================================================================== #

    def __hash__(self) -> int:
        return hash(self.full_path())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DatabricksPath):
            return type(self) is type(other) and self.parts == other.parts
        if isinstance(other, str):
            return self.full_path() == other
        return NotImplemented

    def __truediv__(self, other: Union[str, "DatabricksPath"]) -> "DatabricksPath":
        if not other:
            return self
        return self._copy_with(parts=self.parts + _flatten(other))

    def __enter__(self) -> "DatabricksPath":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client is not None:
            self._client.__exit__(exc_type, exc_val, exc_tb)
        self.close(wait=False)

    def __del__(self) -> None:
        self.close(wait=False)

    def __str__(self) -> str:
        return self.full_path()

    def __repr__(self) -> str:
        return self.url()

    def __fspath__(self) -> str:
        return self.full_path()

    # ── Clone ─────────────────────────────────────────────────────────

    def _copy_with(self, **overrides) -> "DatabricksPath":
        return dc.replace(self, **overrides)

    def clone_instance(self, **kwargs) -> "DatabricksPath":
        return dc.replace(
            self,
            parts   =kwargs.get("parts",   self.parts),
            _client =kwargs.get("client",  self._client),
            _is_file=kwargs.get("is_file", self._is_file),
            _is_dir =kwargs.get("is_dir",  self._is_dir),
            _size   =kwargs.get("size",    self._size),
            _mtime  =kwargs.get("mtime",   self._mtime),
        )

    def url(self) -> str:
        base = (
            self._client.base_url.to_string().rstrip("/")
            if self._client is not None
            else "dbfs://"
        )
        return base + self.full_path()

    def _sdk(self):
        return self.workspace.workspace_client()

    def _full_parts(self) -> List[str]:
        if self.parts and not self.parts[-1]:
            return self.parts[:-1]
        return self.parts


# ═══════════════════════════════════════════════════════════════════════════
# Empty sentinel
# ═══════════════════════════════════════════════════════════════════════════

def _empty(client=None) -> "DBFSPath":
    return DBFSPath(
        parts=[], temporary=False, _client=client,
        _is_file=False, _is_dir=False, _size=0, _mtime=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DBFSPath  (/dbfs/…)
# ═══════════════════════════════════════════════════════════════════════════

@dc.dataclass(eq=False)
class DBFSPath(DatabricksPath):
    kind: ClassVar[DatabricksPathKind] = DatabricksPathKind.DBFS

    def full_path(self) -> str:
        return "/dbfs/" + "/".join(self._full_parts())

    def _refresh_metadata(self) -> None:
        try:
            info = self._sdk().dbfs.get_status(self.full_path())
            self.reset_metadata(
                is_file=not info.is_dir, is_dir=info.is_dir,
                size=info.file_size,
                mtime=info.modification_time / 1000.0 if info.modification_time else None,
            )
            return
        except NOT_FOUND_ERRORS:
            pass
        found = next(self._ls_impl(False, 1, True), None)
        self.reset_metadata(
            is_file=None if found is None else False,
            is_dir =None if found is None else True,
            size=None,
            mtime=found.mtime if found else None,
        )

    def _ls_impl(self, recursive=False, fetch_size=None, allow_not_found=True):
        try:
            for info in self._sdk().dbfs.list(self.full_path()):
                child = DBFSPath(
                    parts=info.path.split("/")[2:],
                    _client=self.workspace,
                    _is_file=not info.is_dir, _is_dir=info.is_dir,
                    _size=info.file_size,
                )
                if recursive and info.is_dir:
                    yield from child._ls_impl(recursive=True, allow_not_found=allow_not_found)
                else:
                    yield child
        except NOT_FOUND_ERRORS:
            if not allow_not_found:
                raise

    def _mkdir_impl(self, parents=True, exist_ok=True):
        try:
            self._sdk().dbfs.mkdirs(self.full_path())
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def _remove_file_impl(self, allow_not_found=True):
        try:
            self._sdk().dbfs.delete(self.full_path(), recursive=False)
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

    def _remove_dir_impl(self, recursive=True, allow_not_found=True, with_root=True):
        path = self.full_path()
        try:
            self._sdk().dbfs.delete(path, recursive=recursive)
            if not with_root:
                self._sdk().dbfs.mkdirs(path)
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()


# ═══════════════════════════════════════════════════════════════════════════
# WorkspacePath  (/Workspace/…)
# ═══════════════════════════════════════════════════════════════════════════

@dc.dataclass(eq=False)
class WorkspacePath(DatabricksPath):
    kind: ClassVar[DatabricksPathKind] = DatabricksPathKind.WORKSPACE

    def full_path(self) -> str:
        return "/Workspace/" + "/".join(self._full_parts())

    def _refresh_metadata(self) -> None:
        try:
            info = self._sdk().workspace.get_status(self.full_path())
            is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
            self.reset_metadata(
                is_file=not is_dir, is_dir=is_dir,
                size=info.size,
                mtime=float(info.modified_at) / 1000.0 if info.modified_at else None,
            )
            return
        except SDK_ERRORS:
            pass
        found = next(self._ls_impl(False, 1, True), None)
        self.reset_metadata(
            is_file=None if found is None else False,
            is_dir =None if found is None else True,
            size=None,
            mtime=found.mtime if found else None,
        )

    def _ls_impl(self, recursive=False, fetch_size=None, allow_not_found=True):
        try:
            for info in self._sdk().workspace.list(self.full_path(), recursive=recursive):
                is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
                yield WorkspacePath(
                    parts=info.path.split("/")[2:],
                    _client=self.workspace,
                    _is_file=not is_dir, _is_dir=is_dir,
                    _size=info.size,
                )
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _mkdir_impl(self, parents=True, exist_ok=True):
        try:
            self._sdk().workspace.mkdirs(self.full_path())
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def _remove_file_impl(self, allow_not_found=True):
        try:
            self._sdk().workspace.delete(self.full_path(), recursive=True)
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

    def _remove_dir_impl(self, recursive=True, allow_not_found=True, with_root=True):
        path = self.full_path()
        try:
            self._sdk().workspace.delete(path, recursive=recursive)
            if not with_root:
                self._sdk().workspace.mkdirs(path)
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()


# ═══════════════════════════════════════════════════════════════════════════
# VolumePath  (/Volumes/catalog/schema/volume/…)
# ═══════════════════════════════════════════════════════════════════════════

@dc.dataclass(eq=False)
class VolumePath(DatabricksPath):
    kind: ClassVar[DatabricksPathKind] = DatabricksPathKind.VOLUME

    _volume_info: Optional[VolumeInfo] = dc.field(
        repr=False, hash=False, compare=False, default=None,
    )

    def full_path(self) -> str:
        return "/Volumes/" + "/".join(self._full_parts())

    # ── UC decomposition ──────────────────────────────────────────────

    def sql_volume_or_table_parts(self):
        p = self.parts
        return (
            p[0] if len(p) > 0 else None,
            p[1] if len(p) > 1 else None,
            p[2] if len(p) > 2 else None,
            p[3:],
        )

    def volume_info(self) -> Optional[VolumeInfo]:
        if self._volume_info is None:
            cat, sch, vol, _ = self.sql_volume_or_table_parts()
            if cat and sch and vol:
                self._volume_info = get_volume_metadata(
                    sdk=self._sdk(), full_name=f"{cat}.{sch}.{vol}",
                )
        return self._volume_info

    def storage_location(self) -> str:
        info = self.volume_info()
        if info is None:
            raise NotFound(f"Volume {self!r} not found.")
        _, _, _, rel = self.sql_volume_or_table_parts()
        base = info.storage_location.rstrip("/")
        return f"{base}/{'/'.join(rel)}" if rel else base

    def temporary_credentials(self, operation: Optional[PathOperation] = None):
        return self._sdk().temporary_path_credentials.generate_temporary_path_credentials(
            url=self.storage_location(),
            operation=operation or PathOperation.PATH_READ,
        )

    # ── Metadata ──────────────────────────────────────────────────────

    def _refresh_metadata(self) -> None:
        is_file, is_dir, size, mtime = get_volume_status(
            sdk=self._sdk(),
            full_path=self.full_path(),
            check_file_first="." in self.name,
            raise_error=False,
        )
        self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)

    def reset_metadata(
        self,
        is_file=None, is_dir=None, size=None, mtime=None,
        volume_info=None,
    ):
        super().reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)
        if volume_info is not None:
            self._volume_info = volume_info
        return self

    def clone_instance(self, *, volume_info=dc.MISSING, **kwargs):
        clone = super().clone_instance(**kwargs)
        clone._volume_info = self._volume_info if volume_info is dc.MISSING else volume_info
        return clone

    # ── Listing ───────────────────────────────────────────────────────

    def _ls_impl(self, recursive=False, fetch_size=None, allow_not_found=True):
        cat, sch, vol, _ = self.sql_volume_or_table_parts()
        if not vol:
            yield from self._ls_uc_hierarchy(
                self._sdk(), cat, sch, recursive, allow_not_found,
            )
            return
        try:
            for info in self._sdk().files.list_directory_contents(
                self.full_path(), page_size=fetch_size,
            ):
                child_parts = info.path.split("/")[2:]
                child = VolumePath(
                    parts=child_parts, _client=self.workspace,
                    _is_file=not info.is_directory, _is_dir=info.is_directory,
                    _size=info.file_size,
                )
                if recursive and info.is_directory:
                    yield from child._ls_impl(recursive=True, allow_not_found=allow_not_found)
                else:
                    yield child
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _ls_uc_hierarchy(self, sdk, catalog, schema, recursive, allow_not_found):
        try:
            if not catalog:
                items = [(i.name, [i.name]) for i in sdk.catalogs.list()]
            elif not schema:
                items = [
                    (i.name, [catalog, i.name])
                    for i in sdk.schemas.list(catalog_name=catalog)
                ]
            else:
                items = [
                    (i.name, [i.catalog_name, i.schema_name, i.name])
                    for i in sdk.volumes.list(catalog_name=catalog, schema_name=schema)
                ]
            for _, pts in items:
                child = VolumePath(
                    parts=pts, _client=self.workspace,
                    _is_file=False, _is_dir=True, _size=0,
                )
                if recursive:
                    yield from child._ls_impl(recursive=True, allow_not_found=allow_not_found)
                else:
                    yield child
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    # ── mkdir ─────────────────────────────────────────────────────────

    def _mkdir_impl(self, parents=True, exist_ok=True):
        sdk = self._sdk()
        path = self.full_path()
        try:
            sdk.files.create_directory(path)
        except (BadRequest, NotFound, ResourceDoesNotExist) as exc:
            if not parents:
                raise
            if "not exist" in str(exc):
                self._ensure_uc_hierarchy(sdk=sdk, exist_ok=exist_ok)
            sdk.files.create_directory(path)
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def _ensure_uc_hierarchy(self, sdk=None, exist_ok=True):
        cat, sch, vol, _ = self.sql_volume_or_table_parts()
        sdk = self._sdk() if sdk is None else sdk
        tags = self.workspace.default_tags()
        if cat:
            try:
                sdk.catalogs.create(name=cat, properties=tags, comment="Auto-created by yggdrasil")
            except ALREADY_EXISTS_ERRORS + (PermissionDenied, InternalError):
                if not exist_ok:
                    raise
        if sch:
            try:
                sdk.schemas.create(catalog_name=cat, name=sch, properties=tags, comment="Auto-created by yggdrasil")
            except ALREADY_EXISTS_ERRORS + (PermissionDenied, InternalError):
                if not exist_ok:
                    raise
        if vol:
            try:
                self._volume_info = sdk.volumes.create(
                    catalog_name=cat, schema_name=sch, name=vol,
                    volume_type=VolumeType.MANAGED, comment="Auto-created by yggdrasil",
                )
            except ALREADY_EXISTS_ERRORS:
                if not exist_ok:
                    raise

    # ── Removal ───────────────────────────────────────────────────────

    def _remove_file_impl(self, allow_not_found=True):
        try:
            self._sdk().files.delete(self.full_path())
        except SDK_ERRORS:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

    def _remove_dir_impl(self, recursive=True, allow_not_found=True, with_root=True):
        cat, sch, vol, rel = self.sql_volume_or_table_parts()
        path = self.full_path()
        if rel:
            try:
                self._sdk().files.delete_directory(path)
            except SDK_ERRORS as exc:
                if recursive and "directory is not empty" in str(exc):
                    for child in self.ls():
                        if child.is_file():
                            child._remove_file_impl(True)
                        else:
                            child._remove_dir_impl(True, True, True)
                    if with_root:
                        self._sdk().files.delete_directory(path)
                elif not allow_not_found:
                    raise
        elif vol:
            try:
                self._sdk().volumes.delete(f"{cat}.{sch}.{vol}")
            except SDK_ERRORS:
                if not allow_not_found:
                    raise
        elif sch:
            try:
                self._sdk().schemas.delete(f"{cat}.{sch}", force=True)
            except SDK_ERRORS:
                if not allow_not_found:
                    raise
        self.reset_metadata()


# ═══════════════════════════════════════════════════════════════════════════
# TablePath  (/Tables/catalog/schema/table)
# ═══════════════════════════════════════════════════════════════════════════

@dc.dataclass(eq=False)
class TablePath(DatabricksPath):
    kind: ClassVar[DatabricksPathKind] = DatabricksPathKind.TABLE

    def full_path(self) -> str:
        return "/Tables/" + "/".join(self._full_parts())

    def sql_volume_or_table_parts(self):
        p = self.parts
        return (
            p[0] if len(p) > 0 else None,
            p[1] if len(p) > 1 else None,
            p[2] if len(p) > 2 else None,
            p[3:],
        )

    # Stubs — tables have no file-system representation
    def _refresh_metadata(self):
        self.reset_metadata(is_file=False, is_dir=True)

    def _ls_impl(self, recursive=False, fetch_size=None, allow_not_found=True):
        return iter([])

    def _mkdir_impl(self, parents=True, exist_ok=True):
        pass

    def _remove_file_impl(self, allow_not_found=True):
        pass

    def _remove_dir_impl(self, recursive=True, allow_not_found=True, with_root=True):
        pass


# ---------------------------------------------------------------------------
# Namespace → class dispatch  (must be after all class definitions)
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, type[DatabricksPath]] = {
    "dbfs":      DBFSPath,
    "workspace": WorkspacePath,
    "volumes":   VolumePath,
    "tables":    TablePath,
}

