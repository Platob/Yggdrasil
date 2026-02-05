"""Databricks path abstraction spanning DBFS, workspace, and volumes."""

# src/yggdrasil/databricks/workspaces/databricks_path.py
import dataclasses
import datetime as dt
import random
import string
import time
from pathlib import PurePosixPath
from threading import Thread
from typing import Optional, Tuple, Union, TYPE_CHECKING, List, IO

import pyarrow as pa
from pyarrow import ArrowInvalid
from pyarrow.fs import FileInfo, FileType, FileSystem

from .io import DatabricksIO
from .path_kind import DatabricksPathKind
from .volumes_path import get_volume_status, get_volume_metadata
from ...libs.databrickslib import databricks
from ...types.cast.registry import convert
from ...types.file_format import FileFormat, ExcelFileFormat, ParquetFileFormat, JsonFileFormat, CsvFileFormat

if databricks is not None:
    from databricks.sdk.errors import InternalError
    from databricks.sdk.service.catalog import VolumeType, VolumeInfo, PathOperation
    from databricks.sdk.service.workspace import ObjectType
    from databricks.sdk.errors.platform import (
        NotFound,
        ResourceDoesNotExist,
        BadRequest,
        PermissionDenied,
        AlreadyExists,
        ResourceAlreadyExists,
    )

    NOT_FOUND_ERRORS = NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied
    ALREADY_EXISTS_ERRORS = AlreadyExists, ResourceAlreadyExists, BadRequest

if TYPE_CHECKING:
    from .workspace import Workspace


__all__ = [
    "DatabricksPathKind",
    "DatabricksPath",
]


def _flatten_parts(
    parts: Union["DatabricksPath", List[str], str],
) -> List[str]:
    """Normalize path parts by splitting on '/' and removing empties.

    Args:
        parts: String or list of path parts.

    Returns:
        A flattened list of path components.
    """
    if not isinstance(parts, list):
        if isinstance(parts, DatabricksPath):
            return parts.parts
        elif isinstance(parts, (set, tuple)):
            parts = list(parts)
        else:
            parts = [str(parts).replace("\\", "/")]

    if any("/" in part for part in parts):
        new_parts: list[str] = []

        for part in parts:
            new_parts.extend(_ for _ in part.split("/") if _)

        parts = new_parts

    return parts


def _rand_str(n: int) -> str:
    """Return a random alphanumeric string of length ``n``.

    Args:
        n: Length of the random string.

    Returns:
        Random alphanumeric string.
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=n))


@dataclasses.dataclass
class DatabricksPath:
    """Path wrapper for Databricks workspace, volumes, and DBFS objects."""
    kind: DatabricksPathKind
    parts: List[str]
    temporary: bool = False

    _is_file: Optional[bool] = dataclasses.field(repr=False, hash=False, default=None)
    _is_dir: Optional[bool] = dataclasses.field(repr=False, hash=False, default=None)
    _size: Optional[int] = dataclasses.field(repr=False, hash=False, default=None)
    _mtime: Optional[float] = dataclasses.field(repr=False, hash=False, default=None)

    _workspace: Optional["Workspace"] = dataclasses.field(repr=False, hash=False, default=None)

    _volume_info: Optional["VolumeInfo"] = dataclasses.field(repr=False, hash=False, default=None)

    def clone_instance(
        self,
        *,
        kind: Optional["DatabricksPathKind"] = None,
        parts: Optional[List[str]] = None,
        workspace: Optional["Workspace"] = dataclasses.MISSING,
        is_file: Optional[bool] = dataclasses.MISSING,
        is_dir: Optional[bool] = dataclasses.MISSING,
        size: Optional[int] = dataclasses.MISSING,
        mtime: Optional[float] = dataclasses.MISSING,
        volume_info: Optional["VolumeInfo"] = dataclasses.MISSING,
    ) -> "DatabricksPath":
        """
        Return a copy of this DatabricksPath, optionally overriding fields.
        Uses dataclasses.replace semantics but lets you intentionally override
        cached metadata (or keep it as-is by default).
        """
        return dataclasses.replace(
            self,
            kind=self.kind if kind is None else kind,
            parts=list(self.parts) if parts is None else list(parts),
            _workspace=self._workspace if workspace is dataclasses.MISSING else workspace,
            _is_file=self._is_file if is_file is dataclasses.MISSING else is_file,
            _is_dir=self._is_dir if is_dir is dataclasses.MISSING else is_dir,
            _size=self._size if size is dataclasses.MISSING else size,
            _mtime=self._mtime if mtime is dataclasses.MISSING else mtime,
            _volume_info=self._volume_info if volume_info is dataclasses.MISSING else volume_info,
        )

    @classmethod
    def empty_instance(cls, workspace: Optional["Workspace"] = None):
        return DatabricksPath(
            kind=DatabricksPathKind.DBFS,
            parts=[],
            temporary=False,
            _workspace=workspace,
            _is_file=False,
            _is_dir=False,
            _size=0,
            _mtime=0.0,
            _volume_info=None,
        )

    @classmethod
    def parse(
        cls,
        obj: Union["DatabricksPath", str, List[str]],
        workspace: Optional["Workspace"] = None,
        temporary: bool = False
    ) -> "DatabricksPath":
        """Parse input into a DatabricksPath instance.

        Args:
            obj: Input path, DatabricksPath, or path parts list.
            workspace: Optional Workspace to bind to the path.
            temporary: Temporary location

        Returns:
            A DatabricksPath instance.
        """
        if not obj:
            return cls.empty_instance(workspace=workspace)

        if isinstance(obj, str):
            obj = [obj]
        if not isinstance(obj, (str, list)):
            if isinstance(obj, DatabricksPath):
                if workspace is not None and obj._workspace is None:
                    obj._workspace = workspace
                return obj

            from .io import DatabricksIO

            if isinstance(obj, DatabricksIO):
                return obj.path

            else:
                obj = str(obj)



        obj = _flatten_parts(obj)

        if obj and not obj[0]:
            obj = obj[1:]

        if not obj:
            return cls.empty_instance(workspace=workspace)

        head, *tail = obj

        if head == "dbfs":
            kind = DatabricksPathKind.DBFS
        elif head in {"Workspace", "workspace"}:
            kind = DatabricksPathKind.WORKSPACE
        elif head in {"Volumes", "volumes"}:
            kind = DatabricksPathKind.VOLUME
        else:
            raise ValueError(f"Invalid DatabricksPath head {head!r} from {obj!r}, must be in ['dbfs', 'workspace', 'volumes']")

        return DatabricksPath(
            kind=kind,
            parts=tail,
            temporary=temporary,
            _workspace=workspace,
        )

    def __hash__(self):
        return hash(self.full_path())

    def __eq__(self, other):
        if not isinstance(other, DatabricksPath):
            if isinstance(other, str):
                return str(self) == other
            return False
        return self.kind == other.kind and self.parts == other.parts

    def __truediv__(self, other):
        if not other:
            return self

        other_parts = _flatten_parts(other)

        return DatabricksPath(
            kind=self.kind,
            parts=self.parts + other_parts,
            _workspace=self._workspace,
        )

    def __enter__(self):
        return self.connect(clone=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._workspace is not None:
            self._workspace.__exit__(exc_type, exc_val, exc_tb)

        self.close(wait=False)

    def __str__(self):
        return self.full_path()

    def __repr__(self):
        return self.url()

    def __del__(self):
        self.close(wait=False)

    def __fspath__(self):
        return self.full_path()

    def url(self):
        if self._workspace is not None:
            return self._workspace.safe_host + self.full_path()
        return "dbfs://%s" % self.full_path()

    def full_path(self) -> str:
        """Return the fully qualified path for this namespace.

        Returns:
            The fully qualified path string.
        """
        if self.kind == DatabricksPathKind.DBFS:
            return self.dbfs_full_path()
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self.workspace_full_path()
        elif self.kind == DatabricksPathKind.VOLUME:
            return self.files_full_path()
        else:
            raise ValueError(f"Unknown DatabricksPath kind: {self.kind!r}")

    def filesystem(self, workspace: Optional["Workspace"] = None):
        """Return a PyArrow filesystem adapter for this workspace.

        Args:
            workspace: Optional workspace override.

        Returns:
            A PyArrow FileSystem instance.
        """
        return self.workspace.filesystem(workspace=workspace)

    @property
    def parent(self):
        """Return the parent path.

        Returns:
            A DatabricksPath representing the parent.
        """
        if not self.parts:
            return self

        if self._is_file is not None or self._is_dir is not None:
            _is_file, _is_dir, _size = False, True, 0
        else:
            _is_file, _is_dir, _size = None, None, None

        return DatabricksPath(
            kind=self.kind,
            parts=self.parts[:-1],
            temporary=False,
            _workspace=self._workspace,
            _is_file=_is_file,
            _is_dir=_is_dir,
            _size=_size,
            _volume_info=self._volume_info
        )

    @property
    def workspace(self):
        """Return the associated Workspace instance.

        Returns:
            The Workspace associated with this path.
        """
        if self._workspace is None:
            from .workspace import Workspace

            self._workspace = Workspace()
        return self._workspace

    @workspace.setter
    def workspace(self, value):
        self._workspace = value

    @property
    def name(self) -> str:
        """Return the final path component.

        Returns:
            The final path name component.
        """
        if not self.parts:
            return ""

        if len(self.parts) == 1:
            return self.parts[-1]

        return self.parts[-1] if self.parts[-1] else self.parts[-2]

    @property
    def extension(self) -> str:
        """Return the file extension for the path, if any.

        Returns:
            The file extension without leading dot.
        """
        name = self.name
        if "." in name:
            return name.split(".")[-1]
        return ""

    @property
    def file_format(self) -> FileFormat:
        """Infer the file format from the file extension.

        Returns:
            A PyArrow FileFormat instance.
        """
        ext = self.extension

        if ext == "parquet":
            return ParquetFileFormat()
        elif ext == "csv":
            return CsvFileFormat()
        elif ext == "json":
            return JsonFileFormat()
        elif ext in {"xlsx", "xlsm", "xls"}:
            return ExcelFileFormat()
        else:
            raise ValueError(
                "Cannot get file format from extension %s" % ext
            )

    @property
    def content_length(self) -> int:
        """Return the size of the path in bytes if known.

        Returns:
            The size in bytes.
        """
        if self._size is None:
            self.refresh_status()
        return self._size or 0

    @content_length.setter
    def content_length(self, value: Optional[int]):
        self._size = value

    @property
    def mtime(self) -> Optional[float]:
        """Return the last-modified time for the path.

        Returns:
            Last-modified timestamp in seconds.
        """
        if self._mtime is None:
            self.refresh_status()
        return self._mtime

    @mtime.setter
    def mtime(self, value: float):
        if not isinstance(value, float):
            if isinstance(value, dt.datetime):
                value = value.timestamp()
            elif isinstance(value, str):
                value = dt.datetime.fromisoformat(value).timestamp()
            else:
                value = float(value)
        self._mtime = value

    @property
    def file_type(self):
        if self.is_file():
            return FileType.File
        elif self.is_dir():
            return FileType.Directory
        else:
            return FileType.NotFound

    @property
    def file_info(self):
        return FileInfo(
            path=self.full_path(),
            type=self.file_type,
            mtime=self.mtime,
            size=self.content_length,
        )

    @property
    def is_local(self):
        return False

    def is_file(self):
        """Return True when the path is a file.

        Returns:
            True if the path is a file.
        """
        if self._is_file is None:
            self.refresh_status()
        return self._is_file

    def is_dir(self):
        """Return True when the path is a directory.

        Returns:
            True if the path is a directory.
        """
        if self._is_dir is None:
            self.refresh_status()
        return self._is_dir

    def is_dir_sink(self):
        """Return True if the path represents a directory sink.

        Returns:
            True if the path represents a directory sink.
        """
        if self.is_dir():
            return True

        if self.is_file():
            return False

        if self.parts and self.parts[-1] == "":
            return True

        return not "." in self.name

    @property
    def connected(self) -> bool:
        return self._workspace is not None and self._workspace.connected

    def connect(self, clone: bool = False) -> "DatabricksPath":
        """Connect the path to its workspace, optionally returning a clone.

        Args:
            clone: Whether to return a cloned instance.

        Returns:
            The connected DatabricksPath.
        """
        workspace = self.workspace.connect(clone=clone)

        if clone:
            return self.clone_instance(
                workspace=workspace
            )

        self._workspace = workspace

        return self

    def close(self, wait: bool = True):
        if self.temporary:
            if wait:
                self.remove(recursive=True)
            else:
                Thread(
                    target=self.remove,
                    kwargs={"recursive": True}
                ).start()

    def storage_location(self) -> str:
        info = self.volume_info()

        if info is None:
            raise NotFound(
                "Volume %s not found" % repr(self)
            )

        _, _, _, parts = self.volume_parts()

        base = info.storage_location.rstrip("/")  # avoid trailing slash
        return f"{base}/{'/'.join(parts)}" if parts else base


    def volume_info(self) -> Optional["VolumeInfo"]:
        if self._volume_info is None and self.kind == DatabricksPathKind.VOLUME:
            catalog, schema, volume, _ = self.volume_parts()

            if catalog and schema and volume:
                self._volume_info = get_volume_metadata(
                    sdk=self.workspace.sdk(),
                    full_name="%s.%s.%s" % (catalog, schema, volume)
                )
        return self._volume_info

    def volume_parts(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[PurePosixPath]]:
        """Return (catalog, schema, volume, rel_path) for volume paths.

        Returns:
            Tuple of (catalog, schema, volume, rel_path).
        """
        if self.kind != DatabricksPathKind.VOLUME:
            return None, None, None, None

        catalog = self.parts[0] if len(self.parts) > 0 and self.parts[0] else None
        schema = self.parts[1] if len(self.parts) > 1 and self.parts[1] else None
        volume = self.parts[2] if len(self.parts) > 2 and self.parts[2] else None

        return catalog, schema, volume, self.parts[3:]  # type: ignore[return-value]

    def refresh_status(self) -> "DatabricksPath":
        """Refresh cached metadata for the path.

        Returns:
            The DatabricksPath instance.
        """
        if self.kind == DatabricksPathKind.VOLUME:
            self._refresh_volume_status()
        elif self.kind == DatabricksPathKind.WORKSPACE:
            self._refresh_workspace_status()
        elif self.kind == DatabricksPathKind.DBFS:
            self._refresh_dbfs_status()
        return self

    def _refresh_volume_status(self):
        full_path = self.files_full_path()
        sdk = self.workspace.sdk()

        is_file, is_dir, size, mtime = get_volume_status(
            sdk=sdk,
            full_path=full_path,
            check_file_first="." in self.name,
            raise_error=False
        )

        self.reset_metadata(
            is_file=is_file,
            is_dir=is_dir,
            size=size,
            mtime=mtime,
            volume_info=self._volume_info
        )

        return self

    def _refresh_workspace_status(self):
        sdk = self.workspace.sdk()

        try:
            info = sdk.workspace.get_status(self.workspace_full_path())
            is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
            is_file = not is_dir
            size = info.size
            mtime = float(info.modified_at) / 1000.0 if info.modified_at is not None else None

            return self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            pass

        found = next(self.ls(fetch_size=1, recursive=False, allow_not_found=True), None)
        size = None

        if found is None:
            is_file, is_dir, mtime = None, None, None
        else:
            is_file, is_dir, mtime = False, True, found.mtime

        return self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)

    def _refresh_dbfs_status(self):
        sdk = self.workspace.sdk()

        try:
            info = sdk.dbfs.get_status(self.dbfs_full_path())
            is_file, is_dir = not info.is_dir, info.is_dir
            size = info.file_size
            mtime = info.modification_time / 1000.0 if info.modification_time else None

            return self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

        found = next(self.ls(fetch_size=1, recursive=False, allow_not_found=True), None)
        size = None
        mtime = found.mtime if found is not None else None

        if found is None:
            is_file, is_dir = None, None
        else:
            is_file, is_dir = False, True

        return self.reset_metadata(
            is_file=is_file, is_dir=is_dir, size=size, mtime=mtime
        )

    def reset_metadata(
        self,
        is_file: Optional[bool] = None,
        is_dir: Optional[bool] = None,
        size: Optional[int] = None,
        mtime: Optional[float] = None,
        volume_info: Optional["VolumeInfo"] = None
    ):
        """Update cached metadata fields.

        Args:
            is_file: Optional file flag.
            is_dir: Optional directory flag.
            size: Optional size in bytes.
            mtime: Optional modification time in seconds.
            volume_info: volume metadata

        Returns:
            The DatabricksPath instance.
        """
        self._is_file = is_file
        self._is_dir = is_dir
        self._size = size
        self._mtime = mtime
        self._volume_info = volume_info

        return self

    # ---- API path normalization helpers ----
    def full_parts(self):
        return self.parts if self.parts[-1] else self.parts[:-1]

    def workspace_full_path(self) -> str:
        """Return the full workspace path string.

        Returns:
            Workspace path string.
        """
        return "/Workspace/%s" % "/".join(self.full_parts())

    def dbfs_full_path(self) -> str:
        """Return the full DBFS path string.

        Returns:
            DBFS path string.
        """
        return "/dbfs/%s" % "/".join(self.full_parts())

    def files_full_path(self) -> str:
        """Return the full files (volume) path string.

        Returns:
            Volume path string.
        """
        return "/Volumes/%s" % "/".join(self.full_parts())

    def exists(self, *, follow_symlinks=True) -> bool:
        """Return True if the path exists.

        Args:
            follow_symlinks: Unused; for compatibility.

        Returns:
            True if the path exists.
        """
        if self.is_file():
            return True

        elif self.is_dir():
            return True

        return False

    def mkdir(self, mode=None, parents=True, exist_ok=True):
        """Create a directory for the path.

        Args:
            mode: Optional mode (unused).
            parents: Whether to create parent directories.
            exist_ok: Whether to ignore existing directories.

        Returns:
            The DatabricksPath instance.
        """
        if self.kind == DatabricksPathKind.WORKSPACE:
            self.make_workspace_dir(parents=parents, exist_ok=exist_ok)
        elif self.kind == DatabricksPathKind.VOLUME:
            self.make_volume_dir(parents=parents, exist_ok=exist_ok)
        elif self.kind == DatabricksPathKind.DBFS:
            self.make_dbfs_dir(parents=parents, exist_ok=exist_ok)

        return self

    def _ensure_volume(self, exist_ok: bool = True, sdk=None):
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.workspace.sdk() if sdk is None else sdk
        default_tags = self.workspace.default_tags()

        if catalog_name:
            try:
                sdk.catalogs.create(
                    name=catalog_name,
                    properties=default_tags,
                    comment="Catalog auto generated by yggdrasil"
                )
            except (AlreadyExists, ResourceAlreadyExists, PermissionDenied, BadRequest, InternalError):
                if not exist_ok:
                    raise

        if schema_name:
            try:
                sdk.schemas.create(
                    catalog_name=catalog_name,
                    name=schema_name,
                    properties=default_tags,
                    comment="Schema auto generated by yggdrasil"
                )
            except (AlreadyExists, ResourceAlreadyExists, PermissionDenied, BadRequest, InternalError):
                if not exist_ok:
                    raise

        if volume_name:
            try:
                self._volume_info = sdk.volumes.create(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    name=volume_name,
                    volume_type=VolumeType.MANAGED,
                    comment="Volume auto generated by yggdrasil"
                )
            except (AlreadyExists, ResourceAlreadyExists, BadRequest):
                if not exist_ok:
                    raise

        return self._volume_info

    def make_volume_dir(self, parents=True, exist_ok=True):
        path = self.files_full_path()
        sdk = self.workspace.sdk()

        try:
            sdk.files.create_directory(path)
        except (BadRequest, NotFound, ResourceDoesNotExist) as e:
            if not parents:
                raise

            message = str(e)
            if "not exist" in message:
                self._ensure_volume(sdk=sdk)

            sdk.files.create_directory(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        return self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def make_workspace_dir(self, parents=True, exist_ok=True):
        path = self.workspace_full_path()
        sdk = self.workspace.sdk()

        try:
            sdk.workspace.mkdirs(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        return self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def make_dbfs_dir(self, parents=True, exist_ok=True):
        path = self.dbfs_full_path()
        sdk = self.workspace.sdk()

        try:
            sdk.dbfs.mkdirs(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        return self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def remove(
        self,
        recursive: bool = True,
        allow_not_found: bool = True
    ):
        """Remove the path as a file or directory.

        Args:
            recursive: Whether to delete directories recursively.
            allow_not_found: Allow not found path

        Returns:
            The DatabricksPath instance.
        """
        if self.kind == DatabricksPathKind.VOLUME:
            return self._remove_volume_obj(recursive=recursive, allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self._remove_workspace_obj(recursive=recursive, allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.DBFS:
            return self._remove_dbfs_obj(recursive=recursive, allow_not_found=allow_not_found)

    def _remove_volume_obj(
        self,
        recursive: bool = True,
        allow_not_found: bool = True
    ):
        if self.is_file():
            return self._remove_volume_file(allow_not_found=allow_not_found)
        elif self.is_dir():
            return self._remove_volume_dir(recursive=recursive, allow_not_found=allow_not_found)

    def _remove_workspace_obj(
        self,
        recursive: bool = True,
        allow_not_found: bool = True
    ):
        if self.is_file():
            return self._remove_workspace_file(allow_not_found=allow_not_found)
        elif self.is_dir():
            return self._remove_workspace_dir(recursive=recursive, allow_not_found=allow_not_found)

    def _remove_dbfs_obj(
        self,
        recursive: bool = True,
        allow_not_found: bool = True
    ):
        if self.is_file():
            return self._remove_dbfs_file(allow_not_found=allow_not_found)
        elif self.is_dir():
            return self._remove_dbfs_dir(recursive=recursive, allow_not_found=allow_not_found)

    def rmfile(self, allow_not_found: bool = True):
        """Remove the path as a file.

        Returns:
            The DatabricksPath instance.
        """
        if self.kind == DatabricksPathKind.VOLUME:
            self._remove_volume_file(allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.WORKSPACE:
            self._remove_workspace_file(allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.DBFS:
            self._remove_dbfs_file(allow_not_found=allow_not_found)

        return self

    def _remove_volume_file(self, allow_not_found: bool = True):
        sdk = self.workspace.sdk()
        try:
            sdk.files.delete(self.files_full_path())
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

        return self

    def _remove_workspace_file(self, allow_not_found: bool = True):
        sdk = self.workspace.sdk()
        try:
            sdk.workspace.delete(self.workspace_full_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

        return self

    def _remove_dbfs_file(self, allow_not_found: bool = True):
        sdk = self.workspace.sdk()
        try:
            sdk.dbfs.delete(self.dbfs_full_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

        return self

    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True
    ):
        """Remove the path as a directory.

        Args:
            recursive: Whether to delete directories recursively.
            allow_not_found: Allow not found location
            with_root: Delete also dir object

        Returns:
            The DatabricksPath instance.
        """
        if self.kind == DatabricksPathKind.VOLUME:
            return self._remove_volume_dir(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=with_root
            )
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self._remove_workspace_dir(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=with_root
            )
        elif self.kind == DatabricksPathKind.DBFS:
            return self._remove_dbfs_dir(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=with_root
            )

    def _remove_workspace_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True
    ):
        sdk = self.workspace.sdk()
        full_path =self.workspace_full_path()

        try:
            sdk.workspace.delete(full_path, recursive=recursive)

            if not with_root:
                sdk.workspace.mkdirs(full_path)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

        return self

    def _remove_dbfs_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True
    ):
        sdk = self.workspace.sdk()
        full_path = self.dbfs_full_path()

        try:
            sdk.dbfs.delete(full_path, recursive=recursive)

            if not with_root:
                sdk.dbfs.mkdirs(full_path)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

        return self

    def _remove_volume_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True
    ):
        full_path = self.files_full_path()
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.workspace.sdk()

        if rel:
            try:
                sdk.files.delete_directory(full_path)
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError) as e:
                message = str(e)

                if recursive and "directory is not empty" in message:
                    for child_path in self.ls():
                        child_path._remove_volume_obj(recursive=True)

                    if with_root:
                        sdk.files.delete_directory(full_path)

                elif not allow_not_found:
                    raise
        elif volume_name:
            try:
                sdk.volumes.delete(f"{catalog_name}.{schema_name}.{volume_name}")
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                if not allow_not_found:
                    raise
        elif schema_name:
            try:
                sdk.schemas.delete(f"{catalog_name}.{schema_name}", force=True)
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                if not allow_not_found:
                    raise

        return self.reset_metadata()

    def ls(
        self,
        recursive: bool = False,
        fetch_size: int = None,
        allow_not_found: bool = True
    ):
        """List directory contents for the path.

        Args:
            recursive: Whether to recurse into subdirectories.
            fetch_size: Optional page size for listings.
            allow_not_found: Whether to suppress missing-path errors.

        Yields:
            DatabricksPath entries.
        """
        if self.kind == DatabricksPathKind.VOLUME:
            yield from self._ls_volume(
                recursive=recursive,
                fetch_size=fetch_size,
                allow_not_found=allow_not_found
            )
        elif self.kind == DatabricksPathKind.WORKSPACE:
            yield from self._ls_workspace(
                recursive=recursive,
                allow_not_found=allow_not_found
            )
        elif self.kind == DatabricksPathKind.DBFS:
            yield from self._ls_dbfs(
                recursive=recursive,
                allow_not_found=allow_not_found
            )

    def _ls_volume(self, recursive: bool = False, fetch_size: int = None, allow_not_found: bool = True):
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.workspace.sdk()

        if rel is None:
            if volume_name is None:
                try:
                    for info in sdk.volumes.list(catalog_name=catalog_name, schema_name=schema_name):
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts=[info.catalog_name, info.schema_name, info.name],
                            _workspace=self.workspace,
                            _is_file=False,
                            _is_dir=True,
                            _size=0,
                        )

                        if recursive:
                            yield from base._ls_volume(recursive=recursive)
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                    if not allow_not_found:
                        raise
            elif schema_name is None:
                try:
                    for info in sdk.schemas.list(catalog_name=catalog_name):
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts=[info.catalog_name, info.name],
                            _workspace=self.workspace,
                            _is_file=False,
                            _is_dir=True,
                            _size=0,
                        )
                        if recursive:
                            yield from base._ls_volume(recursive=recursive)
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                    if not allow_not_found:
                        raise
            else:
                try:
                    for info in sdk.catalogs.list():
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts=[info.name],
                            _workspace=self.workspace,
                            _is_file=False,
                            _is_dir=True,
                            _size=0,
                        )
                        if recursive:
                            yield from base._ls_volume(recursive=recursive)
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                    if not allow_not_found:
                        raise
        else:
            full_path = self.files_full_path()

            try:
                for info in sdk.files.list_directory_contents(full_path, page_size=fetch_size):
                    base = DatabricksPath(
                        kind=DatabricksPathKind.VOLUME,
                        parts=info.path.split("/")[2:],
                        _workspace=self.workspace,
                        _is_file=not info.is_directory,
                        _is_dir=info.is_directory,
                        _size=info.file_size,
                    )

                    if recursive and info.is_directory:
                        yield from base._ls_volume(recursive=recursive)
                    else:
                        yield base
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
                if not allow_not_found:
                    raise

    def _ls_workspace(self, recursive: bool = True, allow_not_found: bool = True):
        sdk = self.workspace.sdk()
        full_path = self.workspace_full_path()

        try:
            for info in sdk.workspace.list(full_path, recursive=recursive):
                is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
                yield DatabricksPath(
                    kind=DatabricksPathKind.WORKSPACE,
                    parts=info.path.split("/")[2:],
                    _workspace=self.workspace,
                    _is_file=not is_dir,
                    _is_dir=is_dir,
                    _size=info.size,
                )
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise

    def _ls_dbfs(self, recursive: bool = True, allow_not_found: bool = True):
        sdk = self.workspace.sdk()
        full_path = self.dbfs_full_path()

        try:
            for info in sdk.dbfs.list(full_path, recursive=recursive):
                yield DatabricksPath(
                    kind=DatabricksPathKind.DBFS,
                    parts=info.path.split("/")[2:],
                    _workspace=self.workspace,
                    _is_file=not info.is_dir,
                    _is_dir=info.is_dir,
                    _size=info.file_size,
                )
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied, InternalError):
            if not allow_not_found:
                raise

    def open(
        self,
        mode="rb",
        encoding=None,
        clone: bool = False,
    ) -> DatabricksIO:
        """Open the path as a DatabricksIO instance.

        Args:
            mode: File mode string.
            encoding: Optional text encoding.
            clone: Whether to return a cloned path instance.

        Returns:
            A DatabricksIO instance.
        """
        path = self.connect(clone=clone)

        return (
            DatabricksIO
            .create_instance(path=path, mode=mode, encoding=encoding)
            .connect(clone=False)
        )

    def copy_to(
        self,
        dest: Union["DatabricksIO", "DatabricksPath", str],
        allow_not_found: bool = True,
    ) -> None:
        """Copy this path to another path or IO destination.

        Args:
            dest: Destination IO, DatabricksPath, or path string.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            None.
        """
        if self.is_file():
            with self.open(mode="rb") as src:
                src.copy_to(dest=dest)

        elif self.is_dir():
            dest_base = self.parse(obj=dest, workspace=self.workspace if dest._workspace is None else dest._workspace)
            dest_base.mkdir(parents=True, exist_ok=True)

            skip_base_parts = len(self.parts)

            for src_child in self.ls(recursive=True, allow_not_found=True):
                src_child: DatabricksPath = src_child
                dest_child_parts = dest_base.parts + src_child.parts[skip_base_parts:]

                src_child.copy_to(
                    dest=dest.clone_instance(parts=dest_child_parts),
                    allow_not_found=allow_not_found
                )

        elif not allow_not_found:
            return None

        else:
            raise FileNotFoundError(f"Path {self} does not exist, or dest is not same file or folder type")

    def temporary_credentials(
        self,
        operation: Optional["PathOperation"] = None
    ):
        if self.kind != DatabricksPathKind.VOLUME:
            raise ValueError(f"Cannot generate temporary credentials for {repr(self)}")

        sdk = self.workspace.sdk()
        client = sdk.temporary_path_credentials
        url = self.storage_location()

        return client.generate_temporary_path_credentials(
            url=url,
            operation=operation or PathOperation.PATH_READ,
        )

    def read_bytes(self, use_cache: bool = False):
        with self.open("rb") as f:
            return f.read_all_bytes(use_cache=use_cache)

    def write_bytes(self, data: Union[bytes, IO[bytes]]):
        with self.open("wb") as f:
            f.write_all_bytes(data=data)

    # -------------------------
    # Data ops (Arrow / Pandas / Polars)
    # -------------------------
    def arrow_dataset(
        self,
        workspace: Optional["Workspace"] = None,
        filesystem: Optional[FileSystem] = None,
        **kwargs
    ):
        """Return a PyArrow dataset referencing this path.

        Args:
            workspace: Optional workspace override.
            filesystem: Optional filesystem override.
            **kwargs: Dataset options.

        Returns:
            A PyArrow Dataset instance.
        """
        import pyarrow.dataset as ds

        filesystem = self.filesystem(workspace=workspace) if filesystem is None else filesystem

        return ds.dataset(
            source=self.full_path(),
            filesystem=filesystem,
            **kwargs
        )

    def read_arrow_table(
        self,
        batch_size: Optional[int] = None,
        concat: bool = True,
        **kwargs
    ) -> pa.Table:
        """Read the path into an Arrow table.

        Args:
            batch_size: Optional batch size for reads.
            concat: Whether to concatenate tables for directories.
            **kwargs: Format-specific options.

        Returns:
            An Arrow Table (or list of tables if concat=False).
        """
        if self.is_file():
            with self.open("rb") as f:
                data = f.read_arrow_table(batch_size=batch_size, **kwargs)
            return data

        elif self.is_dir():
            tables: list[pa.Table] = []
            for child in self.ls(recursive=True):
                if child.is_file():
                    with child.open("rb") as f:
                        tables.append(f.read_arrow_table(batch_size=batch_size, **kwargs))

            if not tables:
                return pa.Table.from_batches([], schema=pa.schema([]))

            if not concat:
                # type: ignore[return-value]
                return tables  # caller asked for raw list

            try:
                return pa.concat_tables(tables)
            except ArrowInvalid:
                # Fallback: concat via polars (diagonal relaxed) then back to Arrow
                from polars import CompatLevel

                return self.read_polars(
                    batch_size=batch_size,
                    how="diagonal_relaxed",
                    rechunk=True,
                    concat=True,
                    **kwargs,
                ).to_arrow(compat_level=CompatLevel.newest())

        raise FileNotFoundError(f"Path does not exist: {self}")

    def write_arrow(
        self,
        table: Union[pa.Table, pa.RecordBatch],
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Write Arrow data to the path.

        Args:
            table: Arrow table or record batch to write.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            The DatabricksPath instance.
        """
        if not isinstance(table, pa.Table):
            table = convert(table, pa.Table)

        return self.write_arrow_table(
            table=table,
            batch_size=batch_size,
            **kwargs
        )

    def write_arrow_table(
        self,
        table: pa.Table,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write an Arrow table to the path, sharding if needed.

        Args:
            table: Arrow table to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            The DatabricksPath instance.
        """
        with self.connect(clone=False) as connected:
            if connected.is_dir_sink():
                seed = int(time.time() * 1000)

                for i, batch in enumerate(table.to_batches(max_chunksize=batch_size)):
                    part_path = connected / f"{seed}-{i:05d}-{_rand_str(4)}.parquet"

                    with part_path.open(mode="wb") as f:
                        f.write_arrow_batch(
                            batch,
                            file_format=file_format,
                            batch_size=batch_size,
                        )

                return connected

            else:
                with connected.open(mode="wb", clone=False) as f:
                    f.write_arrow_table(
                        table,
                        file_format=file_format,
                        batch_size=batch_size,
                    )

        return self

    def read_pandas(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        concat: bool = True,
    ):
        """Read the path into a pandas DataFrame.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.
            concat: Whether to concatenate results for directories.

        Returns:
            A pandas DataFrame or list of DataFrames if concat=False.
        """
        if concat:
            return self.read_arrow_table(
                file_format=file_format,
                batch_size=batch_size,
                concat=True,
            ).to_pandas()

        tables = self.read_arrow_table(
            batch_size=batch_size,
            file_format=file_format,
            concat=False,
        )

        return [t.to_pandas() for t in tables]  # type: ignore[arg-type]

    def write_pandas(
        self,
        df: "pandas.DataFrame",
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write a pandas DataFrame to the path.

        Args:
            df: pandas DataFrame to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            The DatabricksPath instance.
        """
        with self.connect(clone=False) as connected:
            if connected.is_dir_sink():
                seed = int(time.time() * 1000)

                def df_batches(pdf, bs: int):
                    for start in range(0, len(pdf), batch_size):
                        yield pdf.iloc[start:start + batch_size]

                for i, batch in enumerate(df_batches(df, batch_size)):
                    part_path = connected / f"{seed}-{i:05d}-{_rand_str(4)}.parquet"

                    with part_path.open(mode="wb", clone=False) as f:
                        f.write_pandas(
                            batch,
                            file_format=file_format,
                            batch_size=batch_size,
                        )
            else:
                with connected.open(mode="wb", clone=False) as f:
                    f.write_pandas(
                        df,
                        file_format=file_format,
                        batch_size=batch_size,
                    )

        return self

    def read_polars(
        self,
        batch_size: Optional[int] = None,
        how: str = "diagonal_relaxed",
        rechunk: bool = False,
        concat: bool = True,
        **kwargs
    ):
        """Read the path into a polars DataFrame.

        Args:
            batch_size: Optional batch size for reads.
            how: Polars concat strategy.
            rechunk: Whether to rechunk after concat.
            concat: Whether to concatenate results for directories.
            **kwargs: Format-specific options.

        Returns:
            A polars DataFrame or list of DataFrames if concat=False.
        """
        import polars

        if self.is_file():
            with self.open("rb") as f:
                df = f.read_polars(batch_size=batch_size, **kwargs)
            return df

        elif self.is_dir():
            dfs = []
            for child in self.ls(recursive=True):
                if child.is_file():
                    with child.open("rb") as f:
                        dfs.append(f.read_polars(batch_size=batch_size, **kwargs))

            if not dfs:
                return polars.DataFrame()

            if concat:
                return polars.concat(dfs, how=how, rechunk=rechunk)
            return dfs  # type: ignore[return-value]

        else:
            raise FileNotFoundError(f"Path does not exist: {self}")

    def write_polars(
        self,
        df,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Write Polars to a DatabricksPath.

        Behavior:
        - If path is a directory (or ends with a trailing "/"): shard to parquet parts.
          `batch_size` = rows per part (default 1_000_000).
        - If path is a file: write using DatabricksIO.write_polars which is extension-driven
          (parquet/csv/ipc/json/ndjson etc.).

        Args:
            df: polars DataFrame or LazyFrame to write.
            file_format: Optional file format override.
            batch_size: Optional rows per part for directory sinks.

        Returns:
            The DatabricksPath instance.

        Notes:
        - If `df` is a LazyFrame, we collect it first (optionally streaming).
        """
        import polars

        if isinstance(df, polars.LazyFrame):
            df = df.collect()

        with self.connect() as connected:
            if connected.is_dir_sink():
                seed = int(time.time() * 1000)
                rows_per_part = batch_size or 1024 * 1024

                # Always parquet for directory sinks (lake layout standard)
                for i, chunk in enumerate(df.iter_slices(n_rows=rows_per_part)):
                    part_path = connected / f"part-{i:05d}-{seed}-{_rand_str(4)}.parquet"

                    with part_path.open(mode="wb", clone=False) as f:
                        f.write_polars(
                            df,
                            file_format=file_format,
                            batch_size=batch_size,
                        )
            else:
                with connected.open(mode="wb", clone=False) as f:
                    f.write_polars(
                        df,
                        file_format=file_format,
                        batch_size=batch_size,
                    )

        return self

    def sql(
        self,
        query: str,
        engine: str = "auto"
    ):
        """Run a local SQL query against data at this path.

        Args:
            query: SQL query string referencing the path.
            engine: Query engine ("duckdb", "polars", or "auto").

        Returns:
            An Arrow Table with the query results.
        """
        if engine == "auto":
            try:
                import duckdb
                engine = "duckdb"
            except ImportError:
                engine = "polars"

        from_table = "dbfs.`%s`" % self.full_path()

        if from_table not in query:
            raise ValueError(
                "SQL query must contain %s to execute query:\n%s" % (
                    repr(from_table),
                    query
                )
            )

        if engine == "duckdb":
            import duckdb

            __arrow_dataset__ = self.arrow_dataset()

            return (
                duckdb.connect()
                .execute(
                    query=query.replace(from_table, "__arrow_dataset__")
                )
                .fetch_arrow_table()
            )
        elif engine == "polars":
            from polars import CompatLevel

            table_name = "__dbpath__"

            return (
                self.read_polars()
                .sql(
                    query=query.replace(from_table, table_name),
                    table_name=table_name
                )
                .to_arrow(compat_level=CompatLevel.newest())
            )
        else:
            raise ValueError(
                "Invalid engine %s, must be in duckdb, polars" % engine
            )
