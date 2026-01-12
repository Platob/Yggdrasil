"""databricks.workspaces.path module documentation."""

# src/yggdrasil/databricks/workspaces/databricks_path.py
from __future__ import annotations

import dataclasses
import datetime as dt
import random
import string
import time
from pathlib import PurePosixPath
from typing import Optional, Tuple, Union, TYPE_CHECKING, List, Iterable

import pyarrow as pa
from pyarrow.dataset import FileFormat, ParquetFileFormat, CsvFileFormat, JsonFileFormat
from pyarrow.fs import FileInfo, FileType, FileSystem
import pyarrow.dataset as ds

from .io import DatabricksIO
from .path_kind import DatabricksPathKind
from ...libs.databrickslib import databricks
from ...types import cast_arrow_tabular, cast_polars_dataframe
from ...types.cast.cast_options import CastOptions
from ...types.cast.polars_cast import polars_converter
from ...types.cast.polars_pandas_cast import PolarsDataFrame
from ...types.cast.registry import convert, register_converter

if databricks is not None:
    from databricks.sdk.service.catalog import VolumeType
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


def _flatten_parts(parts: Union[list[str], str]) -> list[str]:
    """
    _flatten_parts documentation.
    
    Args:
        parts: Parameter.
    
    Returns:
        The result.
    """

    if isinstance(parts, str):
        parts = [parts]

    if any("/" in part for part in parts):
        new_parts: list[str] = []

        for part in parts:
            new_parts.extend(_ for _ in part.split("/") if _)

        parts = new_parts

    return parts


def _rand_str(n: int) -> str:
    """
    _rand_str documentation.
    
    Args:
        n: Parameter.
    
    Returns:
        The result.
    """

    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=n))


@dataclasses.dataclass
class DatabricksPath:
    kind: DatabricksPathKind
    parts: List[str]

    _workspace: Optional["Workspace"] = None

    _is_file: Optional[bool] = None
    _is_dir: Optional[bool] = None
    _size: Optional[int] = None
    _mtime: Optional[float] = None

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
        )

    @classmethod
    def parse(
        cls,
        obj: Union["DatabricksPath", str, List[str]],
        workspace: Optional["Workspace"] = None,
    ) -> "DatabricksPath":
        """
        parse documentation.
        
        Args:
            obj: Parameter.
            workspace: Parameter.
        
        Returns:
            The result.
        """

        if not obj:
            return DatabricksPath(kind=DatabricksPathKind.DBFS, parts=[], _workspace=workspace)

        if not isinstance(obj, (str, list)):
            if isinstance(obj, DatabricksPath):
                if workspace is not None and obj._workspace is None:
                    obj._workspace = workspace
                return obj

            from .io import DatabricksIO

            if isinstance(obj, DatabricksIO):
                return obj.path

            if not isinstance(obj, Iterable):
                obj = str(obj)

        obj = _flatten_parts(obj)

        if obj and not obj[0]:
            obj = obj[1:]

        if not obj:
            return DatabricksPath(kind=DatabricksPathKind.DBFS, parts=[], _workspace=workspace)

        head, *tail = obj
        head = head.casefold()

        if head == "dbfs":
            kind = DatabricksPathKind.DBFS
        elif head == "workspace":
            kind = DatabricksPathKind.WORKSPACE
        elif head == "volumes":
            kind = DatabricksPathKind.VOLUME
        else:
            raise ValueError(f"Invalid DatabricksPath head {head!r} from {obj!r}, must be in ['dbfs', 'workspace', 'volumes']")

        return DatabricksPath(kind=kind, parts=tail, _workspace=workspace)

    def __hash__(self):
        """
        __hash__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return hash(self.full_path())

    def __eq__(self, other):
        """
        __eq__ documentation.
        
        Args:
            other: Parameter.
        
        Returns:
            The result.
        """

        if not isinstance(other, DatabricksPath):
            if isinstance(other, str):
                return str(self) == other
            return False
        return self.kind == other.kind and self.parts == other.parts

    def __truediv__(self, other):
        """
        __truediv__ documentation.
        
        Args:
            other: Parameter.
        
        Returns:
            The result.
        """

        if not other:
            return self

        other_parts = _flatten_parts(other)

        return DatabricksPath(
            kind=self.kind,
            parts=self.parts + other_parts,
            _workspace=self._workspace,
        )

    def __enter__(self):
        """
        __enter__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.connect(clone=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        __exit__ documentation.
        
        Args:
            exc_type: Parameter.
            exc_val: Parameter.
            exc_tb: Parameter.
        
        Returns:
            The result.
        """

        if self._workspace is not None:
            self._workspace.__exit__(exc_type, exc_val, exc_tb)

    def __str__(self):
        """
        __str__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.full_path()

    def __repr__(self):
        """
        __repr__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.url()

    def __fspath__(self):
        """
        __fspath__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.full_path()

    def url(self):
        """
        url documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return "dbfs://%s" % self.full_path()

    def full_path(self) -> str:
        """
        full_path documentation.
        
        Args:
            None.
        
        Returns:
            The result.
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
        """
        filesystem documentation.
        
        Args:
            workspace: Parameter.
        
        Returns:
            The result.
        """

        return self.workspace.filesytem(workspace=workspace)

    @property
    def parent(self):
        """
        parent documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.parts:
            return self

        if self._is_file is not None or self._is_dir is not None:
            _is_file, _is_dir = False, True
        else:
            _is_file, _is_dir = None, None

        return DatabricksPath(
            kind=self.kind,
            parts=self.parts[:-1],
            _workspace=self._workspace,
            _is_file=_is_file,
            _is_dir=_is_dir,
        )

    @property
    def workspace(self):
        """
        workspace documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._workspace is None:
            from .workspace import Workspace

            return Workspace()
        return self._workspace

    @workspace.setter
    def workspace(self, value):
        """
        workspace documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

        self._workspace = value

    @property
    def name(self) -> str:
        """
        name documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.parts:
            return ""

        if len(self.parts) == 1:
            return self.parts[-1]

        return self.parts[-1] if self.parts[-1] else self.parts[-2]

    @property
    def extension(self) -> str:
        """
        extension documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        name = self.name
        if "." in name:
            return name.split(".")[-1]
        return ""

    @property
    def file_format(self) -> FileFormat:
        """
        file_format documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        ext = self.extension

        if ext == "parquet":
            return ParquetFileFormat()
        elif ext == "csv":
            return CsvFileFormat()
        elif ext == "json":
            return JsonFileFormat()
        else:
            raise ValueError(
                "Cannot get file format from extension %s" % ext
            )

    @property
    def content_length(self):
        """
        content_length documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._size is None:
            self.refresh_status()
        return self._size

    @content_length.setter
    def content_length(self, value: int):
        """
        content_length documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

        self._size = value

    @property
    def mtime(self) -> Optional[float]:
        """
        mtime documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._mtime is None:
            self.refresh_status()
        return self._mtime

    @mtime.setter
    def mtime(self, value: float):
        """
        mtime documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

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
        """
        file_type documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self.is_file():
            return FileType.File
        elif self.is_dir():
            return FileType.Directory
        else:
            return FileType.NotFound

    @property
    def file_info(self):
        """
        file_info documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return FileInfo(
            path=self.full_path(),
            type=self.file_type,
            mtime=self.mtime,
            size=self.content_length,
        )

    def is_file(self):
        """
        is_file documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._is_file is None:
            self.refresh_status()
        return self._is_file

    def is_dir(self):
        """
        is_dir documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._is_dir is None:
            self.refresh_status()
        return self._is_dir

    def is_dir_sink(self):
        """
        is_dir_sink documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.is_dir() or (self.parts and self.parts[-1] == "")

    @property
    def connected(self) -> bool:
        """
        connected documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self._workspace is not None and self._workspace.connected

    def connect(self, clone: bool = False) -> "DatabricksPath":
        """
        connect documentation.
        
        Args:
            clone: Parameter.
        
        Returns:
            The result.
        """

        workspace = self.workspace.connect(clone=clone)

        if clone:
            return self.clone_instance(
                workspace=workspace
            )

        self._workspace = workspace

        return self

    def close(self):
        """
        close documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        pass

    def volume_parts(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[PurePosixPath]]:
        """
        volume_parts documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self.kind != DatabricksPathKind.VOLUME:
            return None, None, None, None

        catalog = self.parts[0] if len(self.parts) > 0 and self.parts[0] else None
        schema = self.parts[1] if len(self.parts) > 1 and self.parts[1] else None
        volume = self.parts[2] if len(self.parts) > 2 and self.parts[2] else None

        # NOTE: rel is used as a true/false “has relative path” indicator in this file.
        # The runtime value is a list[str] (not PurePosixPath). Keeping it that way to avoid behavior changes.
        return catalog, schema, volume, self.parts[3:]  # type: ignore[return-value]

    def refresh_status(self) -> "DatabricksPath":
        """
        refresh_status documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self.kind == DatabricksPathKind.VOLUME:
            self._refresh_volume_status()
        elif self.kind == DatabricksPathKind.WORKSPACE:
            self._refresh_workspace_status()
        elif self.kind == DatabricksPathKind.DBFS:
            self._refresh_dbfs_status()
        return self

    def _refresh_volume_status(self):
        """
        _refresh_volume_status documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        full_path = self.files_full_path()
        sdk = self.workspace.sdk()

        try:
            info = sdk.files.get_metadata(full_path)

            mtime = (
                dt.datetime.strptime(info.last_modified, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)
                if info.last_modified
                else None
            )

            return self.reset_metadata(is_file=True, is_dir=False, size=info.content_length, mtime=mtime)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

        try:
            info = sdk.files.get_directory_metadata(full_path)
            mtime = (
                dt.datetime.strptime(info.last_modified, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=dt.timezone.utc)
                if info.last_modified
                else None
            )

            return self.reset_metadata(is_file=False, is_dir=True, size=info, mtime=mtime)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

        return self

    def _refresh_workspace_status(self):
        """
        _refresh_workspace_status documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()

        try:
            info = sdk.workspace.get_status(self.workspace_full_path())
            is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
            is_file = not is_dir
            size = info.size
            mtime = float(info.modified_at) / 1000.0 if info.modified_at is not None else None
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            found = next(self.ls(fetch_size=1, recursive=False, allow_not_found=True), None)
            size = 0
            mtime = found.mtime if found is not None else None

            if found is None:
                is_file, is_dir = None, None
            else:
                is_file, is_dir = False, True

        return self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)

    def _refresh_dbfs_status(self):
        """
        _refresh_dbfs_status documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()

        try:
            info = sdk.dbfs.get_status(self.dbfs_full_path())
            is_file, is_dir = not info.is_dir, info.is_dir
            size = info.file_size
            mtime = info.modification_time / 1000.0 if info.modification_time else None
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            found = next(self.ls(fetch_size=1, recursive=False, allow_not_found=True), None)
            size = 0
            mtime = found.mtime if found is not None else None

            if found is None:
                is_file, is_dir = None, None
            else:
                is_file, is_dir = False, True

        return self.reset_metadata(is_file=is_file, is_dir=is_dir, size=size, mtime=mtime)

    def reset_metadata(
        self,
        is_file: Optional[bool] = None,
        is_dir: Optional[bool] = None,
        size: Optional[int] = None,
        mtime: Optional[float] = None,
    ):
        """
        reset_metadata documentation.
        
        Args:
            is_file: Parameter.
            is_dir: Parameter.
            size: Parameter.
            mtime: Parameter.
        
        Returns:
            The result.
        """

        self._is_file = is_file
        self._is_dir = is_dir
        self._size = size
        self._mtime = mtime

        return self

    # ---- API path normalization helpers ----

    def workspace_full_path(self) -> str:
        """
        workspace_full_path documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.parts:
            return "/Workspace"

        parts = self.parts if self.parts[-1] else self.parts[:-1]

        return "/Workspace/%s" % "/".join(parts)

    def dbfs_full_path(self) -> str:
        """
        dbfs_full_path documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.parts:
            return "/dbfs"

        parts = self.parts if self.parts[-1] else self.parts[:-1]

        return "/dbfs/%s" % "/".join(parts)

    def files_full_path(self) -> str:
        """
        files_full_path documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.parts:
            return "/Volumes"

        parts = self.parts if self.parts[-1] else self.parts[:-1]

        return "/Volumes/%s" % "/".join(parts)

    def exists(self, *, follow_symlinks=True) -> bool:
        """
        exists documentation.
        
        Args:
            follow_symlinks: Parameter.
        
        Returns:
            The result.
        """

        return bool(self.is_file() or self.is_dir())

    def mkdir(self, mode=None, parents=True, exist_ok=True):
        """
        mkdir documentation.
        
        Args:
            mode: Parameter.
            parents: Parameter.
            exist_ok: Parameter.
        
        Returns:
            The result.
        """

        try:
            if self.kind == DatabricksPathKind.WORKSPACE:
                self.make_workspace_dir(parents=parents, exist_ok=exist_ok)
            elif self.kind == DatabricksPathKind.VOLUME:
                self.make_volume_dir(parents=parents, exist_ok=exist_ok)
            elif self.kind == DatabricksPathKind.DBFS:
                self.make_dbfs_dir(parents=parents, exist_ok=exist_ok)
        except (NotFound, ResourceDoesNotExist):
            if not parents or self.parent == self:
                raise

            self.parent.mkdir(parents=True, exist_ok=True)
            self.mkdir(parents=False, exist_ok=exist_ok)
        except (AlreadyExists, ResourceAlreadyExists):
            if not exist_ok:
                raise

        return self

    def _ensure_volume(self, exist_ok: bool = True, sdk=None):
        """
        _ensure_volume documentation.
        
        Args:
            exist_ok: Parameter.
            sdk: Parameter.
        
        Returns:
            The result.
        """

        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.workspace.sdk() if sdk is None else sdk

        if catalog_name:
            try:
                sdk.catalogs.create(name=catalog_name)
            except (AlreadyExists, ResourceAlreadyExists, PermissionDenied, BadRequest):
                if not exist_ok:
                    raise

        if schema_name:
            try:
                sdk.schemas.create(catalog_name=catalog_name, name=schema_name)
            except (AlreadyExists, ResourceAlreadyExists, PermissionDenied, BadRequest):
                if not exist_ok:
                    raise

        if volume_name:
            try:
                sdk.volumes.create(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    name=volume_name,
                    volume_type=VolumeType.MANAGED,
                )
            except (AlreadyExists, ResourceAlreadyExists, BadRequest):
                if not exist_ok:
                    raise

    def make_volume_dir(self, parents=True, exist_ok=True):
        """
        make_volume_dir documentation.
        
        Args:
            parents: Parameter.
            exist_ok: Parameter.
        
        Returns:
            The result.
        """

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
        """
        make_workspace_dir documentation.
        
        Args:
            parents: Parameter.
            exist_ok: Parameter.
        
        Returns:
            The result.
        """

        path = self.workspace_full_path()
        sdk = self.workspace.sdk()

        try:
            sdk.workspace.mkdirs(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        return self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def make_dbfs_dir(self, parents=True, exist_ok=True):
        """
        make_dbfs_dir documentation.
        
        Args:
            parents: Parameter.
            exist_ok: Parameter.
        
        Returns:
            The result.
        """

        path = self.dbfs_full_path()
        sdk = self.workspace.sdk()

        try:
            sdk.dbfs.mkdirs(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        return self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=time.time())

    def remove(self, recursive: bool = True):
        """
        remove documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        if self.kind == DatabricksPathKind.VOLUME:
            return self._remove_volume_obj(recursive=recursive)
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self._remove_workspace_obj(recursive=recursive)
        elif self.kind == DatabricksPathKind.DBFS:
            return self._remove_dbfs_obj(recursive=recursive)

    def _remove_volume_obj(self, recursive: bool = True):
        """
        _remove_volume_obj documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        if self.is_file():
            return self._remove_volume_file()
        return self._remove_volume_dir(recursive=recursive)

    def _remove_workspace_obj(self, recursive: bool = True):
        """
        _remove_workspace_obj documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        if self.is_file():
            return self._remove_workspace_file()
        return self._remove_workspace_dir(recursive=recursive)

    def _remove_dbfs_obj(self, recursive: bool = True):
        """
        _remove_dbfs_obj documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        if self.is_file():
            return self._remove_dbfs_file()
        return self._remove_dbfs_dir(recursive=recursive)

    def rmfile(self):
        """
        rmfile documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        try:
            if self.kind == DatabricksPathKind.VOLUME:
                return self._remove_volume_file()
            elif self.kind == DatabricksPathKind.WORKSPACE:
                return self._remove_workspace_file()
            elif self.kind == DatabricksPathKind.DBFS:
                return self._remove_dbfs_file()
        finally:
            self.reset_metadata()
        return self

    def _remove_volume_file(self):
        """
        _remove_volume_file documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        try:
            sdk.files.delete(self.files_full_path())
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass
        return self

    def _remove_workspace_file(self):
        """
        _remove_workspace_file documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        try:
            sdk.workspace.delete(self.workspace_full_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass
        return self

    def _remove_dbfs_file(self):
        """
        _remove_dbfs_file documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        try:
            sdk.dbfs.delete(self.dbfs_full_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass
        return self

    def rmdir(self, recursive: bool = True):
        """
        rmdir documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        if self.kind == DatabricksPathKind.VOLUME:
            return self._remove_volume_dir(recursive=recursive)
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self._remove_workspace_dir(recursive=recursive)
        elif self.kind == DatabricksPathKind.DBFS:
            return self._remove_dbfs_dir(recursive=recursive)

    def _remove_workspace_dir(self, recursive: bool = True):
        """
        _remove_workspace_dir documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        try:
            sdk.workspace.delete(self.workspace_full_path(), recursive=recursive)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass
        self.reset_metadata()
        return self

    def _remove_dbfs_dir(self, recursive: bool = True):
        """
        _remove_dbfs_dir documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        try:
            sdk.dbfs.delete(self.dbfs_full_path(), recursive=recursive)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass
        self.reset_metadata()
        return self

    def _remove_volume_dir(self, recursive: bool = True):
        """
        _remove_volume_dir documentation.
        
        Args:
            recursive: Parameter.
        
        Returns:
            The result.
        """

        root_path = self.files_full_path()
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.workspace.sdk()

        if rel:
            try:
                sdk.files.delete_directory(root_path)
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied) as e:
                message = str(e)
                if recursive and "directory is not empty" in message:
                    for child_path in self.ls():
                        child_path._remove_volume_obj(recursive=True)
                    sdk.files.delete_directory(root_path)
                else:
                    pass
        elif volume_name:
            try:
                sdk.volumes.delete(f"{catalog_name}.{schema_name}.{volume_name}")
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                pass
        elif schema_name:
            try:
                sdk.schemas.delete(f"{catalog_name}.{schema_name}", force=True)
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                pass

        return self.reset_metadata()

    def ls(self, recursive: bool = False, fetch_size: int = None, allow_not_found: bool = True):
        """
        ls documentation.
        
        Args:
            recursive: Parameter.
            fetch_size: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if self.kind == DatabricksPathKind.VOLUME:
            yield from self._ls_volume(recursive=recursive, fetch_size=fetch_size, allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.WORKSPACE:
            yield from self._ls_workspace(recursive=recursive, allow_not_found=allow_not_found)
        elif self.kind == DatabricksPathKind.DBFS:
            yield from self._ls_dbfs(recursive=recursive, allow_not_found=allow_not_found)

    def _ls_volume(self, recursive: bool = False, fetch_size: int = None, allow_not_found: bool = True):
        """
        _ls_volume documentation.
        
        Args:
            recursive: Parameter.
            fetch_size: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

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
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
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
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
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
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
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
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                if not allow_not_found:
                    raise

    def _ls_workspace(self, recursive: bool = True, allow_not_found: bool = True):
        """
        _ls_workspace documentation.
        
        Args:
            recursive: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

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
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            if not allow_not_found:
                raise

    def _ls_dbfs(self, recursive: bool = True, allow_not_found: bool = True):
        """
        _ls_dbfs documentation.
        
        Args:
            recursive: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

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
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            if not allow_not_found:
                raise

    def open(
        self,
        mode="rb",
        encoding=None,
        clone: bool = False,
    ) -> DatabricksIO:
        """
        open documentation.
        
        Args:
            mode: Parameter.
            encoding: Parameter.
            clone: Parameter.
        
        Returns:
            The result.
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
        """
        copy_to documentation.
        
        Args:
            dest: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if self.is_file() and dest.is_file():
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

    # -------------------------
    # Data ops (Arrow / Pandas / Polars)
    # -------------------------
    def arrow_dataset(
        self,
        workspace: Optional["Workspace"] = None,
        filesystem: Optional[FileSystem] = None,
        **kwargs
    ):
        """
        arrow_dataset documentation.
        
        Args:
            workspace: Parameter.
            filesystem: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

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
        """
        read_arrow_table documentation.
        
        Args:
            batch_size: Parameter.
            concat: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        if self.is_file():
            with self.open("rb") as f:
                return f.read_arrow_table(batch_size=batch_size, **kwargs)

        if self.is_dir():
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
            except Exception:
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
        """
        write_arrow documentation.
        
        Args:
            table: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
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
        **kwargs
    ):
        """
        write_arrow_table documentation.
        
        Args:
            table: Parameter.
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        with self.connect(clone=False) as connected:
            if connected.is_dir_sink():
                seed = int(time.time() * 1000)

                for i, batch in enumerate(table.to_batches(max_chunksize=batch_size)):
                    part_path = connected / f"{seed}-{i:05d}-{_rand_str(4)}.parquet"

                    with part_path.open(mode="wb") as f:
                        f.write_arrow_batch(batch, file_format=file_format)

                return connected

            connected.open(mode="wb", clone=False).write_arrow_table(
                table,
                file_format=file_format,
                batch_size=batch_size,
                **kwargs
            )

        return self

    def read_pandas(
        self,
        batch_size: Optional[int] = None,
        concat: bool = True,
        **kwargs
    ):
        """
        read_pandas documentation.
        
        Args:
            batch_size: Parameter.
            concat: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        if concat:
            return self.read_arrow_table(batch_size=batch_size, concat=True, **kwargs).to_pandas()

        tables = self.read_arrow_table(batch_size=batch_size, concat=False, **kwargs)
        return [t.to_pandas() for t in tables]  # type: ignore[arg-type]

    def write_pandas(
        self,
        df,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_pandas documentation.
        
        Args:
            df: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self.write_arrow_table(pa.table(df), batch_size=batch_size, **kwargs)

    def read_polars(
        self,
        batch_size: Optional[int] = None,
        how: str = "diagonal_relaxed",
        rechunk: bool = False,
        concat: bool = True,
        **kwargs
    ):
        """
        read_polars documentation.
        
        Args:
            batch_size: Parameter.
            how: Parameter.
            rechunk: Parameter.
            concat: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        import polars as pl

        if self.is_file():
            with self.open("rb") as f:
                return f.read_polars(batch_size=batch_size, **kwargs)

        if self.is_dir():
            dfs = []
            for child in self.ls(recursive=True):
                if child.is_file():
                    with child.open("rb") as f:
                        dfs.append(f.read_polars(batch_size=batch_size, **kwargs))

            if not dfs:
                return pl.DataFrame()

            if concat:
                return pl.concat(dfs, how=how, rechunk=rechunk)
            return dfs  # type: ignore[return-value]

        raise FileNotFoundError(f"Path does not exist: {self}")

    def write_polars(
        self,
        df,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        Write Polars to a DatabricksPath.

        Behavior:
        - If path is a directory (or ends with a trailing "/"): shard to parquet parts.
          `batch_size` = rows per part (default 1_000_000).
        - If path is a file: write using DatabricksIO.write_polars which is extension-driven
          (parquet/csv/ipc/json/ndjson etc.).

        Notes:
        - If `df` is a LazyFrame, we collect it first (optionally streaming).
        """
        import polars as pl

        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"write_polars expects pl.DataFrame or pl.LazyFrame, got {type(df)!r}")

        with self.connect() as connected:
            if connected.is_dir_sink():
                seed = int(time.time() * 1000)
                rows_per_part = batch_size or 1_000_000

                # Always parquet for directory sinks (lake layout standard)
                for i, chunk in enumerate(df.iter_slices(n_rows=rows_per_part)):
                    part_path = connected / f"part-{i:05d}-{seed}-{_rand_str(4)}.parquet"

                    part_path.write_polars(chunk, **kwargs)

                return connected

            # Single file write: format/extension is handled in DatabricksIO.write_polars
            connected.write_polars(df, **kwargs)

            return connected

    def sql(
        self,
        query: str,
        engine: str = "auto"
    ):
        """
        sql documentation.
        
        Args:
            query: Parameter.
            engine: Parameter.
        
        Returns:
            The result.
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
                    from_table,
                    query
                )
            )

        if engine == "duckdb":
            import duckdb

            __arrow_table__ = self.read_arrow_table()

            return (
                duckdb.connect()
                .execute(query=query.replace(from_table, "__arrow_table__"))
                .fetch_arrow_table()
            )
        elif engine == "polars":
            from polars import CompatLevel

            return (
                self.read_polars()
                .sql(query=query.replace(from_table, "self"))
                .to_arrow(compat_level=CompatLevel.newest())
            )
        else:
            raise ValueError(
                "Invalid engine %s, must be in duckdb, polars" % engine
            )


@register_converter(DatabricksPath, pa.Table)
def databricks_path_to_arrow_table(
    data: DatabricksPath,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    databricks_path_to_arrow_table documentation.
    
    Args:
        data: Parameter.
        options: Parameter.
    
    Returns:
        The result.
    """

    return cast_arrow_tabular(
        data.read_arrow_table(),
        options
    )


@polars_converter(DatabricksPath, PolarsDataFrame)
def databricks_path_to_polars(
    data: DatabricksPath,
    options: Optional[CastOptions] = None,
) -> PolarsDataFrame:
    """
    databricks_path_to_polars documentation.
    
    Args:
        data: Parameter.
        options: Parameter.
    
    Returns:
        The result.
    """

    return cast_polars_dataframe(
        data.read_polars(),
        options
    )
