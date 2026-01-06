# src/yggdrasil/databricks/workspaces/databricks_path.py
from __future__ import annotations

import dataclasses
import io
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import PurePosixPath
from typing import BinaryIO, Iterator, Optional, Tuple, Union, TYPE_CHECKING, List

from databricks.sdk.service.catalog import VolumeType

from ...libs.databrickslib import databricks

if databricks is not None:
    from databricks.sdk.service.workspace import ImportFormat, ObjectType
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
    if isinstance(parts, str):
        parts = [parts]

    if any("/" in part for part in parts):
        # flatten parts with slashes
        new_parts = []
        for part in parts:
            split_parts = part.split("/")
            new_parts.extend(split_parts)
        parts = new_parts

    return parts


class DatabricksPathKind(str, Enum):
    WORKSPACE = "workspace"
    VOLUME = "volume"
    DBFS = "dbfs"


@dataclasses.dataclass
class DatabricksPath:
    kind: "DatabricksPathKind"
    parts: List[str]
    workspace: Optional["Workspace"] = None

    _is_file: Optional[bool] = None
    _is_dir: Optional[bool] = None

    _raw_status: Optional[dict] = None
    _raw_status_refresh_time: float = 0.0

    @classmethod
    def parse(
        cls,
        parts: Union[List[str], str],
        workspace: Optional["Workspace"] = None,
    ) -> "DatabricksPath":
        if not parts:
            return DatabricksPath(
                kind=DatabricksPathKind.DBFS,
                parts=[],
                workspace=workspace,
            )

        parts = _flatten_parts(parts)

        if not parts[0]:
            parts = parts[1:]

        if not parts:
            return DatabricksPath(
                kind=DatabricksPathKind.DBFS,
                parts=[],
                workspace=workspace,
            )

        head, *tail = parts

        if head == "dbfs":
            kind = DatabricksPathKind.DBFS
        elif head == "Workspace":
            kind = DatabricksPathKind.WORKSPACE
        elif head == "Volumes":
            kind = DatabricksPathKind.VOLUME
        else:
            raise ValueError(f"Invalid DatabricksPath prefix: {parts!r}")

        return DatabricksPath(
            kind=kind,
            parts=tail,
            workspace=workspace,
        )

    def __hash__(self):
        return hash((self.kind, tuple(self.parts)))

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

        built = DatabricksPath(
            kind=self.kind,
            parts=self.parts + other_parts,
            workspace=self.workspace,
        )

        return built

    def __enter__(self):
        self.safe_workspace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.safe_workspace.__exit__(exc_type, exc_val, exc_tb)

    def __str__(self):
        if self.kind == DatabricksPathKind.DBFS:
            return self.as_dbfs_api_path()
        elif self.kind == DatabricksPathKind.WORKSPACE:
            return self.as_workspace_api_path()
        elif self.kind == DatabricksPathKind.VOLUME:
            return self.as_files_api_path()
        else:
            raise ValueError(f"Unknown DatabricksPath kind: {self.kind!r}")

    def __repr__(self):
        return "dbfs://%s" % self.__str__()

    @property
    def parent(self):
        if not self.parts:
            return self

        if self._is_file is not None or self._is_dir is not None:
            _is_file, _is_dir = False, True
        else:
            _is_file, _is_dir = None, None

        built = DatabricksPath(
            kind=self.kind,
            parts=self.parts[:-1],
            workspace=self.workspace,
            _is_file=_is_file,
            _is_dir=_is_dir,
        )

        return built

    @property
    def safe_workspace(self):
        if self.workspace is None:
            from .workspace import Workspace

            self.workspace = Workspace()
        return self.workspace

    @safe_workspace.setter
    def safe_workspace(self, value):
        self.workspace = value

    @property
    def name(self) -> str:
        if not self.parts:
            return ""
        return self.parts[-1]

    @property
    def extension(self) -> str:
        name = self.name
        if '.' in name:
            return name.split('.')[-1]
        return ''

    def is_file(self):
        if self._is_file is None:
            self.refresh_status()
        return self._is_file

    def is_dir(self):
        if self._is_dir is None:
            self.refresh_status()
        return self._is_dir

    def volume_parts(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[PurePosixPath]]:
        if self.kind != DatabricksPathKind.VOLUME:
            return None, None, None, None

        catalog = self.parts[0] if len(self.parts) > 0 and self.parts[0] else None
        schema = self.parts[1] if len(self.parts) > 1 and self.parts[1] else None
        volume = self.parts[2] if len(self.parts) > 2 and self.parts[2] else None

        return catalog, schema, volume, self.parts[3:]

    def refresh_status(self):
        with self as connected:
            sdk = connected.safe_workspace.sdk()

            try:
                if connected.kind == DatabricksPathKind.VOLUME:
                    info = sdk.files.get_metadata(connected.as_files_api_path())

                    connected._raw_status = info
                    connected._is_file, connected._is_dir = True, False
                elif connected.kind == DatabricksPathKind.WORKSPACE:
                    info = sdk.workspace.get_status(connected.as_workspace_api_path())

                    is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
                    connected._raw_status = info
                    connected._is_file, connected._is_dir = not is_dir, is_dir
                else:
                    info = sdk.dbfs.get_status(connected.as_dbfs_api_path())

                    connected._raw_status = info
                    connected._is_file, connected._is_dir = (not info.is_dir), info.is_dir

                connected._raw_status_refresh_time = time.time()
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                found = next(connected.ls(fetch_size=1, recursive=False, raise_error=False), None)

                if found is None:
                    connected._is_file, connected._is_dir = False, False
                else:
                    connected._is_file, connected._is_dir = False, True

        return connected

    def clear_cache(self):
        self._raw_status = None
        self._raw_status_refresh_time = 0

        self._is_file = None
        self._is_dir = None

    # ---- API path normalization helpers ----

    def as_workspace_api_path(self) -> str:
        """
        Workspace API typically uses paths like /Users/... (not /Workspace/Users/...)
        so we strip the leading /Workspace when present.
        """
        return "/Workspace/%s" % "/".join(self.parts) if self.parts else "/Workspace"

    def as_dbfs_api_path(self) -> str:
        """
        DBFS REST wants absolute DBFS paths like /tmp/x.
        If the user passes /dbfs/tmp/x (FUSE-style), strip the /dbfs prefix.
        """
        return "/dbfs/%s" % "/".join(self.parts) if self.parts else "/dbfs"

    def as_files_api_path(self) -> str:
        """
        Files API takes absolute paths, e.g. /Volumes/<...>/file
        """
        return "/Volumes/%s" % "/".join(self.parts) if self.parts else "/Volumes"

    def exists(self) -> bool:
        if self.is_file():
            return True
        if self.is_dir():
            return True
        return False

    def mkdir(self, parents=True, exist_ok=True):
        """
        Create a new directory at this given path.
        """
        with self as connected:
            connected.clear_cache()

            try:
                if connected.kind == DatabricksPathKind.WORKSPACE:
                    connected.safe_workspace.sdk().workspace.mkdirs(self.as_workspace_api_path())
                elif connected.kind == DatabricksPathKind.VOLUME:
                    return connected._create_volume_dir(parents=parents, exist_ok=exist_ok)
                elif connected.kind == DatabricksPathKind.DBFS:
                    connected.safe_workspace.sdk().dbfs.mkdirs(self.as_dbfs_api_path())

                connected._is_file, connected._is_dir = False, True
            except (NotFound, ResourceDoesNotExist):
                if not parents or self.parent == self:
                    raise

                connected.parent.mkdir(parents=True, exist_ok=True)
                connected.mkdir(parents=False, exist_ok=exist_ok)
            except (AlreadyExists, ResourceAlreadyExists):
                if not exist_ok:
                    raise

    def _ensure_volume(self, exist_ok: bool = True):
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.safe_workspace.sdk()

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

    def _create_volume_dir(self, parents=True, exist_ok=True):
        path = self.as_files_api_path()
        sdk = self.safe_workspace.sdk()

        try:
            sdk.files.create_directory(path)
        except (BadRequest, NotFound, ResourceDoesNotExist) as e:
            if not parents:
                raise

            message = str(e)

            if "olume" in message and "not exist" in message:
                self._ensure_volume()

            sdk.files.create_directory(path)
        except (AlreadyExists, ResourceAlreadyExists, BadRequest):
            if not exist_ok:
                raise

        self.clear_cache()
        self._is_file, self._is_dir = False, True

    def remove(self, recursive: bool = True):
        if self.is_file():
            return self.rmfile()
        else:
            return self.rmdir(recursive=recursive)

    def rmfile(self):
        try:
            if self.kind == DatabricksPathKind.VOLUME:
                return self._remove_volume_file()
            elif self.kind == DatabricksPathKind.WORKSPACE:
                return self._remove_workspace_file()
            elif self.kind == DatabricksPathKind.DBFS:
                return self._remove_dbfs_file()
        finally:
            self.clear_cache()

    def _remove_volume_file(self):
        sdk = self.safe_workspace.sdk()

        try:
            sdk.files.delete(self.as_files_api_path())
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

    def _remove_workspace_file(self):
        sdk = self.safe_workspace.sdk()

        try:
            sdk.workspace.delete(self.as_workspace_api_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

    def _remove_dbfs_file(self):
        sdk = self.safe_workspace.sdk()

        try:
            sdk.dbfs.delete(self.as_dbfs_api_path(), recursive=True)
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            pass

    def rmdir(self, recursive: bool = True):
        with self as connected:
            try:
                if connected.kind == DatabricksPathKind.WORKSPACE:
                    connected.safe_workspace.sdk().workspace.delete(
                        self.as_workspace_api_path(),
                        recursive=recursive,
                    )
                elif connected.kind == DatabricksPathKind.VOLUME:
                    return self._remove_volume_dir(recursive=recursive)
                else:
                    connected.safe_workspace.sdk().dbfs.delete(
                        self.as_dbfs_api_path(),
                        recursive=recursive,
                    )
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                pass
            finally:
                connected.clear_cache()

    def _remove_volume_dir(self, recursive: bool = True):
        root_path = self.as_files_api_path()
        catalog_name, schema_name, volume_name, rel = self.volume_parts()

        sdk = self.safe_workspace.sdk()

        if rel:
            try:
                sdk.files.delete_directory(root_path)
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied) as e:
                message = str(e)

                if recursive and "directory is not empty" in message:
                    for child_path in self.ls():
                        child_path.remove(recursive=True)
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

        self.clear_cache()

    def ls(self, recursive: bool = False, fetch_size: int = None, raise_error: bool = True):
        if self.kind == DatabricksPathKind.VOLUME:
            for _ in self._ls_volume(recursive=recursive, fetch_size=fetch_size, raise_error=raise_error):
                yield _
        elif self.kind == DatabricksPathKind.WORKSPACE:
            for _ in self._ls_workspace(recursive=recursive, raise_error=raise_error):
                yield _
        elif self.kind == DatabricksPathKind.DBFS:
            for _ in self._ls_dbfs(recursive=recursive, raise_error=raise_error):
                yield _

    def _ls_volume(self, recursive: bool = False, fetch_size: int = None, raise_error: bool = True):
        catalog_name, schema_name, volume_name, rel = self.volume_parts()
        sdk = self.safe_workspace.sdk()

        if rel is None:
            if volume_name is None:
                try:
                    for info in sdk.volumes.list(
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                    ):
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts = [info.catalog_name, info.schema_name, info.name],
                            workspace=self.safe_workspace,
                            _is_file=False,
                            _is_dir=True,
                        )

                        if recursive:
                            for sub in base._ls_volume(recursive=recursive):
                                yield sub
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                    if raise_error:
                        raise
            elif schema_name is None:
                try:
                    for info in sdk.schemas.list(catalog_name=catalog_name):
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts=[info.catalog_name, info.name],
                            workspace=self.safe_workspace,
                            _is_file=False,
                            _is_dir=True,
                        )

                        if recursive:
                            for sub in base._ls_volume(recursive=recursive):
                                yield sub
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                    if raise_error:
                        raise
            else:
                try:
                    for info in sdk.catalogs.list():
                        base = DatabricksPath(
                            kind=DatabricksPathKind.VOLUME,
                            parts=[info.name],
                            workspace=self.safe_workspace,
                            _is_file=False,
                            _is_dir=True,
                        )

                        if recursive:
                            for sub in base._ls_volume(recursive=recursive):
                                yield sub
                        else:
                            yield base
                except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                    if raise_error:
                        raise
        else:
            try:
                for info in sdk.files.list_directory_contents(self.as_files_api_path(), page_size=fetch_size):
                    base = DatabricksPath(
                        kind=DatabricksPathKind.VOLUME,
                        parts=info.path.split("/")[2:],
                        workspace=self.safe_workspace,
                        _is_file=not info.is_directory,
                        _is_dir=info.is_directory,
                    )

                    if recursive and info.is_directory:
                        for sub in base._ls_volume(recursive=recursive):
                            yield sub
                    else:
                        yield base
            except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
                if raise_error:
                    raise

    def _ls_workspace(self, recursive: bool = True, raise_error: bool = True):
        sdk = self.safe_workspace.sdk()

        try:
            for info in sdk.workspace.list(self.as_workspace_api_path(), recursive=recursive):
                is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
                base = DatabricksPath(
                    kind=DatabricksPathKind.WORKSPACE,
                    parts=info.path.split("/")[2:],
                    workspace=self.safe_workspace,
                    _is_file=not is_dir,
                    _is_dir=is_dir,
                )
                yield base
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            if raise_error:
                raise

    def _ls_dbfs(self, recursive: bool = True, raise_error: bool = True):
        sdk = self.safe_workspace.sdk()

        try:
            # FIX: DBFS listing should use DBFS-normalized path, not workspace path
            p = self.as_dbfs_api_path()

            for info in sdk.dbfs.list(p, recursive=recursive):
                base = DatabricksPath(
                    kind=DatabricksPathKind.DBFS,
                    parts=info.path.split("/")[2:],
                    workspace=self.safe_workspace,
                    _is_file=not info.is_dir,
                    _is_dir=info.is_dir,
                )

                yield base
        except (NotFound, ResourceDoesNotExist, BadRequest, PermissionDenied):
            if raise_error:
                raise

    @contextmanager
    def open(
        self,
        mode="r",
        encoding=None,
    ) -> Iterator[Union[BinaryIO, io.TextIOBase]]:
        """
        Open this Databricks path using databricks-sdk's WorkspaceClient.

        Supported:
          - read:  "rb", "r"
          - write: "wb", "w"  (buffered; uploads on close for WORKSPACE/VOLUME)
        """
        if mode not in {"rb", "r", "wb", "w"}:
            raise ValueError(f"Unsupported mode {mode!r}. Use r/rb/w/wb.")

        if encoding is None:
            encoding = None if "b" in mode else "utf-8"
        reading = "r" in mode

        if reading:
            with self.open_read(encoding=encoding) as f:
                yield f
        else:
            with self.open_write(encoding=encoding) as f:
                yield f

    @contextmanager
    def open_read(self, encoding: str | None = None):
        with self as connected:
            if connected.kind == DatabricksPathKind.VOLUME:
                with connected._open_read_volume(encoding=encoding) as f:
                    yield f
            elif connected.kind == DatabricksPathKind.WORKSPACE:
                with connected._open_read_workspace(encoding=encoding) as f:
                    yield f
            else:
                with connected._open_read_dbfs(encoding=encoding) as f:
                    yield f

    @contextmanager
    def _open_read_volume(self, encoding: str | None = None):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_files_api_path()

        resp = workspace_client.files.download(path)
        raw = io.BytesIO(resp.contents.read())

        if encoding is not None:
            with io.TextIOWrapper(raw, encoding=encoding) as f:
                yield f
        else:
            with raw as f:
                yield f

    @contextmanager
    def _open_read_workspace(self, encoding: str | None = None):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_workspace_api_path()

        raw = workspace_client.workspace.download(path)  # returns BinaryIO

        if encoding is not None:
            raw = io.BytesIO(raw.read())
            with io.TextIOWrapper(raw, encoding=encoding) as f:
                yield f
        else:
            with raw as f:
                yield f

    @contextmanager
    def _open_read_dbfs(self, encoding: str | None = None):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_dbfs_api_path()

        raw = workspace_client.dbfs.open(path, read=True)

        if encoding is not None:
            with io.TextIOWrapper(raw, encoding=encoding) as f:
                yield f
        else:
            with raw as f:
                yield f

    @contextmanager
    def open_write(self, encoding: str | None = None):
        with self as connected:
            if connected.kind == DatabricksPathKind.VOLUME:
                with connected._open_write_volume(encoding=encoding) as f:
                    yield f
            elif connected.kind == DatabricksPathKind.WORKSPACE:
                with connected._open_write_workspace(encoding=encoding) as f:
                    yield f
            else:
                with connected._open_write_dbfs(encoding=encoding) as f:
                    yield f

    @contextmanager
    def _open_write_volume(self, encoding: str | None = None, overwrite: bool = True):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_files_api_path()

        buf = io.BytesIO()

        if encoding is not None:
            tw = io.TextIOWrapper(buf, encoding=encoding, write_through=True)
            try:
                yield tw
            finally:
                tw.flush()
                buf.seek(0)

                try:
                    workspace_client.files.upload(path, buf, overwrite=overwrite)
                except (NotFound, ResourceDoesNotExist, BadRequest):
                    self.parent.mkdir(parents=True, exist_ok=True)
                    workspace_client.files.upload(path, buf, overwrite=overwrite)

                tw.detach()
        else:
            try:
                yield buf
            finally:
                buf.seek(0)

                try:
                    workspace_client.files.upload(path, buf, overwrite=overwrite)
                except (NotFound, ResourceDoesNotExist, BadRequest):
                    self.parent.mkdir(parents=True, exist_ok=True)
                    workspace_client.files.upload(path, buf, overwrite=overwrite)

    @contextmanager
    def _open_write_workspace(self, encoding: str | None = None, overwrite: bool = True):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_workspace_api_path()

        buf = io.BytesIO()

        if encoding is not None:
            tw = io.TextIOWrapper(buf, encoding=encoding, write_through=True)
            try:
                yield tw
            finally:
                tw.flush()
                buf.seek(0)

                try:
                    workspace_client.workspace.upload(
                        path, buf, format=ImportFormat.AUTO, overwrite=overwrite
                    )
                except Exception as e:
                    message = str(e)
                    if "parent folder" in message and "does not exist" in message:
                        self.parent.mkdir(parents=True)
                        buf.seek(0)
                        workspace_client.workspace.upload(
                            path, buf, format=ImportFormat.AUTO, overwrite=overwrite
                        )
                    else:
                        raise

                tw.detach()
        else:
            try:
                yield buf
            finally:
                buf.seek(0)

                try:
                    workspace_client.workspace.upload(
                        path, buf, format=ImportFormat.AUTO, overwrite=overwrite
                    )
                except Exception as e:
                    message = str(e)
                    if "parent folder" in message and "does not exist" in message:
                        self.parent.mkdir(parents=True)
                        buf.seek(0)
                        workspace_client.workspace.upload(
                            path, buf, format=ImportFormat.AUTO, overwrite=overwrite
                        )
                    else:
                        raise

    @contextmanager
    def _open_write_dbfs(self, encoding: str | None = None, overwrite: bool = True):
        workspace_client = self.safe_workspace.sdk()
        path = self.as_dbfs_api_path()

        raw = workspace_client.dbfs.open(path, write=True, overwrite=overwrite)

        if encoding is not None:
            with io.TextIOWrapper(raw, encoding=encoding) as f:
                yield f
        else:
            with raw as f:
                yield f

        self.clear_cache()
        self._is_file, self._is_dir = True, False
