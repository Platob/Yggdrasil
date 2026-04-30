""":class:`WorkspacePath` — ``/Workspace/...`` via the Workspace API.

Always SDK-mediated; no FUSE counterpart. Thinner than
:class:`DBFSPath` because the only access path is the Workspace
REST API: list / get_status / mkdirs / delete / download / upload.

The buffered-IO machinery lives in :class:`BytesIO`; this class
just plugs in the SDK download/upload calls via the
:meth:`_remote_download` / :meth:`_remote_upload` hooks declared
on :class:`DatabricksPath`.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from databricks.sdk.service.workspace import (
    ExportFormat,
    ImportFormat,
    ObjectType,
)

from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL
from ._errors import (
    ALREADY_EXISTS_ERRORS,
    NOT_FOUND_ERRORS,
    SDK_ERRORS,
)
from .path import DatabricksPath
from .path_kind import DatabricksPathKind

__all__ = ["WorkspacePath"]


LOGGER = logging.getLogger(__name__)


class WorkspacePath(DatabricksPath):
    """Path under ``/Workspace/...`` via the Workspace API."""

    scheme: ClassVar[str] = "dbfs+workspace"
    _NAMESPACE_PREFIX: ClassVar[str] = "/Workspace/"

    @property
    def kind(self) -> DatabricksPathKind:
        return DatabricksPathKind.WORKSPACE

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = self.url.path.lstrip("/")
        return "/Workspace/" + p if p else "/Workspace"

    # ==================================================================
    # SDK transport
    # ==================================================================

    def _remote_download(self, allow_not_found: bool = False) -> bytes:
        """Pull the full object via ``sdk.workspace.download``."""
        sdk = self._sdk()
        try:
            result = sdk.workspace.download(
                path=self.full_path(),
                format=ExportFormat.AUTO,
            )
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                return b""
            raise FileNotFoundError(self.full_path())
        except SDK_ERRORS:
            if allow_not_found:
                return b""
            raise

        if result is None:
            return b""
        return result.read()

    def _remote_upload(self, payload: bytes) -> None:
        """Push the full payload via ``sdk.workspace.upload``."""
        sdk = self._sdk()
        full_path = self.full_path()
        size = len(payload)

        LOGGER.debug("Uploading %s bytes to %s", size, self)
        try:
            sdk.workspace.upload(
                full_path, payload,
                format=ImportFormat.AUTO, overwrite=True,
            )
        except NOT_FOUND_ERRORS:
            # Surface as FileNotFoundError so the base class's
            # write_bytes can do its parent-dir-and-retry dance.
            raise FileNotFoundError(full_path)
        LOGGER.info("Wrote %s bytes to %s", size, self)

    # ==================================================================
    # SDK hooks — stat / ls / mkdir / remove
    # ==================================================================

    def _stat(self) -> PathStats:
        try:
            info = self._sdk().workspace.get_status(self.full_path())
        except SDK_ERRORS:
            found = next(self._ls(recursive=False, allow_not_found=True), None)
            if found is None:
                return PathStats(kind=PathKind.MISSING, size=0, mtime=None)
            return PathStats(
                kind=PathKind.DIRECTORY, size=0, mtime=found.mtime,
            )

        is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
        return PathStats(
            kind=PathKind.DIRECTORY if is_dir else PathKind.FILE,
            size=int(info.size or 0),
            mtime=(
                float(info.modified_at) / 1000.0
                if info.modified_at else None
            ),
        )

    def _ls(self, recursive=False, allow_not_found=True):
        try:
            for info in self._sdk().workspace.list(
                self.full_path(), recursive=recursive,
            ):
                api_path = info.path
                url_path = (
                    api_path[len("/Workspace"):]
                    if api_path.startswith("/Workspace")
                    else api_path
                )
                child = WorkspacePath(
                    url=URL(scheme="workspace", host=self.url.host, path=url_path),
                    client=self._client,
                )
                yield child
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _mkdir(self, parents=True, exist_ok=True):
        try:
            self._sdk().workspace.mkdirs(self.full_path())
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise

    def _remove_file(self, allow_not_found=True):
        # Workspace ``delete`` requires recursive=True for
        # notebook files (legacy quirk — notebooks-with-revisions
        # are tree-shaped on the API side).
        try:
            self._sdk().workspace.delete(self.full_path(), recursive=True)
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        path = self.full_path()
        try:
            self._sdk().workspace.delete(path, recursive=recursive)
            if not with_root:
                self._sdk().workspace.mkdirs(path)
        except SDK_ERRORS:
            if not allow_not_found:
                raise