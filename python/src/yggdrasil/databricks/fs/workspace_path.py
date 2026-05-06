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

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL
from ._errors import (
    ALREADY_EXISTS_ERRORS,
    NOT_FOUND_ERRORS,
    SDK_ERRORS,
    retry_sdk_call,
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

    def _remote_download(self, allow_not_found: bool = False) -> BytesIO:
        """Pull the full object via ``sdk.workspace.download``.

        Drains the SDK response into a project :class:`BytesIO`
        (spill-capable) so large notebooks/artifacts don't have to
        live entirely in RAM. The base ``read_bytes`` /
        ``retry_sdk_call`` wrapper handles transient transport
        flakes around the call.
        """
        sdk = self._sdk()
        out = BytesIO()
        try:
            result = sdk.workspace.download(
                path=self.full_path(),
                format=ExportFormat.AUTO,
            )
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise FileNotFoundError(self.full_path())
        except SDK_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise

        if result is None:
            out.seek(0)
            return out

        try:
            while True:
                chunk = result.read(1 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise FileNotFoundError(self.full_path())

        out.seek(0)
        return out

    def _remote_upload(self, payload: BytesIO) -> None:
        """Push the full payload via ``sdk.workspace.upload``.

        ``payload`` is the project :class:`BytesIO`. The Workspace
        SDK ``upload`` accepts a file-like object, so we hand the
        buffer through directly (it satisfies ``read``/``seek``).
        Resetting before each retry is handled by the base via
        ``on_retry`` in :meth:`DatabricksPath.write_bytes`.
        """
        sdk = self._sdk()
        full_path = self.full_path()
        size = getattr(payload, "size", None)

        LOGGER.debug("Uploading %r bytes to %s", size, self)
        try:
            sdk.workspace.upload(
                full_path, payload,
                format=ImportFormat.AUTO, overwrite=True,
            )
        except NOT_FOUND_ERRORS:
            # Surface as FileNotFoundError so the base class's
            # write_bytes can do its parent-dir-and-retry dance.
            raise FileNotFoundError(full_path)
        LOGGER.info("Wrote %r bytes to %s", size, self)

    # ==================================================================
    # SDK hooks — stat / ls / mkdir / remove
    # ==================================================================

    def _stat(self) -> IOStats:
        try:
            info = retry_sdk_call(
                self._sdk().workspace.get_status, self.full_path(),
            )
        except SDK_ERRORS:
            found = next(self._ls(recursive=False, allow_not_found=True), None)
            if found is None:
                return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)
            return IOStats(
                kind=IOKind.DIRECTORY, size=0,
                mtime=float(found.mtime or 0.0),
            )

        is_dir = info.object_type in (ObjectType.DIRECTORY, ObjectType.REPO)
        return IOStats(
            kind=IOKind.DIRECTORY if is_dir else IOKind.FILE,
            size=int(info.size or 0),
            mtime=(
                float(info.modified_at) / 1000.0
                if info.modified_at else 0.0
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
            retry_sdk_call(self._sdk().workspace.mkdirs, self.full_path())
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise

    def _remove_file(self, allow_not_found=True):
        # Workspace ``delete`` requires recursive=True for
        # notebook files (legacy quirk — notebooks-with-revisions
        # are tree-shaped on the API side).
        try:
            retry_sdk_call(
                self._sdk().workspace.delete,
                self.full_path(), recursive=True,
            )
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        path = self.full_path()
        try:
            retry_sdk_call(
                self._sdk().workspace.delete, path, recursive=recursive,
            )
            if not with_root:
                retry_sdk_call(self._sdk().workspace.mkdirs, path)
        except SDK_ERRORS:
            if not allow_not_found:
                raise