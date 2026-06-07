"""``Wheel`` — a single wheel in the workspace PyPI-like registry.

A handle over a ``.whl`` at ``/Workspace/Shared/pypi/<dist>/<wheel>``: its
distribution + :class:`~yggdrasil.version.VersionInfo` parsed from the filename,
plus the file ops (exists / read / download / delete) routed through the
workspace filesystem. Built by the
:class:`~yggdrasil.databricks.wheels.service.Wheels` service.
"""
from __future__ import annotations

from pathlib import Path as _LocalPath
from typing import TYPE_CHECKING, Optional

from yggdrasil.version import VersionInfo

from ..resource import DatabricksResource

if TYPE_CHECKING:
    from ..fs.workspace_path import WorkspacePath


class Wheel(DatabricksResource):
    """A deployed wheel — ``<dist>-<version>-<pytag>.whl`` in the registry."""

    def __init__(self, service=None, *, path: str) -> None:
        super().__init__(service=service)
        from .service import wheel_parts

        self.path = str(path)
        #: Distribution name + :class:`VersionInfo` + python tag, from the filename.
        self.dist, self.version, self.tag = wheel_parts(self.path)

    @property
    def name(self) -> str:
        return _LocalPath(self.path).name

    @property
    def workspace_path(self) -> "WorkspacePath":
        from ..path import DatabricksPath

        return DatabricksPath.from_(self.path, client=self.client)

    def exists(self) -> bool:
        return self.workspace_path.exists()

    def read_bytes(self) -> bytes:
        return self.workspace_path.read_bytes()

    def download(self, dest: "str | _LocalPath") -> _LocalPath:
        """Pull the wheel down to local *dest* (a file or directory)."""
        dest = _LocalPath(dest)
        if dest.is_dir():
            dest = dest / self.name
        dest.write_bytes(self.read_bytes())
        return dest

    def delete(self) -> "Wheel":
        self.workspace_path.unlink(missing_ok=True)
        return self

    def full_path(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"Wheel({self.dist!r}, {self.version!s})"
