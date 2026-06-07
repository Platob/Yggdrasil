"""``Wheel`` — a single wheel in the workspace PyPI-like registry.

A light handle over a ``.whl`` at ``/Workspace/Shared/pypi/<dist>/<wheel>``:
its distribution / version / python tag parsed from the filename, plus the
file ops (exists / read / download / delete) routed through the workspace
filesystem. Built by the :class:`~yggdrasil.databricks.wheels.service.Wheels`
service; not constructed directly.
"""
from __future__ import annotations

import re
from pathlib import Path as _LocalPath
from typing import TYPE_CHECKING, Optional

from ..resource import DatabricksResource

if TYPE_CHECKING:
    from ..fs.workspace_path import WorkspacePath


class Wheel(DatabricksResource):
    """A deployed wheel — ``<dist>-<version>-<pytag>.whl`` in the registry."""

    def __init__(
        self,
        service=None,
        *,
        path: str,
        dist: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        super().__init__(service=service)
        #: Full workspace path of the ``.whl``.
        self.path = str(path)
        name = _LocalPath(self.path).name
        parts = name[:-4].split("-") if name.endswith(".whl") else name.split("-")
        self.dist = dist or (re.sub(r"[-_.]+", "-", parts[0]).lower() if parts else name)
        self.version = version or (parts[1] if len(parts) >= 2 else None)

    @property
    def name(self) -> str:
        """The wheel filename (``ygg-0.8.57-py3-none-any.whl``)."""
        return _LocalPath(self.path).name

    @property
    def workspace_path(self) -> "WorkspacePath":
        """The :class:`WorkspacePath` handle for the wheel."""
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
        return f"Wheel({self.path!r})"
