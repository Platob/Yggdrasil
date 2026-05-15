"""Databricks filesystem package — :class:`Path` subclasses for
``/dbfs``, ``/Volumes``, and ``/Workspace``.

Importing this package wires every concrete subclass into the
:class:`yggdrasil.io.holder.Holder` scheme registry so
``Holder(url="dbfs+dbfs://...")`` / ``dbfs+volume://...`` /
``dbfs+workspace://...`` all dispatch to the right Path class.
The un-qualified ``dbfs://`` family URL is also supported and
dispatched by the leading namespace in the URL path.
"""

from __future__ import annotations

from ..path import DatabricksPath
from .dbfs_path import DBFSPath
from .volume_path import VolumePath
from .workspace_path import WorkspacePath


__all__ = [
    "DatabricksPath",
    "DBFSPath",
    "VolumePath",
    "WorkspacePath",
]
