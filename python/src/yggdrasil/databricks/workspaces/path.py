"""Backward-compatibility shim — canonical location is ``yggdrasil.databricks.fs.path``."""
from __future__ import annotations

# Re-export everything from the new canonical location
from ..fs.path import (       # noqa: F401
    DatabricksPath,
    DBFSPath,
    WorkspacePath,
    VolumePath,
    TablePath,
    DatabricksStatResult,
)
from ..fs.path_kind import DatabricksPathKind  # noqa: F401

__all__ = [
    "DatabricksPath",
    "DBFSPath",
    "WorkspacePath",
    "VolumePath",
    "TablePath",
    "DatabricksPathKind",
    "DatabricksStatResult",
]
