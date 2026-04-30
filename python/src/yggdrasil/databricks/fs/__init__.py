"""Databricks filesystem package.

Public surface — concrete subclasses + the abstract bases for
isinstance/typing use. Importing this package wires every concrete
subclass into the :class:`yggdrasil.io.fs.path.Path` dispatch
registry via each subclass's ``__init_subclass__`` hook (each
``class XPath(DatabricksPath): ...`` registration fires when this
module is loaded).
"""

from __future__ import annotations

from typing import Dict, Type

from .dbfs_path import DBFSPath
from .path import DatabricksPath
from .path_kind import DatabricksPathKind
from .table_path import TablePath
from .volume_path import VolumePath
from .workspace_path import WorkspacePath


__all__ = [
    # Abstract bases
    "DatabricksPath",
    "DatabricksPathKind",
    # Concrete paths
    "DBFSPath",
    "WorkspacePath",
    "VolumePath",
    "TablePath",
    # Dispatch tables
    "SCHEME_MAP",
    "SCHEME_TO_KIND",
]


# ---------------------------------------------------------------------------
# Scheme dispatch tables
# ---------------------------------------------------------------------------
#
# The :class:`Path` registry already routes scheme strings to
# subclasses via ``handles()``. These tables are the explicit
# version for callers that want a programmatic lookup (e.g. SQL
# helpers picking a backend by ``DatabricksPathKind``).

SCHEME_MAP: Dict[str, Type[DatabricksPath]] = {
    "dbfs": DBFSPath,
    "workspace": WorkspacePath,
    "volumes": VolumePath,
    "tables": TablePath,
}

SCHEME_TO_KIND: Dict[str, DatabricksPathKind] = {
    "dbfs": DatabricksPathKind.DBFS,
    "workspace": DatabricksPathKind.WORKSPACE,
    "volumes": DatabricksPathKind.VOLUME,
    "tables": DatabricksPathKind.TABLE,
}