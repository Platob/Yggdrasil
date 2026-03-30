"""Workspace, filesystem, and path utilities for Databricks.

Path and IO symbols have been moved to ``yggdrasil.databricks.fs``.
They are re-exported here for backward compatibility.
"""

from .service import Workspaces
from .workspace import *

# Re-exports from the new canonical location
from ..fs.path_kind import DatabricksPathKind
from ..fs.path import (
    DatabricksPath,
    DBFSPath,
    WorkspacePath,
    VolumePath,
    TablePath,
    DatabricksStatResult,
)
from ..fs.io import DatabricksIO
