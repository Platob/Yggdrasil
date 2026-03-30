"""Backward-compatibility shim — canonical location is ``yggdrasil.databricks.fs.path_kind``."""
from ..fs.path_kind import DatabricksPathKind  # noqa: F401

__all__ = ["DatabricksPathKind"]
