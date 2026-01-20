"""Enumerations for Databricks path namespaces."""

from enum import Enum


__all__ = ["DatabricksPathKind"]


class DatabricksPathKind(str, Enum):
    """Supported Databricks path kinds for workspace, volumes, and DBFS."""
    WORKSPACE = 1
    VOLUME = 2
    DBFS = 3
