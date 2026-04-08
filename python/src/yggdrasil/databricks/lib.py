"""Optional Databricks SDK dependency helpers."""

from __future__ import annotations

import databricks.sdk as databricks_sdk
from databricks.sdk import WorkspaceClient  # type: ignore
from databricks.sdk.errors import DatabricksError  # type: ignore

__all__ = [
    "databricks_sdk",
    "WorkspaceClient",
    "DatabricksError",
]
