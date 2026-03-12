"""Optional Databricks SDK dependency helpers."""

from __future__ import annotations

try:
    import databricks.sdk as databricks_sdk
except ImportError:
    from yggdrasil.environ import runtime_import_module

    databricks_sdk = runtime_import_module(
        module_name="databricks.sdk",
        pip_name="databricks-sdk>=0.80",
        install=True
    )

from databricks.sdk import WorkspaceClient  # type: ignore
from databricks.sdk.errors import DatabricksError  # type: ignore

__all__ = [
    "databricks_sdk",
    "WorkspaceClient",
    "DatabricksError",
]
