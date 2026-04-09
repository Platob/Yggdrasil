"""Optional Databricks SDK dependency helpers."""

from __future__ import annotations

try:
    import databricks.sdk as databricks_sdk
    from databricks.sdk import AccountClient, WorkspaceClient
    from databricks.sdk.config import Config
    from databricks.sdk.errors import DatabricksError
except ImportError:
    from yggdrasil.environ import PyEnv

    databricks_sdk = PyEnv.runtime_import_module(
        module_name="databricks.sdk",
        pip_name="databricks-sdk",
        install=True,
    )
    _config_module = PyEnv.runtime_import_module(
        module_name="databricks.sdk.config",
        pip_name="databricks-sdk",
        install=True,
    )
    _errors_module = PyEnv.runtime_import_module(
        module_name="databricks.sdk.errors",
        pip_name="databricks-sdk",
        install=True,
    )

    AccountClient = databricks_sdk.AccountClient
    WorkspaceClient = databricks_sdk.WorkspaceClient
    Config = _config_module.Config
    DatabricksError = _errors_module.DatabricksError

__all__ = [
    "databricks_sdk",
    "AccountClient",
    "WorkspaceClient",
    "Config",
    "DatabricksError",
]
