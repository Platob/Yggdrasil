"""Optional Databricks SDK dependency helpers."""
from ..types.dummy_class import DummyModuleClass


class DatabricksDummyClass(DummyModuleClass):
    """Placeholder object that raises if Databricks SDK is required."""

    @classmethod
    def module_name(cls) -> str:
        return "databricks"


def require_databricks_sdk():
    """Ensure the Databricks SDK is available before use.

    Returns:
        None.
    """
    if databricks_sdk is None:
        raise ImportError(
            "databricks_sdk is required to use this function. "
            "Install it with `pip install databricks_sdk`."
        )


try:
    import databricks
    import databricks.sdk

    from databricks.sdk import WorkspaceClient

    databricks = databricks
    databricks_sdk = databricks.sdk
except ImportError:
    databricks = DatabricksDummyClass
    databricks_sdk = DatabricksDummyClass
    WorkspaceClient = DatabricksDummyClass


__all__ = [
    "databricks",
    "databricks_sdk",
    "require_databricks_sdk",
    "WorkspaceClient",
    "DatabricksDummyClass"
]
