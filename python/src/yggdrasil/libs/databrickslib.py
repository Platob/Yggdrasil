try:
    import databricks
    import databricks.sdk  # type: ignore

    databricks = databricks
    databricks_sdk = databricks.sdk
except ImportError:
    databricks = None
    databricks_sdk = None


def require_databricks_sdk():
    if databricks_sdk is None:
        raise ImportError(
            "databricks_sdk is required to use this function. "
            "Install it with `pip install databricks_sdk`."
        )


__all__ = [
    "databricks",
    "databricks_sdk",
    "require_databricks_sdk",
]
