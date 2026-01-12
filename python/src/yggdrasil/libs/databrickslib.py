"""libs.databrickslib module documentation."""

try:
    import databricks
    import databricks.sdk  # type: ignore

    databricks = databricks
    databricks_sdk = databricks.sdk
except ImportError:
    class _DatabricksDummy:
        def __getattr__(self, item):
            """
            __getattr__ documentation.
            
            Args:
                item: Parameter.
            
            Returns:
                The result.
            """

            require_databricks_sdk()

    databricks = _DatabricksDummy
    databricks_sdk = _DatabricksDummy


def require_databricks_sdk():
    """
    require_databricks_sdk documentation.
    
    Args:
        None.
    
    Returns:
        The result.
    """

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
