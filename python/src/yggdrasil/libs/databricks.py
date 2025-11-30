import functools

try:
    import databricks
    import databricks.sdk  # type: ignore

    databricks = databricks
    databricks_sdk = databricks.sdk
except ImportError:
    databricks = None
    databricks_sdk = None


def require_databricks_sdk(_func=None):
    """
    Can be used as:

    @require_databricks_sdk
    def f(...): ...

    or

    @require_databricks_sdk()
    def f(...): ...
    """

    def decorator_require_databricks_sdk(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if databricks_sdk is None:
                raise ImportError(
                    "databricks_sdk is required to use this function. "
                    "Install it with `pip install databricks_sdk`."
                )
            return func(*args, **kwargs)

        return wrapper

    # Used as @require_databricks_sdk()
    if _func is None:
        return decorator_require_databricks_sdk

    # Used as @require_databricks_sdk
    return decorator_require_databricks_sdk(_func)


__all__ = [
    "databricks",
    "databricks_sdk",
    "require_databricks_sdk",
]
