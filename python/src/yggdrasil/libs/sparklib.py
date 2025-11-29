import functools

try:
    import pyspark  # type: ignore
    from pyspark.sql import SparkSession

    pyspark = pyspark
    SparkSession = SparkSession
except ImportError:
    pyspark = None
    SparkSession = None


def require_pyspark(_func=None, *, active_session: bool = False):
    """
    Can be used as:

    @require_pyspark
    def f(...): ...

    or

    @require_pyspark(active_session=True)
    def f(...): ...
    """

    def decorator_require_pyspark(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1) pyspark must be importable
            if pyspark is None:
                raise ImportError(
                    "pyspark is required to use this function. "
                    "Install it or run inside a Spark/Databricks environment."
                )

            # 2) Optionally require an active SparkSession
            if active_session:
                if SparkSession is None:
                    raise ImportError(
                        "pyspark.sql.SparkSession is required to check for an active session."
                    )
                if SparkSession.getActiveSession() is None:
                    raise RuntimeError(
                        "An active SparkSession is required to use this function. "
                        "Create one with SparkSession.builder.getOrCreate()."
                    )

            return func(*args, **kwargs)

        return wrapper

    # Used as @require_pyspark(active_session=True)
    if _func is None:
        return decorator_require_pyspark

    # Used as @require_pyspark
    return decorator_require_pyspark(_func)


__all__ = [
    "pyspark",
    "require_pyspark",
    "SparkSession"
]
