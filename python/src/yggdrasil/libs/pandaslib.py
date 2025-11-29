import functools

try:
    import pandas  # type: ignore
    pandas = pandas
except ImportError:
    pandas = None


def require_pandas(_func=None):
    """
    Can be used as:

    @require_pandas
    def f(...): ...

    or

    @require_pandas()
    def f(...): ...
    """

    def decorator_require_pandas(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if pandas is None:
                raise ImportError(
                    "pandas is required to use this function. "
                    "Install it with `pip install pandas`."
                )
            return func(*args, **kwargs)

        return wrapper

    # Used as @require_pandas()
    if _func is None:
        return decorator_require_pandas

    # Used as @require_pandas
    return decorator_require_pandas(_func)


__all__ = [
    "pandas",
    "require_pandas",
]
