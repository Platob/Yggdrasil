import functools

try:
    import polars  # type: ignore
    polars = polars
except ImportError:
    polars = None


def require_polars(_func=None):
    """
    Can be used as:

    @require_polars
    def f(...): ...

    or

    @require_polars()
    def f(...): ...
    """

    def decorator_require_polars(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if polars is None:
                raise ImportError(
                    "polars is required to use this function. "
                    "Install it with `pip install polars`."
                )
            return func(*args, **kwargs)

        return wrapper

    # Used as @require_polars()
    if _func is None:
        return decorator_require_polars

    # Used as @require_polars
    return decorator_require_polars(_func)


__all__ = [
    "polars",
    "require_polars",
]
