import functools
import pyarrow as pa

try:
    import polars  # type: ignore
    polars = polars

    ARROW_TO_POLARS = {
        pa.bool_(): polars.Boolean()
    }
except ImportError:
    polars = None
    ARROW_TO_POLARS = {}


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
    "arrow_type_to_polars"
]


def arrow_type_to_polars(arrow_type: pa.DataType) -> "polars.DataType":
    existing = ARROW_TO_POLARS.get(arrow_type)

    if existing is not None:
        return existing

    raise ValueError()