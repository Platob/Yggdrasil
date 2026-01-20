"""Optional Polars dependency helpers."""

try:
    import polars  # type: ignore

    polars = polars

    PolarsDataFrame = polars.DataFrame
except ImportError:
    polars = None

    class PolarsDataFrame:
        pass

__all__ = [
    "polars",
    "require_polars",
    "PolarsDataFrame"
]


def require_polars():
    """Ensure polars is available before using polars helpers.

    Returns:
        None.
    """
    if polars is None:
        raise ImportError(
            "polars is required to use this function. "
            "Install it with `pip install polars`."
        )
