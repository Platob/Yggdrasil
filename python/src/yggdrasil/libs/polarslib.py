"""libs.polarslib module documentation."""

try:
    import polars  # type: ignore

    polars = polars
except ImportError:
    polars = None


__all__ = [
    "polars",
    "require_polars",
]


def require_polars():
    """
    require_polars documentation.
    
    Args:
        None.
    
    Returns:
        The result.
    """

    if polars is None:
        raise ImportError(
            "polars is required to use this function. "
            "Install it with `pip install polars`."
        )
