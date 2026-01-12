"""libs.pandaslib module documentation."""

try:
    import pandas  # type: ignore
    pandas = pandas
except ImportError:
    pandas = None


def require_pandas():
    """
    require_pandas documentation.
    
    Args:
        None.
    
    Returns:
        The result.
    """

    if pandas is None:
        raise ImportError(
            "pandas is required to use this function. "
            "Install it with `pip install pandas`."
        )


__all__ = [
    "pandas",
    "require_pandas",
]
