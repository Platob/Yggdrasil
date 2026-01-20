"""Optional pandas dependency helpers."""

try:
    import pandas  # type: ignore
    pandas = pandas

    PandasDataFrame = pandas.DataFrame
except ImportError:
    pandas = None

    class PandasDataFrame:
        pass


def require_pandas():
    """Ensure pandas is available before using pandas helpers.

    Returns:
        None.
    """
    if pandas is None:
        raise ImportError(
            "pandas is required to use this function. "
            "Install it with `pip install pandas`."
        )


__all__ = [
    "pandas",
    "require_pandas",
    "PandasDataFrame"
]
