"""Optional pandas dependency helpers."""

try:
    import pandas  # type: ignore
    pandas = pandas
except ImportError:
    pandas = None


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
]
