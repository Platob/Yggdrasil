try:
    import pandas  # type: ignore
    pandas = pandas
except ImportError:
    pandas = None


def require_pandas():
    if pandas is None:
        raise ImportError(
            "pandas is required to use this function. "
            "Install it with `pip install pandas`."
        )


__all__ = [
    "pandas",
    "require_pandas",
]
