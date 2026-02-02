"""Optional pandas dependency helpers."""

from ..types.dummy_class import DummyModuleClass

class DummyPandasClass(DummyModuleClass):

    @classmethod
    def module_name(cls) -> str:
        return "pandas"


try:
    import pandas  # type: ignore
    PandasDataFrame = pandas.DataFrame
    PandasSeries = pandas.Series
except ImportError:
    pandas = None
    PandasDataFrame = DummyPandasClass
    PandasSeries = DummyPandasClass


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
    "PandasDataFrame",
    "PandasSeries",
    "DummyPandasClass",
]
