"""Optional Polars dependency helpers."""
from yggdrasil.types.dummy_class import DummyModuleClass


class DummyPolarsClass(DummyModuleClass):

    @classmethod
    def module_name(cls) -> str:
        return "polars"


try:
    import polars  # type: ignore

    polars = polars

    PolarsDataFrame = polars.DataFrame
    PolarsSeries = polars.Series
    PolarsExpr = polars.Expr
    PolarsDataFrame = polars.DataFrame
    PolarsField = polars.Field
    PolarsSchema = polars.Schema
    PolarsDataType = polars.DataType


    def require_polars():
        """Ensure polars is available before using polars helpers.

        Returns:
            None.
        """
        return None

except ImportError:
    polars = None
    PolarsDataFrame = DummyPolarsClass
    PolarsSeries = DummyPolarsClass
    PolarsExpr = DummyPolarsClass
    PolarsDataFrame = DummyPolarsClass
    PolarsField = DummyPolarsClass
    PolarsSchema = DummyPolarsClass
    PolarsDataType = DummyPolarsClass


    def require_polars():
        """Ensure polars is available before using polars helpers.

        Returns:
            None.
        """
        import polars


__all__ = [
    "polars",
    "require_polars",
    "PolarsDataFrame",
    "PolarsSeries",
    "PolarsExpr",
    "PolarsDataFrame",
    "PolarsField",
    "PolarsSchema",
    "PolarsDataType",
    "DummyPolarsClass"
]



