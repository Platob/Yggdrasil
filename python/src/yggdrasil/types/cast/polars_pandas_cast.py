"""Polars <-> pandas conversion helpers via Arrow."""

from typing import Optional

from .arrow_cast import CastOptions
from .registry import register_converter

# Reuse existing Polars <-> Arrow helpers
from .polars_cast import (
    polars_dataframe_to_arrow_table,
    arrow_table_to_polars_dataframe,
)

# Reuse existing pandas <-> Arrow helpers
from .pandas_cast import (
    pandas_dataframe_to_arrow_table,
    arrow_table_to_pandas_dataframe,
)

from ...libs.polarslib import polars, require_polars
from ...libs.pandaslib import pandas, require_pandas

__all__ = [
    "polars_dataframe_to_pandas_dataframe",
    "pandas_dataframe_to_polars_dataframe",
]

# ---------------------------------------------------------------------------
# Type aliases + decorator wrapper (safe when one side is missing)
# ---------------------------------------------------------------------------

if polars is not None and pandas is not None:
    require_polars()
    require_pandas()

    PolarsDataFrame = polars.DataFrame
    PandasDataFrame = pandas.DataFrame

    def polars_pandas_converter(*args, **kwargs):
        """Return a register_converter wrapper when both libs are available.

        Args:
            *args: Converter registration args.
            **kwargs: Converter registration kwargs.

        Returns:
            Converter decorator.
        """
        return register_converter(*args, **kwargs)

else:
    # Dummy stand-ins so decorators/annotations don't explode if one lib is absent
    class _Dummy:  # pragma: no cover - only used when Polars or pandas not installed
        """Placeholder type when Polars or pandas are unavailable."""
        pass

    PolarsDataFrame = _Dummy
    PandasDataFrame = _Dummy

    def polars_pandas_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        """Return a no-op decorator when dependencies are missing.

        Args:
            *_args: Ignored positional args.
            **_kwargs: Ignored keyword args.

        Returns:
            No-op decorator.
        """
        def _decorator(func):
            """Return the function unchanged.

            Args:
                func: Callable to return.

            Returns:
                Unchanged callable.
            """
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Polars DataFrame <-> pandas DataFrame via Arrow
# ---------------------------------------------------------------------------


@polars_pandas_converter(PolarsDataFrame, PandasDataFrame)
def polars_dataframe_to_pandas_dataframe(
    dataframe: "polars.DataFrame",
    cast_options: Optional[CastOptions] = None,
) -> "pandas.DataFrame":
    """
    Convert a Polars DataFrame to a pandas DataFrame using Arrow as the bridge.

    Flow:
      Polars DataFrame
        -> (polars_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_pandas_dataframe) pandas.DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the Polars -> Arrow side via
        polars_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_pandas_dataframe to avoid double-casting.
    """
    if polars is None or pandas is None:
        raise RuntimeError("Both polars and pandas are required for this conversion")

    opts = CastOptions.check_arg(cast_options)

    # Polars -> Arrow (includes Arrow-side casting if target_schema is set)
    table = polars_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> pandas (no extra casting; table already conforms to target schema)
    return arrow_table_to_pandas_dataframe(table, None)


@polars_pandas_converter(PandasDataFrame, PolarsDataFrame)
def pandas_dataframe_to_polars_dataframe(
    dataframe: "pandas.DataFrame",
    cast_options: Optional[CastOptions] = None,
) -> "polars.DataFrame":
    """
    Convert a pandas DataFrame to a Polars DataFrame using Arrow as the bridge.

    Flow:
      pandas.DataFrame
        -> (pandas_dataframe_to_arrow_table) pyarrow.Table
        -> (arrow_table_to_polars_dataframe) Polars DataFrame

    Casting behavior:
      - ArrowCastOptions are applied on the pandas -> Arrow side via
        pandas_dataframe_to_arrow_table.
      - The resulting Arrow table is already in the target schema, so we pass
        `None` to arrow_table_to_polars_dataframe to avoid double-casting.
    """
    if polars is None or pandas is None:
        raise RuntimeError("Both polars and pandas are required for this conversion")

    opts = CastOptions.check_arg(cast_options)

    # pandas -> Arrow (includes Arrow-side casting if target_schema is set)
    table = pandas_dataframe_to_arrow_table(dataframe, opts)

    # Arrow -> Polars (no extra casting; table already conforms to target schema)
    return arrow_table_to_polars_dataframe(table, None)
