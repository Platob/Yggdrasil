from __future__ import annotations

from typing import Optional, Union

import polars as pl

from yggdrasil.data.cast import CastOptions, register_converter

__all__ = [
    "register_converter",
    "cast_polars_array",
    "cast_polars_dataframe",
    "cast_polars_lazyframe",
]


@register_converter(pl.Series, pl.Series)
@register_converter(pl.Expr, pl.Expr)
def cast_polars_array(
    array: Union[pl.Series, pl.Expr],
    options: Optional[CastOptions] = None,
) -> Union[pl.Series, pl.Expr]:
    return CastOptions.check(options).cast_polars_series(array)


@register_converter(pl.DataFrame, pl.DataFrame)
def cast_polars_dataframe(
    df: pl.DataFrame,
    options: Optional[CastOptions] = None,
) -> pl.DataFrame:
    return CastOptions.check(options).cast_polars_tabular(df)


@register_converter(pl.LazyFrame, pl.LazyFrame)
def cast_polars_lazyframe(
    df: pl.LazyFrame,
    options: Optional[CastOptions] = None,
) -> pl.LazyFrame:
    return CastOptions.check(options).cast_polars_tabular(df)
