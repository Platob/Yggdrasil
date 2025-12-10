from datetime import datetime
from typing import Optional, Sequence

from ..polarslib import polars as pl

__all__ = [
]

def join_coalesced(
    left: pl.DataFrame,
    right: pl.DataFrame,
    on: str | list[str],
    how: str = "left",
    suffix: str = "_right",
) -> pl.DataFrame:
    """
    Join two DataFrames and merge overlapping columns:
    - prefer values from `left`
    - fallback to `right` where left is null
    """

    # Normalize `on` to a set
    if isinstance(on, str):
        on_cols = {on}
    else:
        on_cols = set(on)

    # Columns that exist in both, excluding join keys
    common = (set(left.columns) & set(right.columns)) - on_cols

    # Regular join with a suffix
    joined = left.join(
        right,
        on=list(on_cols),
        how=how,
        suffix=suffix,
    )

    # Coalesce common columns and drop suffixed ones
    joined = joined.with_columns(
        [
            pl.coalesce(
                pl.col(name),
                pl.col(f"{name}{suffix}"),
            ).alias(name)
            for name in common
        ]
    ).drop([f"{name}{suffix}" for name in common])

    return joined

if pl is not None:
    setattr(pl.DataFrame, "join_coalesced", join_coalesced)