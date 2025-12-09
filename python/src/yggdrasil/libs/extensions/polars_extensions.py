from datetime import datetime
from typing import Optional, Sequence

from ..polarslib import polars as pl

__all__ = [
    "resample_step_intervals"
]


def _normalize_freq(freq: str) -> str:
    f = freq.strip().lower()

    aliases = {
        "15min": "15m",
        "15mins": "15m",
        "15m": "15m",

        "hour": "1h",
        "hourly": "1h",
        "1h": "1h",
        "h": "1h",

        "day": "1d",
        "daily": "1d",
        "1d": "1d",
        "d": "1d",
    }

    return aliases.get(f, freq)  # passthrough like "30m", "2h", etc.


def resample_step_intervals(
    df: "pl.DataFrame",
    ts_col: str,
    next_ts_col: str,
    freq: str,                         # "15min", "1h", "daily", ...
    *,
    group_cols: Optional[Sequence[str]] = None,
    value_col: str = "value",
    begin: Optional[datetime] = None,
    end: Optional[datetime] = None,
    time_unit: str = "us",
) -> "pl.DataFrame":
    """
    Resample interval-style step data:

        <group cols...>, timestamp, next_timestamp, value, [other cols...]

    onto a regular time grid with the given frequency.

    If group_cols is None, use all columns except {ts_col, next_ts_col, value_col}
    as grouping columns.
    """
    freq_norm = _normalize_freq(freq)

    # Auto-detect group columns if not provided
    if group_cols is None:
        exclude = {ts_col, next_ts_col, value_col}
        group_cols = [c for c in df.columns if c not in exclude]

    group_cols = list(group_cols)  # ensure list

    # Sort by group cols + timestamp
    sort_cols = group_cols + [ts_col] if group_cols else [ts_col]
    df = df.sort(sort_cols)

    # Infer begin / end if missing
    if begin is None:
        begin = df.select(pl.col(ts_col).min()).item()

    if end is None:
        max_next = df.select(pl.col(next_ts_col).max()).item()
        if max_next is not None:
            end = max_next
        else:
            end = df.select(pl.col(ts_col).max()).item()

    # Build time grid
    grid_times = pl.datetime_range(
        begin,
        end,
        interval=freq_norm,
        time_unit=time_unit,
        eager=True,
    )
    grid = pl.DataFrame({ts_col: grid_times})

    # Cross join with group columns, if any
    if group_cols:
        groups = df.select(group_cols).unique()
        grid = groups.join(grid, how="cross").sort(sort_cols)

    # As-of join config
    join_kwargs = dict(
        left_on=ts_col,
        right_on=ts_col,
        strategy="backward",
    )
    if group_cols:
        join_kwargs["by"] = group_cols

    sampled = grid.join_asof(df, **join_kwargs)

    # Clip to actual intervals: only keep grid ts inside [timestamp, next_timestamp)
    sampled = sampled.filter(
        pl.col(next_ts_col).is_null() | (pl.col(ts_col) < pl.col(next_ts_col))
    )

    # Figure out what to keep:
    # - group columns
    # - all "other" columns carried from df
    # - ts_col, value_col
    base_cols = group_cols if group_cols else []
    extra_cols = [
        c
        for c in sampled.columns
        if c not in set(base_cols + [ts_col, next_ts_col, value_col])
    ]

    cols_keep = base_cols + extra_cols + [ts_col, value_col]

    sampled = sampled.select(cols_keep)

    # Build new next_timestamp on the resampled grid
    if group_cols:
        sampled = sampled.with_columns(
            pl.col(ts_col).shift(-1).over(group_cols).alias(next_ts_col)
        )
    else:
        sampled = sampled.with_columns(
            pl.col(ts_col).shift(-1).alias(next_ts_col)
        )

    return sampled


if pl is not None:
    setattr(pl.DataFrame, "resample_step_intervals", resample_step_intervals)