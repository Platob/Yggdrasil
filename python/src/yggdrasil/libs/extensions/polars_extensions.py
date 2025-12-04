import datetime as dt
from typing import Sequence, Optional

from ..polarslib import polars as pl

__all__ = []


def time_resample(
    df: "pl.DataFrame",
    time_column: str,
    every: str | dt.timedelta,
    *,
    time_unit: Optional[str] = None,
    time_zone: Optional[str] = None,
    group_by: str | Sequence[str] | None = None,
    maintain_order: bool = False,  # kept for API compatibility
    fill_strategy: Optional[str] = None,
) -> "pl.DataFrame":
    """
    Upsample in the *target* timezone by:

    - casting `time_column` to target `pl.Datetime`,
    - optionally deduplicating on (group_by..., time_column) keeping first,
    - building a regular time axis in the target tz,
    - joining original data onto that axis,
    - optionally filling nulls.
    """

    # --- figure out source + target datetime types ---
    source_dt = df.schema[time_column]
    if not isinstance(source_dt, pl.Datetime):
        source_dt = pl.Datetime(time_unit=time_unit or "us", time_zone=time_zone)

    target_dt = pl.Datetime(
        time_unit=time_unit or source_dt.time_unit,
        time_zone=time_zone or source_dt.time_zone,
    )

    # --- cast to target datetime (including tz conversion if needed) ---
    df = df.with_columns(pl.col(time_column).cast(target_dt))

    # standard fast path: no DST headaches, just use native upsample
    result = df.upsample(
        time_column,
        every=every,
        group_by=group_by,
        maintain_order=maintain_order,
    )

    # --- DST / TZ case detection ---
    no_need_dst_check = (
        source_dt.time_zone is None
        or target_dt.time_zone is None
        or source_dt.time_zone == target_dt.time_zone
    )

    if no_need_dst_check:
        return result

    # --- build base df using distinct group_by combos × datetime_range ---
    if group_by:
        group_by_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    else:
        group_by_cols = []

    # global min/max in *target* tz
    start_end = df.select(
        pl.col(time_column).min().alias("start"),
        pl.col(time_column).max().alias("end"),
    ).row(0)
    start, end = start_end

    # base time axis in the target tz using polars.datetime_range
    time_axis = pl.DataFrame(
        {
            time_column: pl.datetime_range(
                start=start,
                end=end,
                interval=every,
                time_unit=target_dt.time_unit,
                time_zone=target_dt.time_zone,
            )
        }
    )

    if group_by_cols:
        # distinct group keys
        distinct_groups = df.select(group_by_cols).unique()

        # cartesian product: each group gets full time axis
        base = distinct_groups.join(time_axis, how="cross")
    else:
        base = time_axis

    # left join original data onto the dense base grid
    join_keys = group_by_cols + [time_column] if group_by_cols else [time_column]
    out = base.join(df, on=join_keys, how="left")

    # optional fill strategy (only on value columns, not keys / time axis)
    if fill_strategy is not None:
        value_cols = [c for c in out.columns if c not in join_keys]
        out = out.with_columns(
            [pl.col(value_cols).fill_null(strategy=fill_strategy)]
        )

    return out
