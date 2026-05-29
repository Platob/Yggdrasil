from __future__ import annotations

from typing import Any

from .common import StrictModel


class AggMeasure(StrictModel):
    column: str
    agg: str = "sum"  # sum|mean|min|max|count|median|std|var


class AggregateRequest(StrictModel):
    path: str
    group_by: list[str] = []
    measures: list[AggMeasure]
    limit: int = 500          # max result groups returned
    sort_desc: bool = True    # sort by the first measure descending


class AggregateResult(StrictModel):
    node_id: str
    path: str
    columns: list[str]
    rows: list[list[Any]]
    group_count: int          # distinct groups before the limit
    source_rows: int          # rows actually scanned
    truncated: bool           # source exceeded the analysis cap


class DescribeResult(StrictModel):
    node_id: str
    path: str
    statistics: list[str]     # e.g. count, mean, std, min, 25%, 50%, 75%, max
    columns: list[str]        # numeric columns described
    rows: list[list[Any]]     # one row per statistic, aligned to columns
    source_rows: int
    truncated: bool


class FinanceRequest(StrictModel):
    path: str
    column: str               # numeric series (e.g. price)
    order_by: str | None = None
    window: int = 20
    limit: int = 2000


class FinanceResult(StrictModel):
    node_id: str
    path: str
    column: str
    window: int
    index: list[Any]          # x axis (order_by values, or 0..n)
    value: list[float | None]
    pct_change: list[float | None]
    cum_return: list[float | None]
    roll_mean: list[float | None]
    roll_vol: list[float | None]  # rolling std of returns (volatility)
    truncated: bool
