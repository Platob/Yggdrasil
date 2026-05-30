from __future__ import annotations

from typing import Any

from .common import StrictModel


class CastSpec(StrictModel):
    column: str
    dtype: str               # int|float|double|bool|string|date|datetime
    tz: str | None = None    # for datetime: target timezone (default UTC)


class FilterSpec(StrictModel):
    column: str
    op: str                  # ==|!=|>|>=|<|<=|contains|in|is_null|not_null
    value: Any | None = None


class Transform(StrictModel):
    """Reusable lazy transform: filters (predicate pushdown) + casts (incl.
    timezone, UTC by default) + projection + row limit."""
    filters: list[FilterSpec] = []
    casts: list[CastSpec] = []
    columns: list[str] | None = None
    limit: int | None = None


class AggMeasure(StrictModel):
    column: str
    agg: str = "sum"  # sum|mean|min|max|count|median|std|var


class AggregateRequest(StrictModel):
    path: str
    group_by: list[str] = []
    measures: list[AggMeasure]
    filters: list[FilterSpec] = []
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


class SeriesRequest(StrictModel):
    path: str
    column: str               # numeric series to plot
    x: str | None = None      # x/order column (else row index)
    points: int = 800         # target buckets — the grid asks for ~viewport width
    x_min: float | None = None  # zoom window (predicate-pushed into the scan)
    x_max: float | None = None
    filters: list[FilterSpec] = []


class SeriesResult(StrictModel):
    node_id: str
    path: str
    column: str
    x: list[Any]
    y: list[float | None]     # per-bucket mean
    y_min: list[float | None]  # per-bucket envelope (preserves spikes)
    y_max: list[float | None]
    source_rows: int          # rows in the (possibly zoomed) range
    sampled: bool             # True when downsampled into buckets


class OhlcRequest(StrictModel):
    path: str
    column: str               # price
    x: str | None = None      # order/time column
    volume: str | None = None
    buckets: int = 120
    filters: list[FilterSpec] = []


class ExportRequest(StrictModel):
    path: str
    fmt: str = "csv"          # csv|parquet|json|ndjson|arrow|xlsx
    transform: Transform = Transform()


class OhlcResult(StrictModel):
    node_id: str
    path: str
    column: str
    x: list[Any]
    open: list[float | None]
    high: list[float | None]
    low: list[float | None]
    close: list[float | None]
    volume: list[float | None] | None = None
    bars: int
    source_rows: int


class IndicatorSpec(StrictModel):
    type: str                 # rsi | macd | ema | bb (bollinger bands)
    params: dict[str, Any] = {}


class IndicatorRequest(StrictModel):
    path: str
    column: str               # price column
    x: str | None = None      # time/order column (else row index)
    indicators: list[IndicatorSpec]
    filters: list[FilterSpec] = []


class IndicatorSeries(StrictModel):
    type: str
    name: str
    # Different indicators emit different output keys: rsi -> {"rsi": [...]},
    # macd -> {"macd", "signal", "histogram"}, bb -> {"middle", "upper", "lower"}.
    values: dict[str, list[float | None]]


class IndicatorResult(StrictModel):
    node_id: str
    path: str
    column: str
    x: list[Any]
    source_rows: int
    indicators: list[IndicatorSeries]
