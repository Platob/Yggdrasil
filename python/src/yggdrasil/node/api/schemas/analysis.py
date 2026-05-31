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
    # Annualization for the scalar risk metrics: trading periods per year
    # (252 daily, 52 weekly, 12 monthly) and the per-year risk-free rate used
    # for Sharpe/Sortino. Defaults suit daily price bars.
    periods_per_year: int = 252
    risk_free: float = 0.0


class FinanceMetrics(StrictModel):
    """Scalar risk/return summary over the whole series (annualized)."""
    total_return: float | None = None         # last/first - 1
    cagr: float | None = None                 # compound annual growth rate
    ann_return: float | None = None           # mean periodic return, annualized
    ann_volatility: float | None = None       # std of returns, annualized
    sharpe: float | None = None               # (ann_return - rf) / ann_vol
    sortino: float | None = None              # ann_return / annualized downside dev
    max_drawdown: float | None = None         # worst peak-to-trough, as a fraction
    calmar: float | None = None               # cagr / |max_drawdown|


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
    ema: list[float | None]       # exponential moving average (span=window)
    drawdown: list[float | None]  # running peak-to-trough of cum_return
    metrics: FinanceMetrics
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


# -- forecasting ------------------------------------------------------------

class ForecastRequest(StrictModel):
    path: str
    column: str                       # value column to forecast
    x: str | None = None              # time / order column (else row index)
    group: str | None = None          # forecast per group key (capped)
    horizon: int = 24                 # steps ahead
    model: str = "auto"               # auto | xgboost | gbr | ridge
    period: int | None = None         # seasonal period (Fourier features)
    agg: str = "mean"                 # collapse duplicate x: mean|sum|last|max|min
    filters: list[FilterSpec] = []
    points: int = 600                 # history downsample for display
    max_groups: int = 6


class ForecastSeries(StrictModel):
    key: str = ""                     # group value ("" when no group)
    history_x: list[Any]
    history_y: list[float | None]
    forecast_x: list[Any]
    forecast_y: list[float | None]
    lower: list[float | None]
    upper: list[float | None]
    rmse: float | None = None         # in-sample residual RMSE


class ForecastResult(StrictModel):
    node_id: str
    path: str
    column: str
    model_used: str
    horizon: int
    period: int | None = None
    series: list[ForecastSeries]
    source_rows: int
    sampled: bool = False
