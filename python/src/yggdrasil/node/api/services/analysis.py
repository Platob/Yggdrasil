"""Tabular analytics over the LazyTabular layer — pivot/aggregate, summary
stats, finance series, adaptive downsampling, and OHLC resampling.

Everything runs through polars **lazy** scans: ``pl.scan_parquet`` / ``scan_csv``
/ ``scan_ndjson`` give projection + predicate + slice pushdown and stream the
result with bounded memory, so a wide/huge file only ever reads the columns and
rows a query touches. Reductions (aggregate, describe, ohlc, downsample) stream
the *whole* file correctly; the only row-bounded path is finance (it needs the
ordered series in memory for rolling windows). Formats polars can't scan
(json/arrow/xlsx) fall back to a single bounded Arrow read, then ``.lazy()``.
"""
from __future__ import annotations

import math
from functools import partial

import polars as pl
from fastapi.concurrency import run_in_threadpool

from yggdrasil.data.options import CastOptions
from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.path import Path as YggPath

from ...config import Settings
from ...exceptions import ForbiddenError, NotFoundError
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    DescribeResult,
    FinanceRequest,
    FinanceResult,
    OhlcRequest,
    OhlcResult,
    SeriesRequest,
    SeriesResult,
)
from .fs import FsService

_AGGS = {"sum", "mean", "min", "max", "count", "median", "std", "var"}
_STATS = ["count", "null_count", "mean", "std", "min", "25%", "50%", "75%", "max"]


def _safe(v):
    """JSON-safe scalar: drop NaN/inf, stringify anything exotic."""
    if v is None:
        return None
    if isinstance(v, bool) or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return str(v)


def _safe_list(series: pl.Series) -> list:
    return [_safe(v) for v in series.to_list()]


class AnalysisService:
    def __init__(self, settings: Settings, *, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        return await run_in_threadpool(partial(self._aggregate, req))

    async def describe(self, path: str) -> DescribeResult:
        return await run_in_threadpool(partial(self._describe, path))

    async def finance(self, req: FinanceRequest) -> FinanceResult:
        return await run_in_threadpool(partial(self._finance, req))

    async def series(self, req: SeriesRequest) -> SeriesResult:
        return await run_in_threadpool(partial(self._series, req))

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        return await run_in_threadpool(partial(self._ohlc, req))

    # -- lazy scan ----------------------------------------------------------

    def _frame(self, path: str) -> pl.LazyFrame:
        """A LazyFrame over the file. parquet/csv/ndjson scan lazily (push-down
        + streaming); other formats read once through Arrow, then ``.lazy()``."""
        resolved = self.fs._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Not a file: {path!r}")
        ext = resolved.suffix.lstrip(".").lower()
        try:
            if ext in ("parquet", "pq"):
                return pl.scan_parquet(str(resolved))
            if ext == "csv":
                return pl.scan_csv(str(resolved))
            if ext == "ndjson":
                return pl.scan_ndjson(str(resolved))
            with YggPath.from_(str(resolved)).open("rb") as bio:
                table = bio.read_arrow_table(options=CastOptions(row_limit=self.settings.analysis_max_rows + 1))
            return pl.from_arrow(table).lazy()
        except Exception as exc:
            raise BadRequestError(f"Cannot read {path!r} as a table: {exc}")

    # -- aggregate / pivot --------------------------------------------------

    def _aggregate(self, req: AggregateRequest) -> AggregateResult:
        if not req.measures:
            raise BadRequestError("aggregate needs at least one measure")
        lf = self._frame(req.path)
        cols = set(lf.collect_schema().names())
        for g in req.group_by:
            if g not in cols:
                raise BadRequestError(f"group_by column {g!r} not found")
        exprs = []
        for m in req.measures:
            if m.agg not in _AGGS:
                raise BadRequestError(f"unknown agg {m.agg!r}; one of {sorted(_AGGS)}")
            if m.column not in cols:
                raise BadRequestError(f"measure column {m.column!r} not found")
            exprs.append(getattr(pl.col(m.column), m.agg)().alias(f"{m.column}_{m.agg}"))

        # Project to just the touched columns, then stream the group-by over the
        # whole file (correct totals, memory bounded by the group count).
        keep = list(dict.fromkeys(req.group_by + [m.column for m in req.measures]))
        plan = lf.select(keep)
        out = (plan.group_by(req.group_by).agg(exprs) if req.group_by else plan.select(exprs))
        out = out.collect(engine="streaming")
        group_count = out.height
        first = f"{req.measures[0].column}_{req.measures[0].agg}"
        if first in out.columns and out.height:
            out = out.sort(first, descending=req.sort_desc, nulls_last=True)
        out = out.head(req.limit)
        source_rows = lf.select(pl.len()).collect(engine="streaming").item()
        return AggregateResult(
            node_id=self.settings.node_id, path=req.path,
            columns=out.columns,
            rows=[[_safe(v) for v in row] for row in out.iter_rows()],
            group_count=group_count, source_rows=source_rows, truncated=False,
        )

    # -- describe -----------------------------------------------------------

    def _describe(self, path: str) -> DescribeResult:
        lf = self._frame(path)
        schema = lf.collect_schema()
        num = [n for n, dt in schema.items() if dt.is_numeric()]
        if not num:
            return DescribeResult(
                node_id=self.settings.node_id, path=path, statistics=[], columns=[],
                rows=[], source_rows=0, truncated=False,
            )
        # One streaming pass computes every statistic for every numeric column.
        exprs = []
        for c in num:
            col = pl.col(c)
            exprs += [
                col.count().alias(f"{c}|count"), col.null_count().alias(f"{c}|null_count"),
                col.mean().alias(f"{c}|mean"), col.std().alias(f"{c}|std"),
                col.min().alias(f"{c}|min"), col.quantile(0.25).alias(f"{c}|25%"),
                col.quantile(0.50).alias(f"{c}|50%"), col.quantile(0.75).alias(f"{c}|75%"),
                col.max().alias(f"{c}|max"),
            ]
        row = lf.select(num).select(exprs).collect(engine="streaming")
        rows = [[_safe(row[f"{c}|{stat}"][0]) for c in num] for stat in _STATS]
        source_rows = lf.select(pl.len()).collect(engine="streaming").item()
        return DescribeResult(
            node_id=self.settings.node_id, path=path,
            statistics=_STATS, columns=num, rows=rows,
            source_rows=source_rows, truncated=False,
        )

    # -- finance series -----------------------------------------------------

    def _finance(self, req: FinanceRequest) -> FinanceResult:
        lf = self._frame(req.path)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        keep = [req.column] + ([req.order_by] if req.order_by and req.order_by in cols else [])
        plan = lf.select(keep)
        if req.order_by and req.order_by in cols:
            plan = plan.sort(req.order_by)
        cap = req.limit
        df = plan.head(cap + 1).collect(engine="streaming")
        truncated = df.height > cap
        if truncated:
            df = df.head(cap)

        val = df[req.column].cast(pl.Float64, strict=False)
        ret = val / val.shift(1) - 1.0
        cum = (1.0 + ret).fill_null(1.0).cum_prod() - 1.0
        window = max(2, req.window)
        roll_mean = val.rolling_mean(window_size=window)
        roll_vol = ret.rolling_std(window_size=window)
        index = (
            df[req.order_by].to_list() if (req.order_by and req.order_by in cols)
            else list(range(df.height))
        )
        return FinanceResult(
            node_id=self.settings.node_id, path=req.path, column=req.column, window=window,
            index=[_safe(v) for v in index],
            value=_safe_list(val), pct_change=_safe_list(ret), cum_return=_safe_list(cum),
            roll_mean=_safe_list(roll_mean), roll_vol=_safe_list(roll_vol),
            truncated=truncated,
        )

    # -- adaptive downsample ------------------------------------------------

    def _series(self, req: SeriesRequest) -> SeriesResult:
        lf = self._frame(req.path)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        keep = [req.column] + ([req.x] if has_x else [])
        plan = lf.select(keep)
        # Zoom window — predicate pushed into the scan, so a zoomed-in view reads
        # only the rows in range.
        if has_x and req.x_min is not None:
            plan = plan.filter(pl.col(req.x) >= req.x_min)
        if has_x and req.x_max is not None:
            plan = plan.filter(pl.col(req.x) <= req.x_max)
        if has_x:
            plan = plan.sort(req.x)
        plan = plan.with_row_index("__i")

        source_rows = plan.select(pl.len()).collect(engine="streaming").item()
        points = max(16, min(req.points, 5000))
        if source_rows <= points:
            df = plan.collect(engine="streaming")
            x = df[req.x] if has_x else df["__i"]
            y = df[req.column].cast(pl.Float64, strict=False)
            return SeriesResult(
                node_id=self.settings.node_id, path=req.path, column=req.column,
                x=[_safe(v) for v in x.to_list()], y=_safe_list(y),
                y_min=_safe_list(y), y_max=_safe_list(y),
                source_rows=source_rows, sampled=False,
            )
        # Bucket into `points` groups by row position; per bucket keep the mean
        # (line) + min/max (envelope, so spikes survive the downsample).
        size = max(1, math.ceil(source_rows / points))
        bucket = (pl.col("__i") // size).alias("__b")
        xexpr = pl.col(req.x).first().alias("x") if has_x else pl.col("__i").first().alias("x")
        out = (
            plan.with_columns(bucket)
            .group_by("__b")
            .agg([
                xexpr,
                pl.col(req.column).cast(pl.Float64, strict=False).mean().alias("y"),
                pl.col(req.column).cast(pl.Float64, strict=False).min().alias("ymin"),
                pl.col(req.column).cast(pl.Float64, strict=False).max().alias("ymax"),
            ])
            .sort("__b")
            .collect(engine="streaming")
        )
        return SeriesResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            x=[_safe(v) for v in out["x"].to_list()],
            y=_safe_list(out["y"]), y_min=_safe_list(out["ymin"]), y_max=_safe_list(out["ymax"]),
            source_rows=source_rows, sampled=True,
        )

    # -- OHLC resampling ----------------------------------------------------

    def _ohlc(self, req: OhlcRequest) -> OhlcResult:
        lf = self._frame(req.path)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        has_vol = bool(req.volume and req.volume in cols)
        keep = [req.column] + ([req.x] if has_x else []) + ([req.volume] if has_vol else [])
        plan = lf.select(keep)
        if has_x:
            plan = plan.sort(req.x)
        plan = plan.with_row_index("__i")

        source_rows = plan.select(pl.len()).collect(engine="streaming").item()
        buckets = max(2, min(req.buckets, 2000))
        size = max(1, math.ceil(source_rows / buckets))
        price = pl.col(req.column).cast(pl.Float64, strict=False)
        aggs = [
            (pl.col(req.x).first().alias("x") if has_x else pl.col("__i").first().alias("x")),
            price.first().alias("open"), price.max().alias("high"),
            price.min().alias("low"), price.last().alias("close"),
        ]
        if has_vol:
            aggs.append(pl.col(req.volume).cast(pl.Float64, strict=False).sum().alias("volume"))
        out = (
            plan.with_columns((pl.col("__i") // size).alias("__b"))
            .group_by("__b").agg(aggs).sort("__b").collect(engine="streaming")
        )
        return OhlcResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            x=[_safe(v) for v in out["x"].to_list()],
            open=_safe_list(out["open"]), high=_safe_list(out["high"]),
            low=_safe_list(out["low"]), close=_safe_list(out["close"]),
            volume=_safe_list(out["volume"]) if has_vol else None,
            bars=out.height, source_rows=source_rows,
        )
