"""Tabular analytics over the LazyTabular layer — pivot/aggregate, summary
stats, and finance series (returns, cumulative return, rolling volatility).

Reads go through the same ``YggPath.open().read_arrow_table()`` path as the
rest; the Arrow table is handed to polars zero-copy (``pl.from_arrow``) and the
vectorized group-by / rolling kernels do the work. Reads are bounded by
``analysis_max_rows`` so a huge file is sampled, not exploded into memory.
"""
from __future__ import annotations

import math
from functools import partial

import polars as pl
import polars.selectors as cs
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
)
from .fs import FsService

_AGGS = {"sum", "mean", "min", "max", "count", "median", "std", "var"}


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

    # -- helpers ------------------------------------------------------------

    def _load_df(self, path: str, cap: int | None = None) -> tuple[pl.DataFrame, int, bool]:
        resolved = self.fs._resolve(path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Not a file: {path!r}")
        cap = cap or self.settings.analysis_max_rows
        try:
            with YggPath.from_(str(resolved)).open("rb") as bio:
                table = bio.read_arrow_table(options=CastOptions(row_limit=cap + 1))
        except Exception as exc:
            raise BadRequestError(f"Cannot read {path!r} as a table: {exc}")
        truncated = table.num_rows > cap
        if truncated:
            table = table.slice(0, cap)
        df = pl.from_arrow(table)
        if isinstance(df, pl.Series):
            df = df.to_frame()
        return df, table.num_rows, truncated

    # -- aggregate / pivot --------------------------------------------------

    def _aggregate(self, req: AggregateRequest) -> AggregateResult:
        if not req.measures:
            raise BadRequestError("aggregate needs at least one measure")
        df, source_rows, truncated = self._load_df(req.path)
        cols = set(df.columns)
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

        out = df.group_by(req.group_by).agg(exprs) if req.group_by else df.select(exprs)
        group_count = out.height
        first = f"{req.measures[0].column}_{req.measures[0].agg}"
        if first in out.columns and out.height:
            out = out.sort(first, descending=req.sort_desc, nulls_last=True)
        out = out.head(req.limit)
        return AggregateResult(
            node_id=self.settings.node_id,
            path=req.path,
            columns=out.columns,
            rows=[[_safe(v) for v in row] for row in out.iter_rows()],
            group_count=group_count,
            source_rows=source_rows,
            truncated=truncated,
        )

    # -- describe -----------------------------------------------------------

    def _describe(self, path: str) -> DescribeResult:
        df, source_rows, truncated = self._load_df(path)
        num = df.select(cs.numeric())
        if num.width == 0:
            return DescribeResult(
                node_id=self.settings.node_id, path=path, statistics=[], columns=[],
                rows=[], source_rows=source_rows, truncated=truncated,
            )
        d = num.describe()  # 'statistic' col + one col per numeric field
        stats = d["statistic"].to_list()
        columns = [c for c in d.columns if c != "statistic"]
        rows = [[_safe(d[c][i]) for c in columns] for i in range(d.height)]
        return DescribeResult(
            node_id=self.settings.node_id, path=path,
            statistics=stats, columns=columns, rows=rows,
            source_rows=source_rows, truncated=truncated,
        )

    # -- finance series -----------------------------------------------------

    def _finance(self, req: FinanceRequest) -> FinanceResult:
        df, _source_rows, truncated = self._load_df(req.path, cap=req.limit)
        if req.column not in df.columns:
            raise BadRequestError(f"column {req.column!r} not found")
        if req.order_by and req.order_by in df.columns:
            df = df.sort(req.order_by)
        df = df.head(req.limit)

        val = df[req.column].cast(pl.Float64, strict=False)
        ret = val / val.shift(1) - 1.0
        # cumulative return: product of (1+ret), first step neutral
        growth = (1.0 + ret).fill_null(1.0)
        cum = growth.cum_prod() - 1.0
        window = max(2, req.window)
        roll_mean = val.rolling_mean(window_size=window)
        roll_vol = ret.rolling_std(window_size=window)

        index = (
            df[req.order_by].to_list() if (req.order_by and req.order_by in df.columns)
            else list(range(df.height))
        )
        return FinanceResult(
            node_id=self.settings.node_id, path=req.path, column=req.column, window=window,
            index=[_safe(v) for v in index],
            value=_safe_list(val), pct_change=_safe_list(ret), cum_return=_safe_list(cum),
            roll_mean=_safe_list(roll_mean), roll_vol=_safe_list(roll_vol),
            truncated=truncated,
        )
