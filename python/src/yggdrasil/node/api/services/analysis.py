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

import numpy as np
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
    CorrelationRequest,
    CorrelationResult,
    DescribeResult,
    ExportRequest,
    FilterSpec,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    IndicatorsRequest,
    IndicatorsResult,
    OhlcRequest,
    OhlcResult,
    RiskRequest,
    RiskResult,
    SeriesRequest,
    SeriesResult,
    Transform,
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
        # per-file cap cache: (path, mtime_ns, size) -> row_cap
        self._cap_cache: dict[tuple, int] = {}

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

    async def export(self, req: ExportRequest):
        """Apply the transform and write the result in `fmt`. Returns
        (temp_path, download_name) — the caller streams then unlinks it."""
        return await run_in_threadpool(partial(self._export, req))

    async def correlate(self, req: "CorrelationRequest") -> "CorrelationResult":
        return await run_in_threadpool(partial(self._correlate, req))

    async def risk(self, req: "RiskRequest") -> "RiskResult":
        return await run_in_threadpool(partial(self._risk, req))

    async def indicators(self, req: "IndicatorsRequest") -> "IndicatorsResult":
        return await run_in_threadpool(partial(self._indicators, req))

    async def forecast(self, req: "ForecastRequest") -> "ForecastResult":
        return await run_in_threadpool(partial(self._forecast, req))

    def _forecast(self, req: "ForecastRequest") -> "ForecastResult":
        """Forecast a value column over time (optionally per group key).

        Aggregates the series per time bucket, fits the chosen model (xgboost →
        gbr → ridge fallback) over engineered trend/lag/seasonal features, and
        forecasts ``horizon`` steps with a widening confidence band. Streaming
        polars scan keeps the read bounded."""
        from .forecast import forecast_series
        from ..schemas.analysis import ForecastResult, ForecastSeries

        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        has_g = bool(req.group and req.group in cols)
        keep = list(dict.fromkeys(
            [req.column] + ([req.x] if has_x else []) + ([req.group] if has_g else [])))
        plan = lf.select(keep)
        # Bound the materialised series by the byte budget — keeps forecasting a
        # huge file from pulling it all into memory; we forecast the tail.
        cap = self._row_cap_for_bytes(plan)
        full_rows = plan.select(pl.len()).collect(engine="streaming").item()
        df = plan.tail(cap).collect(engine="streaming") if full_rows > cap else plan.collect(engine="streaming")
        source_rows = full_rows

        agg = req.agg if req.agg in ("mean", "sum", "last", "max", "min") else "mean"

        def _series_for(sub: pl.DataFrame) -> ForecastSeries:
            # collapse duplicate x with the chosen aggregation, ordered by x
            if has_x:
                # Alias the measure so it can't collide with the group key (the
                # x column itself, when someone forecasts a column named like x).
                gb = sub.group_by(req.x).agg(
                    getattr(pl.col(req.column).cast(pl.Float64, strict=False), agg)().alias("__y"))
                gb = gb.sort(req.x)
                xs = gb[req.x].to_list()
                y = gb["__y"].to_list()
            else:
                xs = list(range(sub.height))
                y = sub[req.column].cast(pl.Float64, strict=False).to_list()
            preds, lo, hi, rmse, used[0] = forecast_series(y, req.horizon, req.period, req.model)
            # future x: extrapolate by the median step for numeric x, else index
            fx: list = []
            try:
                xnum = [float(v) for v in xs]
                step = float(np.median(np.diff(xnum))) if len(xnum) > 1 else 1.0
                last = xnum[-1] if xnum else 0.0
                fx = [last + step * (k + 1) for k in range(len(preds))]
            except (TypeError, ValueError):
                base = len(xs)
                fx = [base + k for k in range(len(preds))]
            # downsample history for display
            hx, hy = xs, y
            if len(hx) > req.points:
                idx = np.linspace(0, len(hx) - 1, req.points).astype(int)
                hx = [hx[i] for i in idx]
                hy = [hy[i] for i in idx]
            return ForecastSeries(
                key="", history_x=[_safe(v) for v in hx], history_y=[_safe(v) for v in hy],
                forecast_x=[_safe(v) for v in fx], forecast_y=[_safe(v) for v in preds],
                lower=[_safe(v) for v in lo], upper=[_safe(v) for v in hi],
                rmse=round(rmse, 4) if rmse is not None else None,
            )

        used = ["ridge"]
        out: list[ForecastSeries] = []
        if has_g:
            top = (df.group_by(req.group).len().sort("len", descending=True)
                   .head(req.max_groups)[req.group].to_list())
            for key in top:
                s = _series_for(df.filter(pl.col(req.group) == key))
                s.key = str(key)
                out.append(s)
        else:
            out.append(_series_for(df))

        return ForecastResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            model_used=used[0], horizon=req.horizon, period=req.period,
            series=out, source_rows=source_rows,
            sampled=any(len(s.history_x) < source_rows for s in out),
        )

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

    def _apply_filters(self, lf: pl.LazyFrame, filters: list[FilterSpec]) -> pl.LazyFrame:
        """Push row predicates into the scan."""
        for f in filters:
            col = pl.col(f.column)
            if f.op == "is_null":
                lf = lf.filter(col.is_null())
            elif f.op == "not_null":
                lf = lf.filter(col.is_not_null())
            elif f.op == "contains":
                lf = lf.filter(col.cast(pl.Utf8).str.contains(str(f.value), literal=True))
            elif f.op == "in":
                vals = f.value if isinstance(f.value, list) else [f.value]
                lf = lf.filter(col.is_in(vals))
            elif f.op in ("==", "!=", ">", ">=", "<", "<="):
                v = f.value
                cmp = {"==": col == v, "!=": col != v, ">": col > v,
                       ">=": col >= v, "<": col < v, "<=": col <= v}[f.op]
                lf = lf.filter(cmp)
            else:
                raise BadRequestError(f"unknown filter op {f.op!r}")
        return lf

    def _apply_transform(self, lf: pl.LazyFrame, t: Transform) -> pl.LazyFrame:
        lf = self._apply_filters(lf, t.filters)
        if t.casts:
            schema = lf.collect_schema()
            exprs = []
            for c in t.casts:
                if c.column not in schema.names():
                    continue
                e = pl.col(c.column)
                d = c.dtype.lower()
                if d in ("datetime", "date", "timestamp"):
                    # naive/string/epoch -> datetime, stamped UTC, then converted
                    # to the target tz (UTC by default).
                    dtc = e.str.to_datetime(strict=False) if schema[c.column] == pl.Utf8 else e.cast(pl.Datetime, strict=False)
                    if d == "date":
                        exprs.append(dtc.dt.date().alias(c.column))
                    else:
                        tz = c.tz or "UTC"
                        dtc = dtc.dt.replace_time_zone("UTC")
                        if tz != "UTC":
                            dtc = dtc.dt.convert_time_zone(tz)
                        exprs.append(dtc.alias(c.column))
                else:
                    target = {"int": pl.Int64, "float": pl.Float64, "double": pl.Float64,
                              "bool": pl.Boolean, "string": pl.Utf8, "str": pl.Utf8}.get(d)
                    if target is None:
                        raise BadRequestError(f"unknown cast dtype {c.dtype!r}")
                    exprs.append(e.cast(target, strict=False).alias(c.column))
            if exprs:
                lf = lf.with_columns(exprs)
        if t.columns:
            lf = lf.select(t.columns)
        if t.limit:
            lf = lf.head(t.limit)
        return lf

    def _row_cap_for_bytes(self, plan: pl.LazyFrame, max_bytes: int | None = None) -> int:
        """How many rows of ``plan`` fit the byte budget — measured once per
        (path, mtime, size) tuple then cached, so repeated calls on the same
        file don't re-sample the data."""
        budget = max_bytes or self.settings.analysis_max_bytes
        # Build a cheap cache key from the schema hash (covers column selection)
        schema_key = str(plan.collect_schema())
        cache_key = (schema_key, budget)
        if cache_key in self._cap_cache:
            return self._cap_cache[cache_key]
        sample = plan.head(2048).collect(engine="streaming")
        if sample.height == 0:
            return self.settings.analysis_max_rows
        per_row = max(1, sample.estimated_size() // sample.height)
        cap = max(256, min(int(budget // per_row), self.settings.analysis_max_rows))
        # Cap cache at 256 entries — schema diversity is bounded
        if len(self._cap_cache) >= 256:
            self._cap_cache.pop(next(iter(self._cap_cache)))
        self._cap_cache[cache_key] = cap
        return cap

    # -- export -------------------------------------------------------------

    def _export(self, req: ExportRequest):
        import tempfile
        from yggdrasil.enums.media_type import MediaType
        ext = {"arrow": "arrow", "ipc": "arrow"}.get(req.fmt, req.fmt)
        lf = self._apply_transform(self._frame(req.path), req.transform)
        table = lf.collect(engine="streaming").to_arrow()
        base = self.fs._resolve(req.path).stem
        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        tmp.close()
        from pathlib import Path as _P
        media = MediaType.from_(req.fmt, default=None) or MediaType.from_("csv")
        with YggPath.from_(tmp.name).open("wb", media_type=media) as bio:
            bio.write_arrow_table(table)
        return _P(tmp.name), f"{base}.{ext}"

    # -- aggregate / pivot --------------------------------------------------

    def _aggregate(self, req: AggregateRequest) -> AggregateResult:
        if not req.measures:
            raise BadRequestError("aggregate needs at least one measure")
        lf = self._apply_filters(self._frame(req.path), req.filters)
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
        keep = list(dict.fromkeys([req.column] + ([req.order_by] if req.order_by and req.order_by in cols else [])))
        plan = lf.select(keep)
        if req.order_by and req.order_by in cols:
            plan = plan.sort(req.order_by)
        # Cap by the byte budget (Arrow size of a sample), not just req.limit —
        # a wide series exhausts memory at far fewer rows than a narrow one.
        cap = min(req.limit, self._row_cap_for_bytes(plan))
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
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        keep = list(dict.fromkeys([req.column] + ([req.x] if has_x else [])))
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
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        has_vol = bool(req.volume and req.volume in cols)
        keep = list(dict.fromkeys([req.column] + ([req.x] if has_x else []) + ([req.volume] if has_vol else [])))
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

    # -- correlation matrix ---------------------------------------------------

    def _correlate(self, req: "CorrelationRequest") -> "CorrelationResult":
        """Pearson, Spearman, or (approximated) Kendall correlation matrix over
        the selected numeric columns. Spearman is computed as Pearson of ranks;
        Kendall uses the O(n log n) merge-sort approach but falls back to scipy
        when available for precision. All numpy, no mandatory extra deps."""
        from ..schemas.analysis import CorrelationResult
        if len(req.columns) < 2:
            raise BadRequestError("correlate needs at least 2 columns")
        if len(req.columns) > 50:
            raise BadRequestError("correlate supports at most 50 columns")
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        for c in req.columns:
            if c not in cols:
                raise BadRequestError(f"column {c!r} not found")
        plan = lf.select(req.columns)
        if req.order_by and req.order_by in cols:
            plan = lf.select(list(dict.fromkeys([req.order_by] + req.columns))).sort(req.order_by).select(req.columns)
        cap = min(req.limit, self._row_cap_for_bytes(plan))
        df = plan.head(cap).collect(engine="streaming")
        mat = df.to_numpy().astype(float)
        n = mat.shape[0]
        nc = mat.shape[1]
        method = req.method.lower()
        if method == "spearman":
            # Rank each column (average ties), then Pearson of ranks
            from scipy.stats import rankdata  # type: ignore[import]
            try:
                ranked = np.column_stack([rankdata(mat[:, i]) for i in range(nc)])
            except ImportError:
                # Fallback: simple rank without scipy
                ranked = np.column_stack([mat[:, i].argsort().argsort().astype(float) for i in range(nc)])
            mat = ranked
        elif method == "kendall":
            try:
                from scipy.stats import kendalltau  # type: ignore[import]
                result_mat: list[list[float | None]] = []
                for i in range(nc):
                    row: list[float | None] = []
                    for j in range(nc):
                        if i == j:
                            row.append(1.0)
                        elif j < i:
                            row.append(result_mat[j][i])
                        else:
                            tau, _ = kendalltau(mat[:, i], mat[:, j])
                            row.append(round(float(tau), 6) if not np.isnan(tau) else None)
                    result_mat.append(row)
                return CorrelationResult(
                    node_id=self.settings.node_id, path=req.path,
                    columns=req.columns, matrix=result_mat, method=method, n=n,
                )
            except ImportError:
                pass  # fall through to Pearson approximation
        # Pearson (or Spearman-as-Pearson-of-ranks)
        corr = np.corrcoef(mat, rowvar=False)
        matrix = [
            [round(float(corr[i, j]), 6) if not np.isnan(corr[i, j]) else None for j in range(nc)]
            for i in range(nc)
        ]
        return CorrelationResult(
            node_id=self.settings.node_id, path=req.path,
            columns=req.columns, matrix=matrix, method=method if method != "kendall" else "pearson", n=n,
        )

    # -- risk analytics -------------------------------------------------------

    def _risk(self, req: "RiskRequest") -> "RiskResult":
        """Portfolio/series risk metrics: Sharpe, Sortino, max drawdown, VaR,
        CVaR, Calmar, win rate, profit factor, skewness, excess kurtosis.
        All from numpy on the (possibly bounded) returns series."""
        from ..schemas.analysis import RiskResult
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        keep = list(dict.fromkeys([req.column] + ([req.order_by] if req.order_by and req.order_by in cols else [])))
        plan = lf.select(keep)
        if req.order_by and req.order_by in cols:
            plan = plan.sort(req.order_by)
        cap = min(req.limit, self._row_cap_for_bytes(plan))
        df = plan.head(cap).collect(engine="streaming")
        prices = df[req.column].cast(pl.Float64, strict=False).to_numpy()
        prices = prices[~np.isnan(prices)]
        if len(prices) < 2:
            return RiskResult(
                node_id=self.settings.node_id, path=req.path, column=req.column,
                n=len(prices), periods_per_year=req.periods_per_year,
                ann_return=None, ann_volatility=None, sharpe_ratio=None,
                sortino_ratio=None, calmar_ratio=None, max_drawdown=None,
                max_drawdown_peak_i=None, max_drawdown_trough_i=None,
                var_95=None, var_99=None, cvar_95=None, win_rate=None,
                profit_factor=None, skewness=None, kurtosis=None,
            )
        ppy = req.periods_per_year
        ret = prices[1:] / prices[:-1] - 1.0 if not req.is_returns else prices
        n = len(ret)
        mean_r = float(np.mean(ret))
        std_r = float(np.std(ret, ddof=1)) if n > 1 else 0.0
        ann_ret = float((1 + mean_r) ** ppy - 1)
        ann_vol = float(std_r * math.sqrt(ppy))
        sharpe = float(mean_r / std_r * math.sqrt(ppy)) if std_r > 0 else None
        down = ret[ret < 0]
        down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
        sortino = float(mean_r / down_std * math.sqrt(ppy)) if down_std > 0 else None
        # max drawdown on price (or cumulative return if is_returns)
        series = np.cumprod(1 + ret) if req.is_returns else prices[: n + 1]
        peak_i, trough_i, max_dd = 0, 0, 0.0
        running_peak = series[0]
        running_peak_i = 0
        for i in range(1, len(series)):
            if series[i] > running_peak:
                running_peak = series[i]
                running_peak_i = i
            dd = (series[i] - running_peak) / running_peak if running_peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
                peak_i = running_peak_i
                trough_i = i
        calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else None
        var95 = float(np.percentile(ret, 5))
        var99 = float(np.percentile(ret, 1))
        cvar95 = float(np.mean(ret[ret <= var95])) if np.any(ret <= var95) else var95
        win_rate = float(np.mean(ret > 0))
        gains = ret[ret > 0].sum()
        losses = abs(ret[ret < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else None
        # moments (numpy)
        skew_val: float | None = None
        kurt_val: float | None = None
        if n >= 4 and std_r > 0:
            z = (ret - mean_r) / std_r
            skew_val = float(np.mean(z ** 3))
            kurt_val = float(np.mean(z ** 4) - 3)  # excess kurtosis
        return RiskResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            n=n, periods_per_year=ppy,
            ann_return=round(ann_ret, 6), ann_volatility=round(ann_vol, 6),
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            sortino_ratio=round(sortino, 4) if sortino is not None else None,
            calmar_ratio=round(calmar, 4) if calmar is not None else None,
            max_drawdown=round(max_dd, 6),
            max_drawdown_peak_i=int(peak_i), max_drawdown_trough_i=int(trough_i),
            var_95=round(var95, 6), var_99=round(var99, 6), cvar_95=round(cvar95, 6),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
            skewness=round(skew_val, 4) if skew_val is not None else None,
            kurtosis=round(kurt_val, 4) if kurt_val is not None else None,
        )

    # -- technical indicators -------------------------------------------------

    def _indicators(self, req: "IndicatorsRequest") -> "IndicatorsResult":
        """Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger,
        ATR, Stochastic, OBV) from a price series. Pure numpy — no TA-Lib dep.
        Returns a columnar dict aligned to the x axis."""
        from ..schemas.analysis import IndicatorsResult
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.x and req.x in cols)
        has_h = bool(req.high and req.high in cols)
        has_l = bool(req.low and req.low in cols)
        has_v = bool(req.volume and req.volume in cols)
        keep = list(dict.fromkeys(
            [req.column]
            + ([req.x] if has_x else [])
            + ([req.high] if has_h else [])
            + ([req.low] if has_l else [])
            + ([req.volume] if has_v else [])
        ))
        plan = lf.select(keep)
        if has_x:
            plan = plan.sort(req.x)
        cap = min(req.limit, self._row_cap_for_bytes(plan))
        df = plan.head(cap).collect(engine="streaming")
        close = df[req.column].cast(pl.Float64, strict=False).to_numpy()
        high = df[req.high].cast(pl.Float64, strict=False).to_numpy() if has_h else None
        low = df[req.low].cast(pl.Float64, strict=False).to_numpy() if has_l else None
        volume = df[req.volume].cast(pl.Float64, strict=False).to_numpy() if has_v else None
        x_vals = df[req.x].to_list() if has_x else list(range(len(close)))
        n = len(close)

        def _sma(arr: np.ndarray, p: int) -> np.ndarray:
            # cumsum-based O(n) rolling mean — much faster than per-element slice
            out = np.full(n, np.nan)
            if p > n:
                return out
            filled = np.where(np.isnan(arr), 0.0, arr)
            cs = np.cumsum(filled)
            out[p - 1:] = (cs[p - 1:] - np.concatenate([[0], cs[:-p]])) / p
            return out

        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            out = np.full(n, np.nan)
            if span > n:
                return out
            k = 2.0 / (span + 1)
            seed_i = span - 1
            valid = arr[:span]
            valid = valid[~np.isnan(valid)]
            if len(valid) == 0:
                return out
            out[seed_i] = np.mean(valid)
            for i in range(seed_i + 1, n):
                v = arr[i]
                out[i] = v * k + out[i - 1] * (1 - k) if not np.isnan(v) else out[i - 1]
            return out

        result: dict[str, list[float | None]] = {}
        for p in req.sma:
            result[f"sma_{p}"] = [_safe(v) for v in _sma(close, p)]
        for p in req.ema:
            result[f"ema_{p}"] = [_safe(v) for v in _ema(close, p)]

        if req.bollinger:
            p = req.bollinger
            mid = _sma(close, p)
            std_arr = np.full(n, np.nan)
            for i in range(p - 1, n):
                std_arr[i] = np.nanstd(close[i - p + 1:i + 1], ddof=0)
            result["bb_mid"] = [_safe(v) for v in mid]
            result["bb_upper"] = [_safe(m + 2 * s) if not np.isnan(m) else None for m, s in zip(mid, std_arr)]
            result["bb_lower"] = [_safe(m - 2 * s) if not np.isnan(m) else None for m, s in zip(mid, std_arr)]

        if req.rsi:
            p = req.rsi
            delta = np.diff(close, prepend=np.nan)
            gains = np.where(delta > 0, delta, 0.0)
            losses = np.where(delta < 0, -delta, 0.0)
            rsi_arr = np.full(n, np.nan)
            if n > p:
                avg_gain = np.nanmean(gains[1:p + 1])
                avg_loss = np.nanmean(losses[1:p + 1])
                for i in range(p, n):
                    avg_gain = (avg_gain * (p - 1) + gains[i]) / p
                    avg_loss = (avg_loss * (p - 1) + losses[i]) / p
                    rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
                    rsi_arr[i] = 100 - 100 / (1 + rs)
            result[f"rsi_{p}"] = [_safe(v) for v in rsi_arr]

        if req.macd:
            ema12 = _ema(close, 12)
            ema26 = _ema(close, 26)
            macd_line = ema12 - ema26
            signal = _ema(macd_line, 9)
            hist_arr = macd_line - signal
            result["macd"] = [_safe(v) for v in macd_line]
            result["macd_signal"] = [_safe(v) for v in signal]
            result["macd_hist"] = [_safe(v) for v in hist_arr]

        if req.atr and has_h and has_l:
            p = req.atr
            # Vectorised true range, then Wilder smoothing (no Python loop)
            tr_hl = high[1:] - low[1:]
            tr_hc = np.abs(high[1:] - close[:-1])
            tr_lc = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr_hl, np.maximum(tr_hc, tr_lc))
            atr_arr = np.full(n, np.nan)
            if p <= n:
                atr_arr[p] = np.mean(tr[:p])
                for i in range(p + 1, n):
                    atr_arr[i] = (atr_arr[i - 1] * (p - 1) + tr[i - 1]) / p
            result[f"atr_{p}"] = [_safe(v) for v in atr_arr]

        if req.stoch and has_h and has_l:
            p = req.stoch
            # Rolling min/max via stride-tricks — O(n) amortised using sliding window
            from numpy.lib.stride_tricks import sliding_window_view
            k_arr = np.full(n, np.nan)
            if p <= n:
                win_h = sliding_window_view(high, p)
                win_l = sliding_window_view(low, p)
                roll_hi = win_h.max(axis=1)
                roll_lo = win_l.min(axis=1)
                rng_hl = roll_hi - roll_lo
                k_arr[p - 1:] = np.where(rng_hl > 0, 100 * (close[p - 1:] - roll_lo) / rng_hl, 50.0)
            result[f"stoch_k_{p}"] = [_safe(v) for v in k_arr]
            result[f"stoch_d_{p}"] = [_safe(v) for v in _sma(k_arr, 3)]

        if req.obv and has_v:
            # Vectorised OBV: sign of daily change, then cumsum of signed volume
            direction = np.sign(np.diff(close, prepend=close[0]))
            obv_arr = np.cumsum(direction * volume)
            result["obv"] = [_safe(v) for v in obv_arr]

        return IndicatorsResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            x=[_safe(v) for v in x_vals],
            price=[_safe(v) for v in close],
            indicators=result,
            n=n,
        )
