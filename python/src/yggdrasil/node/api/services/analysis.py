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
    FinanceMetrics,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    OhlcRequest,
    OhlcResult,
    SeriesRequest,
    SeriesResult,
    TechnicalRequest,
    TechnicalResult,
    TechnicalSeries,
    TechnicalSignal,
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

    async def forecast(self, req: "ForecastRequest") -> "ForecastResult":
        return await run_in_threadpool(partial(self._forecast, req))

    async def technical(self, req: "TechnicalRequest") -> "TechnicalResult":
        return await run_in_threadpool(partial(self._technical, req))

    async def correlation(self, req: "CorrelationRequest") -> "CorrelationResult":
        return await run_in_threadpool(partial(self._correlation, req))

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

    # -- technical indicators -----------------------------------------------

    def _technical(self, req: "TechnicalRequest") -> "TechnicalResult":
        """Compute RSI, MACD, Bollinger Bands (+ ATR if high/low available)."""
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")

        keep = [req.column]
        if req.x and req.x in cols:
            keep = [req.x] + keep
        for extra in (req.high, req.low, req.volume):
            if extra and extra in cols:
                keep.append(extra)
        keep = list(dict.fromkeys(keep))

        cap = min(req.limit, self.settings.analysis_max_rows)
        df = lf.select(keep).tail(cap).collect(engine="streaming")
        source_rows = lf.select(pl.len()).collect(engine="streaming").item()
        truncated = source_rows > cap

        prices = df[req.column].cast(pl.Float64, strict=False).to_list()
        n = len(prices)
        index: list = df[req.x].to_list() if (req.x and req.x in df.columns) else list(range(n))

        def _ema_list(vals: list[float | None], period: int) -> list[float | None]:
            k = 2.0 / (period + 1)
            out: list[float | None] = [None] * len(vals)
            warm = 0
            prev: float | None = None
            for i, v in enumerate(vals):
                if v is None:
                    continue
                if prev is None:
                    prev = v
                    warm = 1
                else:
                    warm += 1
                    prev = v * k + prev * (1 - k)
                if warm >= period:
                    out[i] = prev
            return out

        def _sma_list(vals: list[float | None], period: int) -> list[float | None]:
            out: list[float | None] = [None] * len(vals)
            buf: list[float] = []
            for i, v in enumerate(vals):
                if v is not None:
                    buf.append(v)
                if len(buf) >= period:
                    out[i] = sum(buf[-period:]) / period
            return out

        # RSI
        rsi: list[float | None] = [None] * n
        gains, losses = [], []
        prev_p: float | None = None
        for i, p in enumerate(prices):
            if p is None:
                continue
            if prev_p is not None:
                d = p - prev_p
                gains.append(max(d, 0.0))
                losses.append(max(-d, 0.0))
                if len(gains) >= req.rsi_period:
                    ag = sum(gains[-req.rsi_period:]) / req.rsi_period
                    al = sum(losses[-req.rsi_period:]) / req.rsi_period
                    rsi[i] = 100.0 - 100.0 / (1.0 + ag / al) if al != 0 else 100.0
            prev_p = p

        # MACD
        ema_fast = _ema_list(prices, req.macd_fast)
        ema_slow = _ema_list(prices, req.macd_slow)
        macd_line = [
            (f - s) if (f is not None and s is not None) else None
            for f, s in zip(ema_fast, ema_slow)
        ]
        macd_signal_line = _ema_list(macd_line, req.macd_signal)
        macd_hist = [
            (m - s) if (m is not None and s is not None) else None
            for m, s in zip(macd_line, macd_signal_line)
        ]

        # Bollinger Bands
        sma = _sma_list(prices, req.bb_period)
        bb_std_list: list[float | None] = [None] * n
        for i, p in enumerate(prices):
            if p is None or sma[i] is None:
                continue
            start = max(0, i - req.bb_period + 1)
            window_vals = [prices[j] for j in range(start, i + 1) if prices[j] is not None]
            if len(window_vals) >= req.bb_period:
                mean = sum(window_vals) / len(window_vals)
                variance = sum((v - mean) ** 2 for v in window_vals) / len(window_vals)
                bb_std_list[i] = variance ** 0.5

        bb_upper = [
            (sma[i] + req.bb_std * bb_std_list[i])
            if (sma[i] is not None and bb_std_list[i] is not None) else None
            for i in range(n)
        ]
        bb_lower = [
            (sma[i] - req.bb_std * bb_std_list[i])
            if (sma[i] is not None and bb_std_list[i] is not None) else None
            for i in range(n)
        ]

        # ATR (optional — needs high + low)
        atr: list[float | None] = [None] * n
        has_hl = req.high and req.high in df.columns and req.low and req.low in df.columns
        if has_hl:
            highs = df[req.high].cast(pl.Float64, strict=False).to_list()
            lows = df[req.low].cast(pl.Float64, strict=False).to_list()
            tr_list: list[float | None] = [None] * n
            for i in range(1, n):
                h, l, cp = highs[i], lows[i], prices[i - 1]
                if h is None or l is None or cp is None:
                    continue
                tr_list[i] = max(h - l, abs(h - cp), abs(l - cp))
            atr_raw = _sma_list(tr_list, req.atr_period)
            atr = atr_raw

        # Signal detection
        signals: list[TechnicalSignal] = []
        for i in range(1, n):
            xv = index[i]
            # RSI oversold/overbought
            if rsi[i] is not None:
                if rsi[i] < 30 and (rsi[i - 1] is None or rsi[i - 1] >= 30):
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="rsi_oversold", value=round(rsi[i], 2)))
                elif rsi[i] > 70 and (rsi[i - 1] is None or rsi[i - 1] <= 70):
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="rsi_overbought", value=round(rsi[i], 2)))
            # MACD crossover
            if (macd_line[i] is not None and macd_signal_line[i] is not None and
                    macd_line[i - 1] is not None and macd_signal_line[i - 1] is not None):
                was_below = macd_line[i - 1] < macd_signal_line[i - 1]
                now_above = macd_line[i] >= macd_signal_line[i]
                if was_below and now_above:
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="macd_cross_up",
                                                   value=round(macd_line[i], 4)))
                elif not was_below and not now_above:
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="macd_cross_down",
                                                   value=round(macd_line[i], 4)))
            # Bollinger breakout
            p = prices[i]
            if p is not None:
                if bb_upper[i] is not None and p > bb_upper[i]:
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="bb_breakout_up",
                                                   value=round(p, 4)))
                elif bb_lower[i] is not None and p < bb_lower[i]:
                    signals.append(TechnicalSignal(idx=i, x_val=xv, kind="bb_breakout_down",
                                                   value=round(p, 4)))

        indicators = [
            TechnicalSeries(name="rsi", values=[_safe(v) for v in rsi]),
            TechnicalSeries(name="macd", values=[_safe(v) for v in macd_line]),
            TechnicalSeries(name="macd_signal", values=[_safe(v) for v in macd_signal_line]),
            TechnicalSeries(name="macd_hist", values=[_safe(v) for v in macd_hist]),
            TechnicalSeries(name="bb_upper", values=[_safe(v) for v in bb_upper]),
            TechnicalSeries(name="bb_middle", values=[_safe(v) for v in sma]),
            TechnicalSeries(name="bb_lower", values=[_safe(v) for v in bb_lower]),
        ]
        if has_hl:
            indicators.append(TechnicalSeries(name="atr", values=[_safe(v) for v in atr]))

        return TechnicalResult(
            node_id=self.settings.node_id,
            path=req.path, column=req.column,
            index=[_safe(v) for v in index],
            price=[_safe(v) for v in prices],
            indicators=indicators,
            signals=signals,
            truncated=truncated,
        )

    def _correlation(self, req: "CorrelationRequest") -> "CorrelationResult":
        """Compute a pairwise correlation matrix across numeric columns."""
        lf = self._apply_filters(self._frame(req.path), req.filters)
        schema = lf.collect_schema()
        all_num = [
            n for n, t in schema.items()
            if t in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        ]
        cols = [c for c in req.columns if c in all_num] if req.columns else all_num[:20]
        if not cols:
            raise BadRequestError("No numeric columns found for correlation")

        cap = min(req.limit, self.settings.analysis_max_rows)
        df = lf.select(cols).tail(cap).collect(engine="streaming")
        source_rows = lf.select(pl.len()).collect(engine="streaming").item()

        method = req.method if req.method in ("pearson", "spearman") else "pearson"
        try:
            mat: list[list[float | None]] = []
            for c1 in cols:
                row: list[float | None] = []
                for c2 in cols:
                    try:
                        if method == "spearman":
                            s1 = df[c1].cast(pl.Float64, strict=False).rank()
                            s2 = df[c2].cast(pl.Float64, strict=False).rank()
                        else:
                            s1 = df[c1].cast(pl.Float64, strict=False)
                            s2 = df[c2].cast(pl.Float64, strict=False)
                        corr = s1.pearson_corr(s2)
                        row.append(_safe(corr))
                    except Exception:
                        row.append(None)
                mat.append(row)
        except Exception as exc:
            raise BadRequestError(f"Correlation failed: {exc}")

        return CorrelationResult(
            node_id=self.settings.node_id,
            path=req.path,
            columns=cols,
            matrix=mat,
            source_rows=source_rows,
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
        """How many rows of ``plan`` fit the byte budget — measured, not guessed.

        Streams a small sample, measures its in-memory Arrow size, and divides
        the budget by the per-row cost. A wide row (many/large columns) yields a
        small cap; a narrow row a large one. Bounded by ``analysis_max_rows`` so
        a degenerate estimate can't blow up. Returns the row cap."""
        budget = max_bytes or self.settings.analysis_max_bytes
        sample = plan.head(2048).collect(engine="streaming")
        if sample.height == 0:
            return self.settings.analysis_max_rows
        per_row = max(1, sample.estimated_size() // sample.height)
        cap = int(budget // per_row)
        return max(256, min(cap, self.settings.analysis_max_rows))

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
        ema = val.ewm_mean(span=window, ignore_nulls=True)
        # Running drawdown: equity = cumulative growth from the first bar; the
        # drawdown at each point is how far equity sits below its running peak.
        equity = (1.0 + cum)
        peak = equity.cum_max()
        drawdown = equity / peak - 1.0
        index = (
            df[req.order_by].to_list() if (req.order_by and req.order_by in cols)
            else list(range(df.height))
        )

        # Scalar risk/return summary (annualized) over the realized returns.
        ppy = max(1, req.periods_per_year)
        r = ret.drop_nulls().to_numpy()
        metrics = FinanceMetrics()
        if r.size and val.drop_nulls().len() >= 2:
            first = float(val.drop_nulls()[0])
            last = float(val.drop_nulls()[-1])
            total_ret = (last / first - 1.0) if first else None
            years = r.size / ppy
            cagr = ((last / first) ** (1.0 / years) - 1.0) if (first and years > 0 and last > 0) else None
            mean_r = float(np.nanmean(r))
            std_r = float(np.nanstd(r, ddof=1)) if r.size > 1 else 0.0
            ann_return = mean_r * ppy
            ann_vol = std_r * math.sqrt(ppy)
            rf_per = req.risk_free / ppy
            downside = r[r < rf_per]
            down_dev = (float(np.sqrt(np.mean((downside - rf_per) ** 2))) * math.sqrt(ppy)
                        if downside.size else 0.0)
            max_dd = float(drawdown.min()) if drawdown.drop_nulls().len() else None
            metrics = FinanceMetrics(
                total_return=total_ret,
                cagr=cagr,
                ann_return=ann_return,
                ann_volatility=ann_vol,
                sharpe=((ann_return - req.risk_free) / ann_vol) if ann_vol else None,
                sortino=((ann_return - req.risk_free) / down_dev) if down_dev else None,
                max_drawdown=max_dd,
                calmar=(cagr / abs(max_dd)) if (cagr is not None and max_dd) else None,
            )

        return FinanceResult(
            node_id=self.settings.node_id, path=req.path, column=req.column, window=window,
            index=[_safe(v) for v in index],
            value=_safe_list(val), pct_change=_safe_list(ret), cum_return=_safe_list(cum),
            roll_mean=_safe_list(roll_mean), roll_vol=_safe_list(roll_vol),
            ema=_safe_list(ema), drawdown=_safe_list(drawdown), metrics=metrics,
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
