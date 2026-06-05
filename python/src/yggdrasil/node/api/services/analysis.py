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
    DescribeResult,
    ExportRequest,
    FilterSpec,
    FinanceMetrics,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    IndicatorSeries,
    IndicatorsRequest,
    IndicatorsResult,
    OhlcRequest,
    OhlcResult,
    PivotRequest,
    PivotResult,
    PortfolioAsset,
    PortfolioRequest,
    PortfolioResult,
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

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        return await run_in_threadpool(partial(self._aggregate, req))

    async def pivot(self, req: PivotRequest) -> PivotResult:
        return await run_in_threadpool(partial(self._pivot, req))

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

    async def indicators(self, req: "IndicatorsRequest") -> "IndicatorsResult":
        return await run_in_threadpool(partial(self._indicators, req))

    async def portfolio(self, req: "PortfolioRequest") -> "PortfolioResult":
        return await run_in_threadpool(partial(self._portfolio, req))

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

    def _pivot(self, req: PivotRequest) -> PivotResult:
        """Excel-style cross-tab. Streams a group-by over ``rows + columns``
        (memory bounded, correct totals), then shapes the bounded grouped
        frame into a wide table in-process — full control over the flattened
        column headers and top-N column capping (avoids polars' version-y
        pivot column naming)."""
        if not req.measures:
            raise BadRequestError("pivot needs at least one measure")
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        keys = list(dict.fromkeys(req.rows + req.columns))
        for k in keys:
            if k not in cols:
                raise BadRequestError(f"pivot field {k!r} not found")
        exprs, measure_names = [], []
        for m in req.measures:
            if m.agg not in _AGGS:
                raise BadRequestError(f"unknown agg {m.agg!r}; one of {sorted(_AGGS)}")
            if m.column not in cols:
                raise BadRequestError(f"measure column {m.column!r} not found")
            name = f"{m.column}_{m.agg}"
            exprs.append(getattr(pl.col(m.column), m.agg)().alias(name))
            measure_names.append(name)

        keep = list(dict.fromkeys(keys + [m.column for m in req.measures]))
        plan = lf.select(keep)
        grouped = (plan.group_by(keys).agg(exprs) if keys else plan.select(exprs))
        grouped = grouped.collect(engine="streaming")
        source_rows = lf.select(pl.len()).collect(engine="streaming").item()
        if grouped.height > self.settings.pivot_max_groups:
            raise BadRequestError(
                f"pivot would materialise {grouped.height:,} groups (cap "
                f"{self.settings.pivot_max_groups:,}); add filters or drop a field"
            )

        # Rows-only (or no fields): the grouped frame is already the result —
        # row fields followed by one column per measure.
        if not req.columns:
            out = grouped.sort(req.rows) if req.rows else grouped
            row_count = out.height
            out = out.head(req.row_limit)
            result_rows = [[_safe(v) for v in r] for r in out.iter_rows()]
            has_total = bool(req.totals and req.rows)
            if has_total:  # grand-total row aggregated over the whole frame
                grand = plan.select(exprs).collect(engine="streaming")
                total_row = ["Total"] + [""] * (len(req.rows) - 1) + [_safe(grand[mn][0]) for mn in measure_names]
                result_rows.append(total_row)
            return PivotResult(
                node_id=self.settings.node_id, path=req.path,
                row_fields=req.rows, column_fields=[], measures=measure_names,
                columns=out.columns, rows=result_rows,
                row_count=row_count, col_count=0,
                total_columns=0, has_total_row=has_total,
                source_rows=source_rows, truncated=False,
            )

        # Cross-tab: shape the (bounded) grouped frame into a wide table.
        multi_measure = len(measure_names) > 1
        cell: dict[tuple, dict[tuple, list]] = {}
        col_labels: dict[tuple, str] = {}
        weights: dict[tuple, float] = {}
        for r in grouped.iter_rows(named=True):
            rt = tuple(r[k] for k in req.rows)
            ct = tuple(r[k] for k in req.columns)
            cell.setdefault(rt, {})[ct] = [r[mn] for mn in measure_names]
            col_labels.setdefault(ct, " / ".join("∅" if v is None else str(v) for v in ct))
            w = r[measure_names[0]]
            weights[ct] = weights.get(ct, 0.0) + (float(w) if isinstance(w, (int, float)) and w == w else 0.0)

        col_count = len(col_labels)
        combos = list(col_labels)
        truncated = col_count > req.col_limit
        if truncated:  # keep the top-N column groups by the first measure
            combos = sorted(combos, key=lambda c: weights.get(c, 0.0), reverse=True)[:req.col_limit]
        combos.sort(key=lambda c: tuple("" if v is None else str(v) for v in c))

        header = list(req.rows)
        plan_cols: list[tuple[tuple, int]] = []
        for ct in combos:
            for mi, mn in enumerate(measure_names):
                header.append(f"{col_labels[ct]} · {mn}" if multi_measure else col_labels[ct])
                plan_cols.append((ct, mi))

        # Totals: aggregated over the *source* (not summed cells), so the total
        # of a mean/median is correct. Row totals group by rows, the grand-total
        # row groups by columns, the corner cell groups by nothing.
        do_totals = req.totals
        row_tot: dict[tuple, list] = {}
        col_tot: dict[tuple, list] = {}
        grand_vals: list = [None] * len(measure_names)
        if do_totals:
            if req.rows:
                for r in plan.group_by(req.rows).agg(exprs).collect(engine="streaming").iter_rows(named=True):
                    row_tot[tuple(r[k] for k in req.rows)] = [r[mn] for mn in measure_names]
            for r in plan.group_by(req.columns).agg(exprs).collect(engine="streaming").iter_rows(named=True):
                col_tot[tuple(r[k] for k in req.columns)] = [r[mn] for mn in measure_names]
            g = plan.select(exprs).collect(engine="streaming")
            grand_vals = [g[mn][0] for mn in measure_names]
            for mn in measure_names:
                header.append(f"Total · {mn}" if multi_measure else "Total")

        row_tuples = sorted(cell, key=lambda rt: tuple("" if v is None else str(v) for v in rt))
        row_count = len(row_tuples)
        row_tuples = row_tuples[:req.row_limit]
        result_rows = []
        for rt in row_tuples:
            row = [_safe(v) for v in rt]
            bycol = cell[rt]
            for ct, mi in plan_cols:
                vals = bycol.get(ct)
                row.append(_safe(vals[mi]) if vals is not None else None)
            if do_totals:
                tv = (row_tot.get(rt) if req.rows else grand_vals)
                for mi in range(len(measure_names)):
                    row.append(_safe(tv[mi]) if tv is not None else None)
            result_rows.append(row)

        has_total_row = bool(do_totals and req.rows)
        if has_total_row:  # bottom grand-total row: column totals + corner
            grow = ["Total"] + [""] * (len(req.rows) - 1)
            for ct, mi in plan_cols:
                cv = col_tot.get(ct)
                grow.append(_safe(cv[mi]) if cv is not None else None)
            for mi in range(len(measure_names)):
                grow.append(_safe(grand_vals[mi]))
            result_rows.append(grow)

        return PivotResult(
            node_id=self.settings.node_id, path=req.path,
            row_fields=req.rows, column_fields=req.columns, measures=measure_names,
            columns=header, rows=result_rows,
            row_count=row_count, col_count=col_count,
            total_columns=len(measure_names) if do_totals else 0,
            has_total_row=has_total_row,
            source_rows=source_rows, truncated=truncated,
        )

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
        exprs.append(pl.len().alias("__n"))
        row = lf.select(num).select(exprs).collect(engine="streaming")
        rows = [[_safe(row[f"{c}|{stat}"][0]) for c in num] for stat in _STATS]
        source_rows = int(row["__n"][0])
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

    # -- trading indicators -------------------------------------------------

    def _indicators(self, req: IndicatorsRequest) -> IndicatorsResult:
        lf = self._apply_filters(self._frame(req.path), req.filters)
        cols = set(lf.collect_schema().names())
        if req.column not in cols:
            raise BadRequestError(f"column {req.column!r} not found")
        has_x = bool(req.order_by and req.order_by in cols)
        keep = list(dict.fromkeys([req.column] + ([req.order_by] if has_x else [])))
        plan = lf.select(keep)
        if has_x:
            plan = plan.sort(req.order_by)
        cap = min(req.limit, self._row_cap_for_bytes(plan))
        df = plan.head(cap + 1).collect(engine="streaming")
        truncated = df.height > cap
        if truncated:
            df = df.head(cap)

        val = df[req.column].cast(pl.Float64, strict=False)
        index = df[req.order_by].to_list() if has_x else list(range(df.height))
        win = max(2, req.window)
        out: list[IndicatorSeries] = []

        def _emit(name: str, series: pl.Series) -> None:
            out.append(IndicatorSeries(name=name, values=_safe_list(series)))

        for ind in req.indicators:
            if ind == "rsi":
                # RSI: rolling mean of gains over losses. Equal-weighted rolling
                # avg (not Wilder's EMA) keeps it polars-native.
                delta = val - val.shift(1)
                gain = delta.clip(lower_bound=0.0)
                loss = (-delta).clip(lower_bound=0.0)
                avg_gain = gain.rolling_mean(window_size=win)
                avg_loss = loss.rolling_mean(window_size=win)
                rs = avg_gain / avg_loss
                rsi = (100.0 - 100.0 / (1.0 + rs)).fill_nan(None)
                # Zero average loss → no downside in the window → RSI 100 (but
                # keep the leading NaN-warmup rows null, where avg_gain is null).
                hot = (avg_loss == 0.0) & avg_gain.is_not_null()
                rsi = rsi.zip_with(~hot, pl.repeat(100.0, rsi.len(), eager=True))
                _emit("rsi", rsi)
            elif ind == "macd":
                ema12 = val.ewm_mean(span=12, ignore_nulls=True)
                ema26 = val.ewm_mean(span=26, ignore_nulls=True)
                macd_line = ema12 - ema26
                signal = macd_line.ewm_mean(span=9, ignore_nulls=True)
                hist = macd_line - signal
                _emit("macd_line", macd_line)
                _emit("macd_signal", signal)
                _emit("macd_hist", hist)
            elif ind == "bb":
                mid = val.rolling_mean(window_size=win)
                sd = val.rolling_std(window_size=win)
                _emit("bb_upper", mid + 2.0 * sd)
                _emit("bb_mid", mid)
                _emit("bb_lower", mid - 2.0 * sd)
            elif ind == "atr":
                # Single-column fallback: true range collapses to the absolute
                # bar-to-bar move when high/low aren't available.
                tr = (val - val.shift(1)).abs()
                _emit("atr", tr.rolling_mean(window_size=win))
            elif ind == "stoch":
                # Single-column fallback: %K over rolling price extremes.
                lo = val.rolling_min(window_size=win)
                hi = val.rolling_max(window_size=win)
                k = (100.0 * (val - lo) / (hi - lo)).fill_nan(None)
                d = k.rolling_mean(window_size=3)
                _emit("stoch_k", k)
                _emit("stoch_d", d)
            else:
                raise BadRequestError(
                    f"unknown indicator {ind!r}; one of rsi|macd|bb|atr|stoch")

        return IndicatorsResult(
            node_id=self.settings.node_id, path=req.path, column=req.column,
            index=[_safe(v) for v in index], price=_safe_list(val),
            indicators=out, truncated=truncated,
        )

    # -- portfolio analytics ------------------------------------------------

    def _portfolio(self, req: PortfolioRequest) -> PortfolioResult:
        if len(req.paths) != len(req.columns):
            raise BadRequestError(
                f"paths ({len(req.paths)}) and columns ({len(req.columns)}) "
                "must be the same length")
        if not req.paths:
            raise BadRequestError("portfolio needs at least one asset")
        if len(req.paths) > 8:
            raise BadRequestError(f"max 8 assets, got {len(req.paths)}")

        ppy = max(1, req.periods_per_year)
        labels: list[str] = []
        prices: list[np.ndarray] = []
        truncated = False
        for i, (path, column) in enumerate(zip(req.paths, req.columns)):
            lf = self._frame(path)
            cols = set(lf.collect_schema().names())
            if column not in cols:
                raise BadRequestError(f"column {column!r} not found in {path!r}")
            has_x = bool(req.order_by and req.order_by in cols)
            keep = list(dict.fromkeys([column] + ([req.order_by] if has_x else [])))
            plan = lf.select(keep)
            if has_x:
                plan = plan.sort(req.order_by)
            cap = min(req.limit, self._row_cap_for_bytes(plan))
            df = plan.head(cap + 1).collect(engine="streaming")
            if df.height > cap:
                truncated = True
                df = df.head(cap)
            prices.append(df[column].cast(pl.Float64, strict=False).to_numpy())
            label = req.labels[i] if i < len(req.labels) else self.fs._resolve(path).stem
            labels.append(label)

        # Align by position to the shortest series — the simplest cross-asset
        # join that needs no shared key (documented assumption).
        n = min(p.shape[0] for p in prices)
        px = np.vstack([p[:n].astype(float) for p in prices])  # (assets, n)
        # Periodic simple returns; first column is NaN (no prior bar).
        rets = np.full_like(px, np.nan)
        rets[:, 1:] = px[:, 1:] / px[:, :-1] - 1.0

        assets: list[PortfolioAsset] = []
        import warnings
        with warnings.catch_warnings():
            # First column is all-NaN (no prior bar) → empty-slice mean; ignore.
            warnings.simplefilter("ignore", RuntimeWarning)
            port_ret = np.nanmean(rets, axis=0)  # equal-weight portfolio return
        for ai, label in enumerate(labels):
            p = px[ai]
            r = rets[ai][1:]
            r = r[np.isfinite(r)]
            metric = PortfolioAsset(label=label)
            if r.size and np.isfinite(p[0]) and p[0]:
                first, last = float(p[0]), float(p[-1])
                metric.total_return = (last / first - 1.0) if first else None
                mean_r = float(np.nanmean(r))
                std_r = float(np.nanstd(r, ddof=1)) if r.size > 1 else 0.0
                ann_ret = mean_r * ppy
                ann_vol = std_r * math.sqrt(ppy)
                metric.ann_return = ann_ret
                metric.ann_volatility = ann_vol
                metric.sharpe = ((ann_ret - req.risk_free) / ann_vol) if ann_vol else None
                equity = np.cumprod(1.0 + np.nan_to_num(rets[ai], nan=0.0))
                peak = np.maximum.accumulate(equity)
                dd = equity / peak - 1.0
                metric.max_drawdown = float(dd.min()) if dd.size else None
                # Beta vs the equal-weight portfolio over aligned bars.
                a = rets[ai][1:]
                b = port_ret[1:]
                mask = np.isfinite(a) & np.isfinite(b)
                if mask.sum() > 1:
                    var_b = float(np.var(b[mask], ddof=1))
                    cov = float(np.cov(a[mask], b[mask], ddof=1)[0, 1])
                    metric.beta = (cov / var_b) if var_b else None
            assets.append(metric)

        # Correlation matrix over the aligned per-asset return rows.
        valid = rets[:, 1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(np.nan_to_num(valid, nan=0.0)) if n > 2 else np.full((len(labels), len(labels)), np.nan)
        corr = np.atleast_2d(corr)
        correlation = [[_safe(float(corr[i, j])) for j in range(len(labels))] for i in range(len(labels))]

        # Equal-weight portfolio VaR/CVaR at the requested confidence: the
        # loss quantile of the combined return distribution.
        pr = port_ret[np.isfinite(port_ret)]
        var = cvar = None
        if pr.size:
            q = max(0.0, min(1.0, 1.0 - req.confidence))
            var = float(np.quantile(pr, q))
            tail = pr[pr <= var]
            cvar = float(tail.mean()) if tail.size else var

        index = list(range(n))
        return PortfolioResult(
            node_id=self.settings.node_id, labels=labels,
            index=[_safe(v) for v in index],
            prices=[[_safe(float(v)) for v in px[i]] for i in range(len(labels))],
            returns=[[_safe(float(v)) for v in rets[i]] for i in range(len(labels))],
            correlation=correlation, assets=assets,
            var_95=_safe(var), cvar_95=_safe(cvar), truncated=truncated,
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
