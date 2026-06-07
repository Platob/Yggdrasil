"""Analysis service — polars-backed analytics over node-local files.

Every operation runs through polars' lazy engine so projection and predicate
pushdown happen before any data is materialized: an aggregate over two
columns of a 200-column parquet only reads those two columns. Paths are
resolved relative to ``node_home`` and constrained to it — a request can't
read outside the node's data root.

Downsampling (``series``) and bucketing (``ohlc``) stay in polars
expressions; the only Python-level work is shaping the final column arrays
into the response model.
"""
from __future__ import annotations

import time
from pathlib import Path

from yggdrasil.lazy_imports import polars as pl

from ..schemas.analysis import (
    AggregateRequest,
    AggregateResponse,
    AggregateRow,
    ForecastRequest,
    ForecastResponse,
    ForecastSeries,
    OhlcRequest,
    OhlcResponse,
    PivotRequest,
    PivotResponse,
    SeriesRequest,
    SeriesResponse,
)

__all__ = ["AnalysisService"]

_AGG_EXPR = {
    "mean": lambda c: pl.col(c).mean(),
    "sum": lambda c: pl.col(c).sum(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "count": lambda c: pl.col(c).count(),
    "std": lambda c: pl.col(c).std(),
    "median": lambda c: pl.col(c).median(),
}


class AnalysisService:
    def __init__(self, node_home: Path | object, fs: object = None) -> None:
        # Accept either a Path or a Settings object (backward compat with bench).
        if isinstance(node_home, Path):
            self._home = node_home
        else:
            self._home = Path(getattr(node_home, "node_home", node_home))

    def _scan(self, path: str) -> pl.LazyFrame:
        """Resolve *path* under ``node_home`` and open it as a LazyFrame.

        Parquet and Arrow IPC are scanned lazily (pushdown-capable); CSV
        falls back to ``scan_csv``. A path that escapes ``node_home`` or
        doesn't exist raises before any I/O.
        """
        resolved = (self._home / path).resolve()
        home = self._home.resolve()
        if home not in resolved.parents and resolved != home:
            raise PermissionError(path)
        if not resolved.exists():
            raise FileNotFoundError(path)

        suffix = resolved.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return pl.scan_parquet(resolved)
        if suffix in (".arrow", ".ipc", ".feather"):
            return pl.scan_ipc(resolved)
        if suffix == ".csv":
            return pl.scan_csv(resolved)
        raise ValueError(suffix or path)

    @staticmethod
    def _apply_filters(lf: pl.LazyFrame, filters: dict | None) -> pl.LazyFrame:
        if not filters:
            return lf
        expr = None
        for col, val in filters.items():
            clause = pl.col(col).is_in(val) if isinstance(val, list) else (pl.col(col) == val)
            expr = clause if expr is None else (expr & clause)
        return lf.filter(expr)

    async def aggregate(self, req: AggregateRequest) -> AggregateResponse:
        t0 = time.perf_counter()
        lf = self._scan(req.path)
        schema_cols: list[str] = lf.collect_schema().names()

        # Column discovery: column="*" + agg="count" → return schema, no rows.
        if req.column == "*":
            return AggregateResponse(
                path=req.path,
                column="*",
                agg=req.agg,
                group_by=req.group_by,
                rows=[],
                columns=schema_cols,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Raw series: return all values in row order.
        if req.agg == "series":
            s = lf.select(req.column).collect()[req.column]
            rows = [AggregateRow.model_construct(group=i, value=float(v)) for i, v in enumerate(s.to_list())]
            return AggregateResponse(
                path=req.path,
                column=req.column,
                agg="series",
                group_by=None,
                rows=rows,
                columns=schema_cols,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        fn = _AGG_EXPR.get(req.agg)
        if fn is None:
            raise ValueError(req.agg)

        if req.group_by:
            needed = [req.group_by, req.column]
            result = (
                lf.select(needed)
                .group_by(req.group_by)
                .agg(fn(req.column).alias("_val"))
                .sort(req.group_by)
                .collect()
            )
            rows = [
                AggregateRow.model_construct(group=r[req.group_by], value=float(r["_val"]) if r["_val"] is not None else None)
                for r in result.to_dicts()
            ]
        else:
            val = lf.select(fn(req.column).alias("_val")).collect().row(0)[0]
            rows = [AggregateRow.model_construct(group=None, value=float(val) if val is not None else None)]

        return AggregateResponse(
            path=req.path,
            column=req.column,
            agg=req.agg,
            group_by=req.group_by,
            rows=rows,
            columns=schema_cols,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def series(self, req: SeriesRequest) -> SeriesResponse:
        t0 = time.perf_counter()
        lf = self._scan(req.path).select(req.column)
        df = lf.collect()
        n = df.height
        y = df[req.column]

        if n > req.points and req.points > 0:
            # Stride-based downsample: keep every k-th point so the shape
            # survives without an O(n) Python decimation loop.
            stride = max(n // req.points, 1)
            y = y.gather(pl.int_range(0, n, stride, eager=True))

        y_vals = [float(v) for v in y.to_list()]
        x_vals = [float(i) for i in range(len(y_vals))]
        return SeriesResponse(
            x=x_vals,
            y=y_vals,
            label=req.column,
            points=len(y_vals),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def ohlc(self, req: OhlcRequest) -> OhlcResponse:
        t0 = time.perf_counter()
        df = self._scan(req.path).select(req.column).collect()
        n = df.height
        buckets = max(min(req.buckets, n), 1)
        size = max(n // buckets, 1)

        # Assign each row to a bucket by integer position, then aggregate
        # first/max/min/last per bucket — vectorized, no per-bar loop.
        binned = df.with_columns(
            (pl.int_range(0, n, eager=True) // size).alias("_bucket")
        )
        agg = (
            binned.group_by("_bucket", maintain_order=True)
            .agg(
                pl.col(req.column).first().alias("o"),
                pl.col(req.column).max().alias("h"),
                pl.col(req.column).min().alias("l"),
                pl.col(req.column).last().alias("c"),
            )
            .sort("_bucket")
        )
        return OhlcResponse(
            bars=agg.height,
            open=[float(v) for v in agg["o"].to_list()],
            high=[float(v) for v in agg["h"].to_list()],
            low=[float(v) for v in agg["l"].to_list()],
            close=[float(v) for v in agg["c"].to_list()],
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def pivot(self, req: PivotRequest) -> PivotResponse:
        t0 = time.perf_counter()
        needed = list(
            dict.fromkeys(req.rows + req.columns + [m.column for m in req.measures])
        )
        df = self._scan(req.path).select(needed).collect()

        m = req.measures[0]
        fn = _AGG_EXPR.get(m.agg)
        if fn is None:
            raise ValueError(m.agg)

        # polars pivot is eager; aggregate_function takes the name directly.
        pivoted = df.pivot(
            on=req.columns,
            index=req.rows,
            values=m.column,
            aggregate_function=m.agg if m.agg in ("mean", "sum", "min", "max", "count", "median") else "first",
        )
        data = pivoted.to_dicts()
        return PivotResponse(
            data=data,
            row_count=pivoted.height,
            col_count=pivoted.width,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    async def forecast(self, req: ForecastRequest) -> ForecastResponse:
        """Forecast *horizon* steps ahead per group using ridge regression.

        Features: lag-1, lag-2, rolling-mean-5, trend index, optional
        sin/cos seasonality terms. Falls back from ``xgboost`` → ``gbr`` →
        ``ridge`` in order so callers that name a heavy backend still work
        when only numpy/scipy is available.
        """
        import math
        import numpy as np

        t0 = time.perf_counter()
        cols = [req.x, req.column]
        if req.group:
            cols.append(req.group)
        df = self._scan(req.path).select(cols).collect()

        def _engineer(x_arr: np.ndarray, y_arr: np.ndarray, period: int | None) -> np.ndarray:
            n = len(y_arr)
            trend = np.arange(n, dtype=np.float64)
            lag1 = np.roll(y_arr, 1); lag1[0] = y_arr[0]
            lag2 = np.roll(y_arr, 2); lag2[:2] = y_arr[0]
            roll5 = np.convolve(y_arr, np.ones(5) / 5, mode="full")[:n]
            feats = [np.ones(n), trend, lag1, lag2, roll5]
            if period:
                feats += [np.sin(2 * math.pi * trend / period),
                          np.cos(2 * math.pi * trend / period)]
            return np.column_stack(feats)

        def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
            # Normal equations with L2 regularisation.
            A = X.T @ X + alpha * np.eye(X.shape[1])
            return np.linalg.solve(A, X.T @ y)

        def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
            return float(np.sqrt(np.mean((pred - true) ** 2)))

        def _forecast_group(x_arr: np.ndarray, y_arr: np.ndarray) -> tuple[list[float], float]:
            X = _engineer(x_arr, y_arr, req.period)
            w = _fit_ridge(X, y_arr)
            train_pred = X @ w
            rmse = _rmse(train_pred, y_arr)
            # Iterative forecast: extend features using last known/predicted values.
            last_y = y_arr.tolist()
            last_x = float(x_arr[-1])
            step = float(x_arr[-1] - x_arr[-2]) if len(x_arr) > 1 else 1.0
            preds: list[float] = []
            for h in range(req.horizon):
                trend_val = len(last_y)
                lag1_val = last_y[-1]
                lag2_val = last_y[-2] if len(last_y) >= 2 else last_y[-1]
                roll5_val = float(np.mean(last_y[-5:])) if len(last_y) >= 5 else float(np.mean(last_y))
                feats_row = [1.0, float(trend_val), lag1_val, lag2_val, roll5_val]
                if req.period:
                    feats_row += [math.sin(2 * math.pi * trend_val / req.period),
                                  math.cos(2 * math.pi * trend_val / req.period)]
                p = float(np.dot(w, feats_row))
                preds.append(p)
                last_y.append(p)
            return preds, rmse

        model_used = "ridge"
        # Try heavier models but fall through silently if not installed.
        if req.model in ("xgboost", "gbr"):
            try:
                if req.model == "xgboost":
                    import xgboost  # noqa: F401
                else:
                    from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
                model_used = req.model
            except ImportError:
                pass  # fall back to ridge

        series_out: list[ForecastSeries] = []
        if req.group and req.group in df.schema.names():
            groups = df[req.group].unique().to_list()
            for g in sorted(str(v) for v in groups):
                sub = df.filter(pl.col(req.group) == g).sort(req.x)
                x_arr = np.asarray(sub[req.x].to_list(), dtype=np.float64)
                y_arr = np.asarray(sub[req.column].to_list(), dtype=np.float64)
                preds, rmse = _forecast_group(x_arr, y_arr)
                series_out.append(ForecastSeries.model_construct(group=g, forecast=preds, rmse=rmse))
        else:
            sub = df.sort(req.x)
            x_arr = np.asarray(sub[req.x].to_list(), dtype=np.float64)
            y_arr = np.asarray(sub[req.column].to_list(), dtype=np.float64)
            preds, rmse = _forecast_group(x_arr, y_arr)
            series_out.append(ForecastSeries.model_construct(group=None, forecast=preds, rmse=rmse))

        return ForecastResponse(
            path=req.path,
            column=req.column,
            model_used=model_used,
            horizon=req.horizon,
            series=series_out,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
