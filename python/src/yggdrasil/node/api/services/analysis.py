"""Lazy analytics over node-local parquet/arrow files.

Every read is a polars ``scan_*`` so projection pushdown reaches the file: an
aggregate that touches 2 of 30 columns reads only those 2. The group-by, the
adaptive downsample, the OHLC resample and the cross-tab pivot all stay in
polars expressions — no Python loops over rows. ``forecast`` engineers lag +
cyclical features and fits xgboost → gradient-boosting → ridge in that order,
degrading to a NumPy ridge when neither ML backend is installed. ``finance``
computes the standard return/risk metric set with polars window expressions.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from ..schemas.analysis import (
    AggregateRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
)
from .fs import FsService

_AGG = {
    "mean": pl.Expr.mean,
    "sum": pl.Expr.sum,
    "count": pl.Expr.count,
    "min": pl.Expr.min,
    "max": pl.Expr.max,
    "std": pl.Expr.std,
}


class AggregateResult(BaseModel):
    group_count: int
    rows: list[dict]


class SeriesResult(BaseModel):
    x: list
    y: list


class OhlcResult(BaseModel):
    bars: int
    data: list[dict]


class PivotResult(BaseModel):
    row_count: int
    col_count: int
    rows: list[dict]


class ForecastSeriesResult(BaseModel):
    group: str | None
    x: list
    y_actual: list
    y_forecast: list
    rmse: float


class ForecastResult(BaseModel):
    model_used: str
    series: list[ForecastSeriesResult]


def _scan(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.scan_parquet(str(path))
    if suffix in (".arrow", ".arrows", ".ipc", ".feather"):
        return pl.scan_ipc(str(path))
    if suffix == ".csv":
        return pl.scan_csv(str(path))
    raise ValueError(
        f"Can't scan {path.name!r}: unsupported extension {suffix!r}. "
        f"Expected .parquet/.arrow/.csv."
    )


class AnalysisService:
    def __init__(self, settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs
        self._root = Path(settings.node_home)

    def _lf(self, relative: str) -> pl.LazyFrame:
        return _scan(self.fs._resolve(relative))

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        measure_cols = [m.column for m in req.measures]
        needed = list(dict.fromkeys([*req.group_by, *measure_cols]))
        # select() before group_by() lets the optimizer push the projection
        # into the parquet reader — only `needed` columns leave disk.
        aggs = [
            _AGG[m.agg](pl.col(m.column)).alias(f"{m.column}_{m.agg}")
            for m in req.measures
        ]
        out = (
            self._lf(req.path)
            .select(needed)
            .group_by(req.group_by)
            .agg(aggs)
            .sort(req.group_by)
            .collect()
        )
        return AggregateResult(group_count=out.height, rows=out.to_dicts())

    async def series(self, req: SeriesRequest) -> SeriesResult:
        df = self._lf(req.path).select(pl.col(req.column)).collect()
        n = df.height
        col = df.get_column(req.column)
        if n <= req.points:
            return SeriesResult(x=list(range(n)), y=col.to_list())
        # Adaptive downsample: bucket the row index into `points` groups and
        # take each bucket's mean — bounded output regardless of source size.
        idx = pl.int_range(0, n, eager=True)
        bucket = (idx * req.points // n).alias("_b")
        grouped = (
            pl.DataFrame({"_b": bucket, "v": col})
            .group_by("_b")
            .agg(pl.col("v").mean())
            .sort("_b")
        )
        return SeriesResult(x=grouped.get_column("_b").to_list(), y=grouped.get_column("v").to_list())

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        cols = [req.column] + ([req.ts_column] if req.ts_column else [])
        df = self._lf(req.path).select(cols).collect()
        n = df.height
        buckets = max(1, min(req.buckets, n))
        idx = pl.int_range(0, n, eager=True)
        bucket = (idx * buckets // n).alias("_b")
        work = df.with_columns(bucket)
        bars = (
            work.group_by("_b")
            .agg(
                pl.col(req.column).first().alias("open"),
                pl.col(req.column).max().alias("high"),
                pl.col(req.column).min().alias("low"),
                pl.col(req.column).last().alias("close"),
            )
            .sort("_b")
        )
        return OhlcResult(bars=bars.height, data=bars.drop("_b").to_dicts())

    async def pivot(self, req: PivotRequest) -> PivotResult:
        measure_cols = [m.column for m in req.measures]
        needed = list(dict.fromkeys([*req.rows, *req.columns, *measure_cols]))
        aggs = [
            _AGG[m.agg](pl.col(m.column)).alias(f"{m.column}_{m.agg}")
            for m in req.measures
        ]
        # Push projection, stream the group-by on rows+columns, then shape the
        # (bounded) cross-tab in memory.
        long = (
            self._lf(req.path)
            .select(needed)
            .group_by([*req.rows, *req.columns])
            .agg(aggs)
            .collect()
        )
        value_col = aggs and f"{req.measures[0].column}_{req.measures[0].agg}"
        wide = long.pivot(
            on=req.columns,
            index=req.rows,
            values=value_col,
            aggregate_function="sum",
        ).sort(req.rows)
        return PivotResult(row_count=wide.height, col_count=wide.width, rows=wide.to_dicts())

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        cols = [req.column, req.x] + ([req.group] if req.group else [])
        df = self._lf(req.path).select(list(dict.fromkeys(cols))).sort(req.x).collect()

        fit, model_used = _pick_model(req.model)
        groups: list[tuple[str | None, pl.DataFrame]]
        if req.group:
            groups = [(str(k[0]) if isinstance(k, tuple) else str(k), g)
                      for k, g in df.group_by(req.group, maintain_order=True)]
        else:
            groups = [(None, df)]

        series: list[ForecastSeriesResult] = []
        for key, g in groups:
            y = g.get_column(req.column).to_numpy()
            xs = g.get_column(req.x).to_numpy()
            yhat, future_x, future_y, rmse = _forecast_one(
                xs, y, req.horizon, req.period, fit)
            series.append(ForecastSeriesResult(
                group=key,
                x=future_x.tolist(),
                y_actual=y[-min(len(y), req.horizon):].tolist(),
                y_forecast=future_y.tolist(),
                rmse=round(float(rmse), 6),
            ))
        return ForecastResult(model_used=model_used, series=series)

    async def finance(self, path: str, column: str, ts_column: str | None = None) -> dict:
        cols = [column] + ([ts_column] if ts_column else [])
        df = self._lf(path).select(list(dict.fromkeys(cols)))
        if ts_column:
            df = df.sort(ts_column)
        df = df.collect()
        price = df.get_column(column)
        n = price.len()
        if n < 2:
            raise ValueError(f"Need >=2 rows to compute finance metrics, got {n}.")

        ret = price.pct_change().drop_nulls()
        running_max = price.cum_max()
        drawdown = (price - running_max) / running_max
        ema = price.ewm_mean(span=20, ignore_nulls=True)

        initial = float(price[0])
        final = float(price[-1])
        # Treat each row as one trading day for the annualization factor.
        years = max(n / 252.0, 1e-9)
        total_return = final / initial - 1.0
        cagr = (final / initial) ** (1.0 / years) - 1.0 if initial > 0 else 0.0
        mean_r = float(ret.mean() or 0.0)
        std_r = float(ret.std() or 0.0)
        neg = ret.filter(ret < 0)
        std_neg = float(neg.std() or 0.0)
        ann_return = mean_r * 252.0
        ann_vol = std_r * math.sqrt(252.0)
        sharpe = (mean_r / std_r * math.sqrt(252.0)) if std_r > 0 else 0.0
        sortino = (mean_r / std_neg * math.sqrt(252.0)) if std_neg > 0 else 0.0
        max_dd = float(drawdown.min() or 0.0)
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

        return {
            "total_return": round(total_return, 6),
            "cagr": round(cagr, 6),
            "ann_return": round(ann_return, 6),
            "ann_volatility": round(ann_vol, 6),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_dd, 6),
            "calmar": round(calmar, 4),
            "ema": ema.to_list(),
            "drawdown": drawdown.to_list(),
        }


# ---------------------------------------------------------------------------
# forecasting internals — lazy ML backend selection + feature engineering
# ---------------------------------------------------------------------------

def _pick_model(requested: str):
    """Return ``(fit_fn, label)`` for the best available backend.

    ``fit_fn(X, y) -> predict(X) -> yhat``. Order: xgboost → gbr → ridge, with
    a NumPy ridge as the always-available floor. ``requested`` narrows the
    search ("ridge" never reaches for xgboost).
    """
    order = {"auto": ("xgboost", "gbr", "ridge"),
             "xgboost": ("xgboost", "gbr", "ridge"),
             "gbr": ("gbr", "ridge"),
             "ridge": ("ridge",)}[requested]

    for name in order:
        if name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                continue

            def fit(X, y, _M=XGBRegressor):
                m = _M(n_estimators=120, max_depth=4, learning_rate=0.1, verbosity=0)
                m.fit(X, y)
                return m.predict
            return fit, "xgboost"
        if name == "gbr":
            try:
                from sklearn.ensemble import GradientBoostingRegressor
            except ImportError:
                continue

            def fit(X, y, _M=GradientBoostingRegressor):
                m = _M(n_estimators=120, max_depth=3, learning_rate=0.1)
                m.fit(X, y)
                return m.predict
            return fit, "gbr"
        # ridge: prefer sklearn, else a closed-form NumPy ridge (always works).
        try:
            from sklearn.linear_model import Ridge

            def fit(X, y, _M=Ridge):
                m = _M(alpha=1.0)
                m.fit(X, y)
                return m.predict
            return fit, "ridge"
        except ImportError:
            return _numpy_ridge_fit, "ridge"
    return _numpy_ridge_fit, "ridge"


def _numpy_ridge_fit(X, y):
    import numpy as np
    Xb = np.column_stack([np.ones(len(X)), X])
    lam = 1.0
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[0, 0] -= lam  # don't regularize the intercept
    w = np.linalg.solve(A, Xb.T @ y)

    def predict(Xp):
        Xpb = np.column_stack([np.ones(len(Xp)), Xp])
        return Xpb @ w
    return predict


def _features(x, y, period: int | None, n_lags: int):
    """Engineer lag + cyclical + trend features aligned to targets ``y[n_lags:]``."""
    import numpy as np
    n = len(y)
    rows = n - n_lags
    feats = [np.asarray(x[n_lags:], dtype=float)]  # trend term
    for lag in range(1, n_lags + 1):
        feats.append(np.asarray(y[n_lags - lag:n - lag], dtype=float))
    if period:
        phase = 2 * np.pi * (np.asarray(x[n_lags:], dtype=float) % period) / period
        feats.append(np.sin(phase))
        feats.append(np.cos(phase))
    X = np.column_stack(feats)
    target = np.asarray(y[n_lags:], dtype=float)
    return X, target, rows


def _forecast_one(x, y, horizon: int, period: int | None, fit):
    import numpy as np
    n_lags = min(24, max(1, len(y) // 4))
    if len(y) <= n_lags + 2:
        # Too short to fit; flat-forecast the last value.
        last = float(y[-1]) if len(y) else 0.0
        step = (x[-1] - x[-2]) if len(x) >= 2 else 1
        fut_x = np.array([x[-1] + step * (i + 1) for i in range(horizon)], dtype=float)
        return y, fut_x, np.full(horizon, last), 0.0

    X, target, _ = _features(x, y, period, n_lags)
    predict = fit(X, target)
    in_sample = predict(X)
    rmse = float(np.sqrt(np.mean((in_sample - target) ** 2)))

    # Roll the forecast forward, feeding each prediction back as the next lag.
    step = (x[-1] - x[-2]) if len(x) >= 2 else 1
    hist = list(np.asarray(y, dtype=float))
    fut_x = []
    fut_y = []
    cur_x = float(x[-1])
    for _ in range(horizon):
        cur_x += step
        feats = [cur_x]
        for lag in range(1, n_lags + 1):
            feats.append(hist[-lag])
        if period:
            phase = 2 * np.pi * (cur_x % period) / period
            feats.append(np.sin(phase))
            feats.append(np.cos(phase))
        pred = float(predict(np.array([feats]))[0])
        hist.append(pred)
        fut_x.append(cur_x)
        fut_y.append(pred)
    return y, np.array(fut_x), np.array(fut_y), rmse
