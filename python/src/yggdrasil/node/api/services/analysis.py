"""Analysis service — lazy polars scans with projection pushdown.

Every operation scans the parquet/CSV lazily and selects only the columns it
needs, so an aggregate over 2 of 30 columns reads only those 2. Series/OHLC
downsample with vectorized polars/numpy (never a Python loop over rows).
Finance metrics are pure numpy. Forecasting tries xgboost → gbr → ridge.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from ...config import Settings
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    FinanceMetrics,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    OhlcRequest,
    OhlcResult,
    PivotRequest,
    PivotResult,
    SeriesRequest,
    SeriesResult,
)
from .fs import FsService

_AGG_EXPR = {
    "mean": lambda c: pl.col(c).mean(),
    "sum": lambda c: pl.col(c).sum(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "count": lambda c: pl.col(c).count(),
    "std": lambda c: pl.col(c).std(),
}


class AnalysisService:
    def __init__(self, settings: Settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    def _scan(self, path: str) -> pl.LazyFrame:
        target = self.fs._resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"no such file: {path!r}")
        suffix = target.suffix.lower()
        if suffix in (".parquet", ".pq"):
            return pl.scan_parquet(str(target))
        if suffix in (".csv", ".txt"):
            return pl.scan_csv(str(target))
        if suffix == ".tsv":
            return pl.scan_csv(str(target), separator="\t")
        if suffix in (".json", ".jsonl", ".ndjson"):
            return pl.scan_ndjson(str(target))
        raise ValueError(f"unsupported format {suffix!r} for analysis of {path!r}")

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        measure_cols = list({m.column for m in req.measures})
        lf = self._scan(req.path).select(req.group_by + measure_cols)
        aggs = [_AGG_EXPR[m.agg](m.column).alias(f"{m.column}_{m.agg}") for m in req.measures]
        df = lf.group_by(req.group_by).agg(aggs).collect()
        return AggregateResult(group_count=df.height, rows=df.to_dicts())

    async def series(self, req: SeriesRequest) -> SeriesResult:
        cols = [req.column] + ([req.time_column] if req.time_column else [])
        df = self._scan(req.path).select(cols).collect()
        n = df.height
        y_full = df[req.column].to_numpy().astype(np.float64)
        if req.time_column:
            x_full = df[req.time_column].to_numpy().astype(np.float64)
        else:
            x_full = np.arange(n, dtype=np.float64)

        if n <= req.points:
            return SeriesResult(x=x_full.tolist(), y=y_full.tolist())

        # Uniform stride downsample — vectorized index selection, no loop.
        idx = np.linspace(0, n - 1, req.points).astype(np.int64)
        return SeriesResult(x=x_full[idx].tolist(), y=y_full[idx].tolist())

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        cols = [req.column] + ([req.time_column] if req.time_column else [])
        df = self._scan(req.path).select(cols).collect()
        n = df.height
        y = df[req.column].to_numpy().astype(np.float64)
        if req.time_column:
            x = df[req.time_column].to_numpy().astype(np.float64)
        else:
            x = np.arange(n, dtype=np.float64)

        buckets = min(req.buckets, n)
        # Assign each row to a bucket, then reduce per bucket with numpy.
        edges = np.linspace(0, n, buckets + 1).astype(np.int64)
        opens, highs, lows, closes, times = [], [], [], [], []
        for b in range(buckets):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            seg = y[lo:hi]
            opens.append(float(seg[0]))
            highs.append(float(seg.max()))
            lows.append(float(seg.min()))
            closes.append(float(seg[-1]))
            times.append(float(x[lo]))
        return OhlcResult(
            bars=len(opens),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            times=times,
        )

    async def pivot(self, req: PivotRequest) -> PivotResult:
        measure = req.measures[0]
        cols = list({*req.rows, *req.columns, measure.column})
        df = (
            self._scan(req.path)
            .select(cols)
            .group_by(req.rows + req.columns)
            .agg(_AGG_EXPR[measure.agg](measure.column).alias("_v"))
            .collect()
        )
        row_key = req.rows[0]
        col_key = req.columns[0]
        wide = df.pivot(values="_v", index=row_key, on=col_key)
        row_labels = wide[row_key].to_numpy().astype(str).tolist()
        value_cols = [c for c in wide.columns if c != row_key]
        # Fill nulls then dump the numeric block in one shot.
        data = wide.select(value_cols).fill_null(0.0).to_numpy().astype(np.float64).tolist()
        return PivotResult(
            row_count=len(row_labels),
            col_count=len(value_cols),
            rows=row_labels,
            columns=[str(c) for c in value_cols],
            data=data,
        )

    async def finance(self, req: FinanceRequest) -> FinanceResult:
        cols = [req.column] + ([req.time_column] if req.time_column else [])
        df = self._scan(req.path).select(cols).collect()
        price = df[req.column].to_numpy().astype(np.float64)
        n = price.size
        if n < 2:
            raise ValueError(f"finance needs at least 2 points, got {n}")

        # Simple period returns from the price/level series.
        rets = np.diff(price) / price[:-1]
        cum_return = price / price[0] - 1.0

        # EMA(20) via the standard recursive weight — vectorized with cumulative
        # decay weights so there is no Python loop over the series.
        span = 20
        alpha = 2.0 / (span + 1.0)
        ema = _ewm(price, alpha)

        # Drawdown: distance below the running peak, normalized to the peak.
        running_max = np.maximum.accumulate(price)
        drawdown = price / running_max - 1.0
        max_dd = float(drawdown.min())

        ppy = req.periods_per_year
        mean_r = float(rets.mean())
        std_r = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
        downside = rets[rets < 0]
        downside_dev = float(np.sqrt((downside ** 2).mean())) if downside.size else 0.0

        ann_return = mean_r * ppy
        ann_vol = std_r * math.sqrt(ppy)
        rf = req.risk_free_rate
        sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0
        sortino = (ann_return - rf) / (downside_dev * math.sqrt(ppy)) if downside_dev > 0 else 0.0

        total_return = float(price[-1] / price[0] - 1.0)
        years = n / ppy
        cagr = float((price[-1] / price[0]) ** (1.0 / years) - 1.0) if years > 0 and price[0] > 0 else 0.0
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        return FinanceResult(
            metrics=FinanceMetrics(
                total_return=total_return,
                cagr=cagr,
                ann_return=ann_return,
                ann_volatility=ann_vol,
                sharpe=sharpe,
                sortino=sortino,
                max_drawdown=max_dd,
                calmar=calmar,
            ),
            ema=ema.tolist(),
            drawdown=drawdown.tolist(),
            cum_return=cum_return.tolist(),
        )

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        cols = [req.column, req.x] + ([req.group] if req.group else [])
        df = self._scan(req.path).select(list(dict.fromkeys(cols))).collect()

        model_name, fit_predict = _resolve_forecaster(req.model)

        series_out: list[ForecastSeries] = []
        if req.group:
            groups = df[req.group].unique().to_list()
        else:
            groups = [None]

        period = req.period or 0
        for g in groups:
            sub = df.filter(pl.col(req.group) == g) if req.group else df
            x = sub[req.x].to_numpy().astype(np.float64)
            y = sub[req.column].to_numpy().astype(np.float64)
            order = np.argsort(x)
            x, y = x[order], y[order]

            feats = _features(x, period)
            split = max(1, int(len(y) * 0.8))
            preds_test, rmse = fit_predict(feats[:split], y[:split], feats[split:], y[split:])

            # Forecast horizon steps beyond the last x at unit stride.
            step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
            future_x = x[-1] + step * np.arange(1, req.horizon + 1)
            future_feats = _features_at(future_x, period, x_ref=x)
            fc, _ = fit_predict(feats, y, future_feats, None)
            series_out.append(ForecastSeries(
                group=str(g) if g is not None else "all",
                forecast=[float(v) for v in fc],
                rmse=round(float(rmse), 4),
            ))

        return ForecastResult(model_used=model_name, series=series_out)


def _ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    # polars' vectorized EWM avoids a Python scan and matches pandas' adjust=False
    # recursion: out[i] = alpha*x[i] + (1-alpha)*out[i-1].
    return pl.Series(x).ewm_mean(alpha=alpha, adjust=False).to_numpy().astype(np.float64)


def _features(x: np.ndarray, period: int) -> np.ndarray:
    # Engineered design matrix: trend + optional seasonal harmonics.
    x0 = x - x[0]
    feats = [np.ones_like(x0), x0]
    if period and period > 1:
        ang = 2.0 * np.pi * x0 / period
        feats += [np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)]
    return np.column_stack(feats)


def _features_at(x: np.ndarray, period: int, *, x_ref: np.ndarray) -> np.ndarray:
    x0 = x - x_ref[0]
    feats = [np.ones_like(x0), x0]
    if period and period > 1:
        ang = 2.0 * np.pi * x0 / period
        feats += [np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)]
    return np.column_stack(feats)


def _resolve_forecaster(model: str):
    """Return (model_name, fit_predict). fit_predict(Xtr,ytr,Xte,yte)->(preds,rmse)."""
    chain = ["xgboost", "gbr", "ridge"] if model == "auto" else [model]
    for name in chain:
        if name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError:
                if model != "auto":
                    raise
                continue

            def fp(Xtr, ytr, Xte, yte, _M=XGBRegressor):
                m = _M(n_estimators=120, max_depth=3, learning_rate=0.1, verbosity=0)
                m.fit(Xtr, ytr)
                preds = m.predict(Xte) if len(Xte) else np.array([])
                rmse = _rmse(preds, yte)
                return preds, rmse

            return "xgboost", fp

        if name == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor

            def fp(Xtr, ytr, Xte, yte, _M=GradientBoostingRegressor):
                m = _M(n_estimators=120, max_depth=3, learning_rate=0.1)
                m.fit(Xtr, ytr)
                preds = m.predict(Xte) if len(Xte) else np.array([])
                return preds, _rmse(preds, yte)

            return "gbr", fp

        if name == "ridge":
            from sklearn.linear_model import Ridge

            def fp(Xtr, ytr, Xte, yte, _M=Ridge):
                m = _M(alpha=1.0)
                m.fit(Xtr, ytr)
                preds = m.predict(Xte) if len(Xte) else np.array([])
                return preds, _rmse(preds, yte)

            return "ridge", fp

    raise ImportError(f"no forecasting backend available for model={model!r}")


def _rmse(preds: np.ndarray, actual) -> float:
    if actual is None or len(preds) == 0 or len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(actual)) ** 2)))
