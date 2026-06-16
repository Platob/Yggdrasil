"""Trading-focused parquet analytics on polars lazy scans.

Everything runs through ``pl.scan_parquet`` so projection pushdown reads
only the columns a query touches and the group-by streams instead of
materialising the whole table. Operations:

- :meth:`aggregate` — group-by + measures.
- :meth:`series`    — evenly-spaced downsample of a numeric column.
- :meth:`ohlc`      — open/high/low/close resample into N buckets.
- :meth:`pivot`     — cross-tab (rows x columns) of a measure.
- :meth:`forecast`  — time-series forecast over engineered features,
  trying xgboost -> gbr -> ridge and skipping uninstalled backends.

No Python loops over rows — shaping is done with polars/numpy. Output
list-of-dicts are bounded grids, so the final ``to_dicts`` is cheap.
"""
from __future__ import annotations

import math

import polars as pl

from yggdrasil.exceptions.node import NodeBadRequestError, NodeNotFoundError

from ...config import Settings
from ..schemas.analysis import (
    AggregateRequest,
    AggregateResult,
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


_AGG_EXPRS = {
    "sum": lambda c: pl.col(c).sum(),
    "mean": lambda c: pl.col(c).mean(),
    "avg": lambda c: pl.col(c).mean(),
    "min": lambda c: pl.col(c).min(),
    "max": lambda c: pl.col(c).max(),
    "count": lambda c: pl.col(c).count(),
    "median": lambda c: pl.col(c).median(),
    "std": lambda c: pl.col(c).std(),
    "first": lambda c: pl.col(c).first(),
    "last": lambda c: pl.col(c).last(),
}


class AnalysisService:
    """Lazy, pushdown-friendly analytics over node-local parquet files."""

    def __init__(self, settings: Settings, *, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    def _scan(self, path: str) -> pl.LazyFrame:
        local = self.settings.node_home / path
        if not local.is_file():
            raise NodeNotFoundError(
                f"No parquet file {path!r} under {self.settings.node_home}."
            )
        return pl.scan_parquet(local)

    @staticmethod
    def _measure_expr(column: str, agg: str):
        builder = _AGG_EXPRS.get(agg)
        if builder is None:
            raise NodeBadRequestError(
                f"Unknown aggregation {agg!r}. Valid: {sorted(_AGG_EXPRS)}."
            )
        return builder(column).alias(f"{column}_{agg}")

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        lf = self._scan(req.path)
        select_cols = list(dict.fromkeys([*req.group_by, *(m.column for m in req.measures)]))
        aggs = [self._measure_expr(m.column, m.agg) for m in req.measures]
        # select() before group_by lets the optimizer push the projection
        # into the scan — only the touched columns are read off disk.
        out = (
            lf.select(select_cols)
            .group_by(req.group_by)
            .agg(aggs)
            .collect()
        )
        rows = out.to_dicts()
        return AggregateResult(rows=rows, group_count=out.height)

    async def series(self, req: SeriesRequest) -> SeriesResult:
        lf = self._scan(req.path)
        col = pl.col(req.column)
        # Pull just the numeric column (pushdown), then stride it down to
        # `points` evenly-spaced samples with gather — no per-row Python.
        df = lf.select(req.column).collect()
        n = df.height
        points = max(1, min(req.points, n))
        s = df.get_column(req.column)
        if n <= points:
            y = s.to_list()
            x = list(range(n))
        else:
            import numpy as np

            idx = np.linspace(0, n - 1, points).astype("int64")
            y = s.gather(pl.Series(idx)).to_list()
            x = idx.tolist()
        return SeriesResult(x=x, y=y, points=len(y))

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        lf = self._scan(req.path)
        df = lf.select(req.column).collect()
        n = df.height
        buckets = max(1, min(req.buckets, n))
        # Assign each row a bucket index by position, then aggregate
        # first/max/min/last per bucket — vectorized, no Python loop.
        bucket = (pl.int_range(0, n, dtype=pl.Int64) * buckets // max(1, n)).alias("_b")
        bars = (
            df.lazy()
            .with_columns(bucket)
            .group_by("_b")
            .agg(
                pl.col(req.column).first().alias("open"),
                pl.col(req.column).max().alias("high"),
                pl.col(req.column).min().alias("low"),
                pl.col(req.column).last().alias("close"),
            )
            .sort("_b")
            .drop("_b")
            .collect()
        )
        return OhlcResult(bars=bars.height, data=bars.to_dicts())

    async def pivot(self, req: PivotRequest) -> PivotResult:
        lf = self._scan(req.path)
        measure = req.measures[0]
        needed = list(dict.fromkeys([*req.rows, *req.columns, measure.column]))
        # Stream the group-by on just rows+cols+measure (pushdown), then
        # pivot the small grouped frame into the bounded grid.
        grouped = (
            lf.select(needed)
            .group_by([*req.rows, *req.columns])
            .agg(self._measure_expr(measure.column, measure.agg))
            .collect()
        )
        value_col = f"{measure.column}_{measure.agg}"
        wide = grouped.pivot(
            on=req.columns,
            index=req.rows,
            values=value_col,
            aggregate_function="first",
        )
        return PivotResult(
            row_count=wide.height,
            col_count=wide.width,
            data=wide.to_dicts(),
        )

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        import numpy as np

        Model, model_used = self._pick_model(req.model)

        needed = [req.x, req.column]
        if req.group:
            needed.append(req.group)
        needed = list(dict.fromkeys(needed))
        df = self._scan(req.path).select(needed).collect()

        if req.group:
            group_values = df.get_column(req.group).unique().sort().to_list()
        else:
            group_values = [None]

        series: list[ForecastSeries] = []
        for gv in group_values:
            sub = df.filter(pl.col(req.group) == gv) if req.group else df
            sub = sub.sort(req.x)
            y = sub.get_column(req.column).to_numpy().astype("float64")
            if y.shape[0] < 8:
                # too short to engineer lags — emit a flat hold forecast
                last = float(y[-1]) if y.shape[0] else 0.0
                series.append(
                    ForecastSeries(
                        group=str(gv) if gv is not None else None,
                        rmse=0.0,
                        predictions=[last] * req.horizon,
                    )
                )
                continue

            X, target = self._features(y, req.period)
            model = Model()
            model.fit(X, target)
            in_pred = model.predict(X)
            rmse = float(np.sqrt(np.mean((in_pred - target) ** 2)))

            preds = self._roll_forward(model, y, req.period, req.horizon)
            series.append(
                ForecastSeries(
                    group=str(gv) if gv is not None else None,
                    rmse=round(rmse, 6),
                    predictions=[float(p) for p in preds],
                )
            )

        return ForecastResult(series=series, model_used=model_used)

    # ------------------------- forecast internals ----------------------- #

    @staticmethod
    def _pick_model(requested: str):
        """Resolve the requested backend, falling back through the chain
        xgboost -> gbr -> ridge and skipping any that aren't installed."""
        chain = {
            "xgboost": ["xgboost", "gbr", "ridge"],
            "gbr": ["gbr", "ridge"],
            "ridge": ["ridge"],
        }.get(requested, ["ridge"])

        for name in chain:
            if name == "xgboost":
                try:
                    from xgboost import XGBRegressor  # type: ignore

                    return (lambda: XGBRegressor(
                        n_estimators=120, max_depth=4, learning_rate=0.1,
                        verbosity=0, n_jobs=1,
                    ), "xgboost")
                except Exception:
                    continue
            if name == "gbr":
                try:
                    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

                    return (lambda: GradientBoostingRegressor(), "gbr")
                except Exception:
                    continue
            if name == "ridge":
                try:
                    from sklearn.linear_model import Ridge  # type: ignore

                    return (lambda: Ridge(alpha=1.0), "ridge")
                except Exception:
                    continue

        raise NodeBadRequestError(
            f"No forecast backend available for model {requested!r}. "
            "Install one of: xgboost, scikit-learn."
        )

    @staticmethod
    def _features(y, period: int):
        """Build lag + rolling-mean + seasonal (sin/cos) features.

        Returns ``(X, target)`` aligned so row i predicts ``y[i]`` from its
        recent past. The first ``warmup`` points are dropped (no full lag
        window yet)."""
        import numpy as np

        n = y.shape[0]
        lags = [1, 2, 3, period] if period > 3 else [1, 2, 3]
        warmup = max(lags)
        roll = 5

        idx = np.arange(warmup, n)
        cols = []
        for lag in lags:
            cols.append(y[idx - lag])
        # rolling mean of the preceding `roll` points
        rolling = np.array([y[i - roll:i].mean() for i in idx]) if roll < warmup + 1 else \
            np.array([y[max(0, i - roll):i].mean() for i in idx])
        cols.append(rolling)
        # seasonal phase
        phase = 2 * math.pi * (idx % period) / period
        cols.append(np.sin(phase))
        cols.append(np.cos(phase))

        X = np.column_stack(cols)
        target = y[idx]
        return X, target

    @staticmethod
    def _roll_forward(model, y, period: int, horizon: int):
        """Recursively forecast `horizon` steps, feeding predictions back
        in as the new lags."""
        import numpy as np

        lags = [1, 2, 3, period] if period > 3 else [1, 2, 3]
        roll = 5
        hist = list(y.astype("float64"))
        preds = []
        for step in range(horizon):
            pos = len(hist)
            feats = [hist[pos - lag] for lag in lags]
            feats.append(float(np.mean(hist[max(0, pos - roll):pos])))
            phase = 2 * math.pi * (pos % period) / period
            feats.append(math.sin(phase))
            feats.append(math.cos(phase))
            yhat = float(model.predict(np.array([feats]))[0])
            preds.append(yhat)
            hist.append(yhat)
        return preds
