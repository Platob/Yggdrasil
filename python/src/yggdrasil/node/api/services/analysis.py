"""Analysis engine — lazy polars scans with projection pushdown + streaming.

Everything runs on ``pl.scan_parquet``: an aggregate that names 2 of 30
columns reads only those 2 (pushdown) and streams the group-by, instead of
loading the whole table first. :meth:`series` adaptively downsamples by
striding, :meth:`ohlc` resamples into fixed buckets, :meth:`pivot` shapes a
bounded cross-tab, and :meth:`forecast` engineers lag + seasonality features
and fits one model per group (xgboost → sklearn GBR → ridge, whichever is
installed and requested).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from yggdrasil.exceptions.api import BadRequestError, NotFoundError
from yggdrasil.node.api.schemas.analysis import (
    AggregateRequest,
    AggregateResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    OhlcBar,
    OhlcRequest,
    OhlcResult,
    PivotRequest,
    PivotResult,
    SeriesRequest,
    SeriesResult,
)
from yggdrasil.node.config import Settings
from yggdrasil.node.api.services.fs import FsService

_AGG = {
    "mean": pl.Expr.mean,
    "sum": pl.Expr.sum,
    "min": pl.Expr.min,
    "max": pl.Expr.max,
    "count": pl.Expr.count,
    "std": pl.Expr.std,
}


class AnalysisService:
    def __init__(self, settings: Settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    def _scan(self, rel: str) -> pl.LazyFrame:
        path = self.fs._resolve(rel)
        if not path.is_file():
            raise NotFoundError(f"No such file: {rel!r}.")
        return pl.scan_parquet(str(path))

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        lf = self._scan(req.path)
        exprs = [_measure_expr(m) for m in req.measures]
        # Project to only the group + measure columns so the parquet reader
        # skips every other column in the file.
        keep = list(dict.fromkeys([*req.group_by, *(m.column for m in req.measures)]))
        out = lf.select(keep).group_by(req.group_by).agg(exprs).collect()
        return AggregateResult(
            group_count=out.height,
            columns=out.columns,
            data=out.to_dicts(),
        )

    async def series(self, req: SeriesRequest) -> SeriesResult:
        cols = [req.column] + ([req.time_col] if req.time_col else [])
        lf = self._scan(req.path).select(cols)
        n = lf.select(pl.len()).collect().item()
        stride = max(1, n // req.points) if req.points > 0 else 1
        # Stride the row index, then gather — only the strided rows materialize.
        sampled = (
            lf.with_row_index("__i")
            .filter(pl.col("__i") % stride == 0)
            .collect()
        )
        y = sampled[req.column].cast(pl.Float64).to_list()
        if req.time_col:
            x = sampled[req.time_col].to_list()
        else:
            x = sampled["__i"].to_list()
        return SeriesResult(x=x, y=y, column=req.column)

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        lf = self._scan(req.path).select([req.column])
        n = lf.select(pl.len()).collect().item()
        if n == 0:
            return OhlcResult(bars=0, data=[])
        per = max(1, n // req.buckets)
        out = (
            lf.with_row_index("__i")
            .with_columns((pl.col("__i") // per).alias("__b"))
            .group_by("__b")
            .agg([
                pl.col(req.column).first().alias("open"),
                pl.col(req.column).max().alias("high"),
                pl.col(req.column).min().alias("low"),
                pl.col(req.column).last().alias("close"),
            ])
            .sort("__b")
            .collect()
        )
        bars = [
            OhlcBar(open=float(o), high=float(h), low=float(lo), close=float(c), bucket=int(b))
            for b, o, h, lo, c in zip(out["__b"], out["open"], out["high"], out["low"], out["close"])
        ]
        return OhlcResult(bars=len(bars), data=bars)

    async def pivot(self, req: PivotRequest) -> PivotResult:
        if not req.measures:
            raise BadRequestError("pivot needs at least one measure.")
        measure = req.measures[0]
        keep = list(dict.fromkeys([*req.rows, *req.columns, measure.column]))
        long = (
            self._scan(req.path)
            .select(keep)
            .group_by([*req.rows, *req.columns])
            .agg(_measure_expr(measure).alias(measure.column))
            .collect()
        )
        wide = long.pivot(
            on=req.columns, index=req.rows, values=measure.column, aggregate_function="first",
        )
        return PivotResult(
            row_count=wide.height,
            col_count=len(wide.columns),
            data=wide.to_dicts(),
        )

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        model_fn, model_used = _resolve_model(req.model)
        cols = list(dict.fromkeys([c for c in (req.x, req.column, req.group) if c]))
        df = self._scan(req.path).select(cols).collect()

        groups: list[tuple[str | None, pl.DataFrame]] = (
            [(None, df)] if not req.group
            else [(str(g), df.filter(pl.col(req.group) == g).sort(req.x))
                  for g in df[req.group].unique().sort().to_list()]
        )

        series_out: list[ForecastSeries] = []
        for name, gdf in groups:
            gdf = gdf.sort(req.x)
            feats, target, x_vals = _engineer(gdf, req)
            n = feats.shape[0]
            if n < 10:
                raise BadRequestError(f"group {name!r} has too few rows ({n}) to forecast.")
            split = int(n * 0.8)
            model = model_fn()
            model.fit(feats[:split], target[:split])
            import numpy as np

            pred_hold = model.predict(feats[split:])
            rmse = float(np.sqrt(np.mean((pred_hold - target[split:]) ** 2)))

            # Roll the engineered features forward horizon steps.
            x_future, y_pred = _project(model, gdf, req, x_vals, target)
            series_out.append(ForecastSeries(
                group=name, rmse=round(rmse, 6), x_future=x_future, y_pred=y_pred,
            ))
        return ForecastResult(model_used=model_used, series=series_out)


def _measure_expr(m) -> pl.Expr:
    fn = _AGG.get(m.agg)
    if fn is None:
        raise BadRequestError(
            f"Unknown agg {m.agg!r}. Valid: {', '.join(sorted(_AGG))}."
        )
    return fn(pl.col(m.column)).alias(f"{m.column}_{m.agg}")


def _resolve_model(name: str):
    """Return ``(factory, model_used)`` honoring the requested backend, falling
    back xgboost → gbr → ridge as availability dictates."""
    name = (name or "ridge").lower()
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor

            return (lambda: XGBRegressor(n_estimators=120, max_depth=4, n_jobs=-1, verbosity=0)), "xgboost"
        except ImportError:
            name = "gbr"
    if name == "gbr":
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            return (lambda: GradientBoostingRegressor(n_estimators=120, max_depth=3)), "gbr"
        except ImportError:
            name = "ridge"
    from sklearn.linear_model import Ridge

    return (lambda: Ridge(alpha=1.0)), "ridge"


def _engineer(gdf: pl.DataFrame, req: ForecastRequest):
    """Build (features, target, x_values) — lag features + optional seasonality."""
    import numpy as np

    x = gdf[req.x].cast(pl.Float64).to_numpy()
    y = gdf[req.column].cast(pl.Float64).to_numpy()
    feats = [x]
    lags = (1, 2, 3, 24) if req.period else (1, 2, 3)
    for lag in lags:
        shifted = np.empty_like(y)
        shifted[:lag] = y[0]
        shifted[lag:] = y[:-lag]
        feats.append(shifted)
    if req.period:
        feats.append(np.sin(2 * np.pi * x / req.period))
        feats.append(np.cos(2 * np.pi * x / req.period))
    return np.column_stack(feats), y, x


def _project(model, gdf: pl.DataFrame, req: ForecastRequest, x_vals, target):
    import numpy as np

    step = float(x_vals[-1] - x_vals[-2]) if len(x_vals) > 1 else 1.0
    history = list(target)
    x_future: list[float] = []
    y_pred: list[float] = []
    last_x = float(x_vals[-1])
    for _ in range(req.horizon):
        last_x += step
        row = [last_x]
        for lag in ((1, 2, 3, 24) if req.period else (1, 2, 3)):
            row.append(history[-lag] if len(history) >= lag else history[0])
        if req.period:
            row.append(float(np.sin(2 * np.pi * last_x / req.period)))
            row.append(float(np.cos(2 * np.pi * last_x / req.period)))
        yhat = float(model.predict(np.array([row]))[0])
        history.append(yhat)
        x_future.append(last_x)
        y_pred.append(round(yhat, 6))
    return x_future, y_pred
