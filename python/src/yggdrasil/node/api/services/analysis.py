"""Analysis service — polars lazy scans with projection pushdown + sklearn forecast.

Aggregate/pivot scan parquet lazily and select only the columns they touch, so
a query over 2 of 30 columns reads only those 2. Series/OHLC downsample a single
column into a bounded number of points/bars. Forecast engineers periodic time
features and fits ridge / gradient boosting / xgboost (falling back to ridge).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from yggdrasil.node.api.schemas.analysis import (
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
from yggdrasil.node.api.services.fs import FsService

_AGG_EXPR = {
    "mean": lambda c: c.mean(),
    "sum": lambda c: c.sum(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
    "count": lambda c: c.count(),
}


class AnalysisService:
    """Lazy-scan analytics rooted at ``settings.node_home``."""

    def __init__(self, settings: object, fs: FsService) -> None:
        self._root = Path(settings.node_home)
        self._fs = fs

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        lf = pl.scan_parquet(str(self._root / req.path))
        cols_needed = req.group_by + [m.column for m in req.measures]
        lf = lf.select(cols_needed)
        agg_exprs = [
            _AGG_EXPR[m.agg](pl.col(m.column)).alias(f"{m.column}_{m.agg}")
            for m in req.measures
        ]
        result = lf.group_by(req.group_by).agg(agg_exprs).collect()
        return AggregateResult(group_count=len(result), data=result.to_dicts())

    async def series(self, req: SeriesRequest) -> SeriesResult:
        df = pl.scan_parquet(str(self._root / req.path)).select([req.column]).collect()
        n = len(df)
        step = max(1, n // req.points)
        downsampled = df[::step][req.column].to_list()
        x = [float(i * step) for i in range(len(downsampled))]
        return SeriesResult(x=x, y=downsampled)

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        df = pl.scan_parquet(str(self._root / req.path)).select([req.column]).collect()
        n = len(df)
        bucket_size = max(1, n // req.buckets)
        vals = df[req.column].to_list()
        opens, highs, lows, closes, timestamps = [], [], [], [], []
        i = 0
        while i < n:
            chunk = vals[i:i + bucket_size]
            opens.append(float(chunk[0]))
            highs.append(float(max(chunk)))
            lows.append(float(min(chunk)))
            closes.append(float(chunk[-1]))
            timestamps.append(i)
            i += bucket_size
        return OhlcResult(
            bars=len(opens), open=opens, high=highs, low=lows, close=closes, timestamps=timestamps
        )

    async def pivot(self, req: PivotRequest) -> PivotResult:
        all_cols = req.rows + req.columns + [m.column for m in req.measures]
        lf = pl.scan_parquet(str(self._root / req.path)).select(all_cols)
        agg_exprs = [pl.col(m.column).sum().alias(m.column) for m in req.measures]
        result = lf.group_by(req.rows + req.columns).agg(agg_exprs).collect()
        return PivotResult(
            row_count=len(result), col_count=len(result.columns), data=result.to_dicts()
        )

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        import numpy as np

        cols = [req.x, req.column] + ([req.group] if req.group else [])
        df = pl.scan_parquet(str(self._root / req.path)).select(cols).collect()
        groups = df[req.group].unique().to_list() if req.group else [None]

        model_used = req.model
        series_list: list[ForecastSeries] = []
        period = req.period

        for grp in groups:
            subset = df.filter(pl.col(req.group) == grp) if grp is not None else df
            x_raw = np.asarray(subset[req.x].to_list(), dtype=float)
            y = np.asarray(subset[req.column].to_list(), dtype=float)
            x_feats = np.column_stack(
                [x_raw, np.sin(2 * np.pi * x_raw / period), np.cos(2 * np.pi * x_raw / period)]
            )

            try:
                if req.model == "xgboost":
                    from xgboost import XGBRegressor

                    m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
                    model_used = "xgboost"
                elif req.model == "gbr":
                    from sklearn.ensemble import GradientBoostingRegressor

                    m = GradientBoostingRegressor(n_estimators=100, max_depth=3)
                    model_used = "gbr"
                else:
                    from sklearn.linear_model import Ridge

                    m = Ridge(alpha=1.0)
                    model_used = "ridge"
            except ImportError:
                from sklearn.linear_model import Ridge

                m = Ridge(alpha=1.0)
                model_used = "ridge"

            m.fit(x_feats, y)

            last_x = x_raw[-1]
            future_x = np.arange(last_x + 1, last_x + req.horizon + 1)
            future_feats = np.column_stack(
                [
                    future_x,
                    np.sin(2 * np.pi * future_x / period),
                    np.cos(2 * np.pi * future_x / period),
                ]
            )
            predictions = m.predict(future_feats).tolist()
            rmse = float(np.sqrt(np.mean((y - m.predict(x_feats)) ** 2)))
            series_list.append(
                ForecastSeries(
                    group=str(grp) if grp is not None else None,
                    values=predictions,
                    rmse=rmse,
                )
            )

        return ForecastResult(model_used=model_used, series=series_list)

    async def indicators(self, req: object) -> dict:
        """Compute SMA, RSI, Bollinger Bands on a parquet column using polars expressions."""
        import polars as pl

        column: str = req.column  # type: ignore[attr-defined]
        sma_periods: list[int] = req.sma_periods  # type: ignore[attr-defined]
        rsi_period: int = req.rsi_period  # type: ignore[attr-defined]
        bb_period: int = req.bbands_period  # type: ignore[attr-defined]
        bb_std: float = req.bbands_stddev  # type: ignore[attr-defined]

        df = pl.scan_parquet(str(self._root / req.path)).select([column]).collect()  # type: ignore[attr-defined]
        col = pl.col(column)

        exprs = [col]
        for period in sma_periods:
            exprs.append(col.rolling_mean(window_size=period).alias(f"sma_{period}"))

        # RSI: rolling gain / loss averages
        diff = df[column].diff()
        gain = diff.clip(lower_bound=0)
        loss = (-diff).clip(lower_bound=0)
        avg_gain = gain.rolling_mean(window_size=rsi_period)
        avg_loss = loss.rolling_mean(window_size=rsi_period)
        rs = avg_gain / avg_loss.replace(0, None)
        rsi_vals = (100 - 100 / (1 + rs)).to_list()

        # Bollinger Bands
        exprs.append(col.rolling_mean(window_size=bb_period).alias("bb_mid"))
        exprs.append(
            (col.rolling_mean(window_size=bb_period) + bb_std * col.rolling_std(window_size=bb_period))
            .alias("bb_upper")
        )
        exprs.append(
            (col.rolling_mean(window_size=bb_period) - bb_std * col.rolling_std(window_size=bb_period))
            .alias("bb_lower")
        )

        result = df.with_columns(exprs)

        return {
            "values": result[column].to_list(),
            "sma": {str(p): result[f"sma_{p}"].to_list() for p in sma_periods},
            "rsi": rsi_vals,
            "bollinger": {
                "mid": result["bb_mid"].to_list(),
                "upper": result["bb_upper"].to_list(),
                "lower": result["bb_lower"].to_list(),
            },
        }
