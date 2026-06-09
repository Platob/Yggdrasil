"""AnalysisService — lazy polars aggregation, series, OHLC, pivot,
forecast, finance risk metrics, technical indicators, signal detection.

All read paths use Polars lazy scan with projection pushdown so only the
touched columns are read from disk.  The heavy tabular result is never
materialized beyond what the response needs.

Finance endpoint: ``GET /api/v2/analysis/finance``
Returns ``ema[]``, ``drawdown[]``, and ``metrics{}``
(total_return, cagr, ann_return, ann_volatility, sharpe, sortino,
max_drawdown, calmar).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from yggdrasil.node.api.schemas.analysis import (
    AggMeasure,
    AggregateRequest,
    AggregateResult,
    FinanceMetrics,
    FinanceRequest,
    FinanceResult,
    ForecastRequest,
    ForecastResult,
    ForecastSeries,
    IndicatorRequest,
    IndicatorResult,
    OhlcRequest,
    OhlcResult,
    PivotRequest,
    PivotResult,
    SeriesRequest,
    SeriesResult,
    Signal,
    SignalRequest,
    SignalResult,
)

_TABULAR_EXTS = {".parquet", ".pq", ".csv", ".ndjson", ".jsonl", ".arrow", ".ipc", ".feather"}


def _lazy(path: Path) -> pl.LazyFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pl.scan_parquet(str(path))
    if suf == ".csv":
        return pl.scan_csv(str(path))
    if suf in (".ndjson", ".jsonl"):
        return pl.scan_ndjson(str(path))
    # arrow/feather: eager, wrap
    import pyarrow.ipc as ipc
    return pl.from_arrow(ipc.open_file(str(path)).read_all()).lazy()


class AnalysisService:
    def __init__(self, settings: Any, fs: Any) -> None:
        self.settings = settings
        self.fs = fs
        self.root = Path(settings.node_home)

    def _resolve(self, rel: str) -> Path:
        target = (self.root / rel.lstrip("/")).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"path {rel!r} escapes node home {root}.")
        return target

    async def aggregate(self, req: AggregateRequest) -> AggregateResult:
        path = self._resolve(req.path)
        cols = list({m.column for m in req.measures} | set(req.group_by))
        lf = _lazy(path).select(cols)
        aggs = [
            getattr(pl.col(m.column), m.agg)().alias(f"{m.agg}_{m.column}")
            for m in req.measures
        ]
        df = lf.group_by(req.group_by).agg(aggs).collect()
        return AggregateResult(group_count=len(df), data=df.to_dicts())

    async def series(self, req: SeriesRequest) -> SeriesResult:
        path = self._resolve(req.path)
        lf = _lazy(path).select(pl.col(req.column).cast(pl.Float64))
        df = lf.collect()
        total = len(df)
        if total <= req.points:
            ys = df[req.column].to_list()
        else:
            step = total / req.points
            idxs = [int(i * step) for i in range(req.points)]
            ys = [df[req.column][i] for i in idxs]
        xs = list(range(len(ys)))
        return SeriesResult(x=xs, y=ys)

    async def ohlc(self, req: OhlcRequest) -> OhlcResult:
        path = self._resolve(req.path)
        df = _lazy(path).select(pl.col(req.column).cast(pl.Float64)).collect()
        vals = df[req.column].to_list()
        n = len(vals)
        if n == 0:
            return OhlcResult(bars=0, data=[])
        size = max(1, n // req.buckets)
        bars = []
        for i in range(0, n, size):
            chunk = vals[i:i + size]
            bars.append({"open": chunk[0], "high": max(chunk), "low": min(chunk),
                         "close": chunk[-1], "i": i})
        return OhlcResult(bars=len(bars), data=bars)

    async def pivot(self, req: PivotRequest) -> PivotResult:
        path = self._resolve(req.path)
        cols = list({m.column for m in req.measures} | set(req.rows) | set(req.columns))
        lf = _lazy(path).select(cols)
        aggs = [
            getattr(pl.col(m.column), m.agg)().alias(f"{m.agg}_{m.column}")
            for m in req.measures
        ]
        df = lf.group_by(req.rows + req.columns).agg(aggs).collect()
        data = df.to_dicts()
        return PivotResult(row_count=len(df), col_count=len(df.columns), data=data)

    async def forecast(self, req: ForecastRequest) -> ForecastResult:
        path = self._resolve(req.path)
        cols = [req.x, req.column]
        if req.group:
            cols.append(req.group)
        df = _lazy(path).select([pl.col(c) for c in cols]).collect()

        def _fit(x_arr: list, y_arr: list, model: str):
            import numpy as np
            X_raw = np.array(x_arr, dtype=float).reshape(-1, 1)
            y = np.array(y_arr)
            # Normalise the trend features (linear + quadratic) to [0,1] so
            # Ridge gets a well-conditioned matrix.  The sin seasonality term
            # uses the original x scale so period has the right meaning.
            x_min, x_range = X_raw.min(), X_raw.max() - X_raw.min()
            X = (X_raw - x_min) / (x_range if x_range else 1.0)
            feats = np.hstack([X, X ** 2, np.sin(2 * math.pi * X_raw / max(req.period, 1))])
            if model == "xgboost":
                try:
                    from xgboost import XGBRegressor
                    m = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                     verbosity=0, random_state=42)
                    m.fit(feats, y)
                    return m, feats, "xgboost"
                except ImportError:
                    pass
            if model in ("gbr", "xgboost"):
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                              learning_rate=0.1, random_state=42)
                m.fit(feats, y)
                return m, feats, "gbr"
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
            m.fit(feats, y)
            return m, feats, "ridge"

        groups: dict[str, tuple[list, list]] = {}
        if req.group and req.group in df.columns:
            for g in df[req.group].unique().to_list():
                sub = df.filter(pl.col(req.group) == g)
                groups[str(g)] = (sub[req.x].to_list(), sub[req.column].to_list())
        else:
            groups[""] = (df[req.x].to_list(), df[req.column].to_list())

        series_list: list[ForecastSeries] = []
        model_used = req.model
        for g, (xs, ys) in groups.items():
            if not xs:
                continue
            import numpy as np
            m, _, mu = _fit(xs, ys, req.model)
            model_used = mu
            xs_arr = np.array(xs, dtype=float)
            x_min, x_range = xs_arr.min(), xs_arr.max() - xs_arr.min()
            _norm = lambda x: (x - x_min) / (x_range if x_range else 1.0)  # noqa: E731
            last_x = xs_arr.max()
            step = x_range / max(len(xs) - 1, 1) if len(xs) > 1 else 1.0
            fut_x_raw = np.array([last_x + step * (i + 1) for i in range(req.horizon)])
            fut_x = _norm(fut_x_raw).reshape(-1, 1)
            fut_feats = np.hstack([fut_x, fut_x ** 2,
                                   np.sin(2 * math.pi * fut_x_raw.reshape(-1, 1) / max(req.period, 1))])
            preds = m.predict(fut_feats).tolist()
            train_x = _norm(xs_arr).reshape(-1, 1)
            train_feats = np.hstack([
                train_x, train_x ** 2,
                np.sin(2 * math.pi * xs_arr.reshape(-1, 1) / max(req.period, 1)),
            ])
            res_sq = (np.array(ys) - m.predict(train_feats)) ** 2
            rmse = float(np.sqrt(res_sq.mean()))
            series_list.append(ForecastSeries(group=g, rmse=rmse, forecast=preds))

        return ForecastResult(model_used=model_used, series=series_list)

    async def finance(self, req: FinanceRequest) -> FinanceResult:
        """Compute EMA, drawdown curve, and risk metrics for a price series.

        Returns ema[], drawdown[], and metrics{} (total_return, cagr,
        ann_return, ann_volatility, sharpe, sortino, max_drawdown, calmar).
        """
        import numpy as np

        path = self._resolve(req.path)
        df = _lazy(path).select(pl.col(req.column).cast(pl.Float64)).collect()
        prices = np.array(df[req.column].to_list(), dtype=float)
        n = len(prices)
        if n < 2:
            zero_metrics = FinanceMetrics(
                total_return=0.0, cagr=0.0, ann_return=0.0, ann_volatility=0.0,
                sharpe=0.0, sortino=0.0, max_drawdown=0.0, calmar=0.0,
            )
            return FinanceResult(ema=prices.tolist(), drawdown=[0.0] * n, metrics=zero_metrics)

        # EMA (span=20, adjust=False — pandas-compatible formula)
        alpha = 2.0 / (20 + 1)
        ema = np.empty(n)
        ema[0] = prices[0]
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        # Drawdown: (price - running_max) / running_max
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / np.where(running_max != 0, running_max, 1.0)

        # Daily returns
        rets = np.diff(prices) / np.where(prices[:-1] != 0, prices[:-1], 1.0)

        trading_days = 252
        total_return = float((prices[-1] - prices[0]) / prices[0]) if prices[0] != 0 else 0.0
        years = max(n / trading_days, 1e-9)
        cagr = float((prices[-1] / prices[0]) ** (1.0 / years) - 1) if prices[0] > 0 else 0.0
        ann_return = float(np.mean(rets) * trading_days)
        ann_vol = float(np.std(rets, ddof=1) * math.sqrt(trading_days))
        rf = req.risk_free_rate / trading_days
        sharpe = float((np.mean(rets) - rf) / np.std(rets, ddof=1) * math.sqrt(trading_days)) \
            if np.std(rets, ddof=1) > 0 else 0.0
        neg_rets = rets[rets < rf]
        downside_std = float(np.std(neg_rets, ddof=1)) if len(neg_rets) > 1 else 0.0
        sortino = float((np.mean(rets) - rf) / downside_std * math.sqrt(trading_days)) \
            if downside_std > 0 else 0.0
        max_dd = float(np.min(drawdown))
        calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

        metrics = FinanceMetrics(
            total_return=round(total_return, 6),
            cagr=round(cagr, 6),
            ann_return=round(ann_return, 6),
            ann_volatility=round(ann_vol, 6),
            sharpe=round(sharpe, 4),
            sortino=round(sortino, 4),
            max_drawdown=round(max_dd, 6),
            calmar=round(calmar, 4),
        )
        return FinanceResult(
            ema=[round(v, 6) for v in ema.tolist()],
            drawdown=[round(v, 6) for v in drawdown.tolist()],
            metrics=metrics,
        )

    async def indicators(self, req: IndicatorRequest) -> IndicatorResult:
        path = self._resolve(req.path)
        df = _lazy(path).select([req.timestamp, req.column]).collect()
        prices = df[req.column].cast(pl.Float64).to_list()
        n = len(prices)
        result: dict[str, list] = {req.timestamp: df[req.timestamp].to_list()}

        def _ema(vals: list[float], span: int) -> list[float]:
            a = 2.0 / (span + 1)
            out = [vals[0]]
            for v in vals[1:]:
                out.append(a * v + (1 - a) * out[-1])
            return out

        for ind in req.indicators:
            low = ind.lower()
            if low == "rsi":
                gains = [max(0.0, prices[i] - prices[i - 1]) for i in range(1, n)]
                losses = [max(0.0, prices[i - 1] - prices[i]) for i in range(1, n)]
                def _rma(xs: list[float], p: int) -> list[float]:
                    out: list[float] = [sum(xs[:p]) / p]
                    for x in xs[p:]:
                        out.append((out[-1] * (p - 1) + x) / p)
                    return out
                rma_g = _rma(gains, 14)
                rma_l = _rma(losses, 14)
                rsi = [100 - 100 / (1 + g / l) if l > 0 else 100.0 for g, l in zip(rma_g, rma_l)]
                result["rsi"] = [None] * 14 + rsi
            elif low == "macd":
                fast = _ema(prices, 12)
                slow = _ema(prices, 26)
                macd_line = [f - s for f, s in zip(fast, slow)]
                signal = _ema(macd_line, 9)
                result["macd"] = macd_line
                result["macd_signal"] = signal
                result["macd_hist"] = [m - s for m, s in zip(macd_line, signal)]
            elif low.startswith("ema_"):
                span = int(low.split("_", 1)[1])
                result[f"ema_{span}"] = _ema(prices, span)
            elif low.startswith("sma_"):
                period = int(low.split("_", 1)[1])
                sma = [None] * (period - 1)
                for i in range(period - 1, n):
                    sma.append(sum(prices[i - period + 1:i + 1]) / period)
                result[f"sma_{period}"] = sma
            elif low == "bb":
                period = 20
                mid = [None] * (period - 1)
                upper = [None] * (period - 1)
                lower = [None] * (period - 1)
                for i in range(period - 1, n):
                    w = prices[i - period + 1:i + 1]
                    m = sum(w) / period
                    std = math.sqrt(sum((x - m) ** 2 for x in w) / period)
                    mid.append(m)
                    upper.append(m + 2 * std)
                    lower.append(m - 2 * std)
                result["bb_mid"] = mid
                result["bb_upper"] = upper
                result["bb_lower"] = lower

        df_out = pl.DataFrame(result)
        return IndicatorResult(rows=len(df_out), columns=df_out.columns, data=df_out.to_dicts())

    async def signals(self, req: SignalRequest) -> SignalResult:
        path = self._resolve(req.path)
        df = _lazy(path).select([req.timestamp_col, req.close_col]).collect()
        prices = df[req.close_col].cast(pl.Float64).to_list()
        ts = df[req.timestamp_col].to_list()
        n = len(prices)

        def _ema(vals: list[float], span: int) -> list[float]:
            a = 2.0 / (span + 1)
            out = [vals[0]]
            for v in vals[1:]:
                out.append(a * v + (1 - a) * out[-1])
            return out

        fast = _ema(prices, 50)
        slow = _ema(prices, 200)
        sigs: list[Signal] = []
        for i in range(1, n):
            if fast[i - 1] < slow[i - 1] and fast[i] >= slow[i]:
                sigs.append(Signal(kind="golden_cross", index=i,
                                   timestamp=ts[i], price=prices[i]))
            elif fast[i - 1] > slow[i - 1] and fast[i] <= slow[i]:
                sigs.append(Signal(kind="death_cross", index=i,
                                   timestamp=ts[i], price=prices[i]))
        return SignalResult(count=len(sigs), signals=sigs)
