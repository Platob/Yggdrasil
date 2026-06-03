"""Trading analytics service — technical indicators, correlation, portfolio, VaR, signals.

Mirrors the AnalysisService layout: a polars **lazy** scan loads only the price
column(s) a query touches, with parquet/csv/ndjson pushdown and a bounded read
for formats polars can't scan. Indicator math runs in NumPy. CPU-bound work is
dispatched off the event loop with ``run_in_threadpool``.
"""
from __future__ import annotations

import math
import os
from functools import partial
from typing import Any

import numpy as np
import polars as pl
from fastapi.concurrency import run_in_threadpool

from yggdrasil.data.options import CastOptions
from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.path import Path as YggPath

from ...config import Settings
from ...exceptions import ForbiddenError, NotFoundError
from ..schemas.trading import (
    CorrelationRequest,
    CorrelationResult,
    IndicatorRequest,
    IndicatorResult,
    PortfolioMetrics,
    PortfolioRequest,
    PortfolioResult,
    SignalRequest,
    SignalResult,
    TradeSignal,
    VaRRequest,
    VaRResult,
)
from .fs import FsService


def _safe(v):
    """JSON-safe scalar: drop NaN/inf, stringify anything exotic."""
    if v is None:
        return None
    if isinstance(v, bool) or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    return str(v)


def _safe_list(arr) -> list:
    """Convert a numpy array to a JSON-safe Python list (NaN/Inf → None)."""
    if isinstance(arr, np.ndarray):
        # Vectorised path: replace NaN/Inf with None in one pass.
        finite = np.isfinite(arr)
        out: list = [None] * len(arr)
        for i, (f, v) in enumerate(zip(finite, arr)):
            if f:
                out[i] = float(v)
        return out
    return [_safe(v) for v in arr]


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI — uses scipy.signal.lfilter for the smoothed avg-gain/loss."""
    n = len(prices)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    delta = np.diff(prices.astype(float))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    ag_seed = float(np.mean(gain[:period]))
    al_seed = float(np.mean(loss[:period]))
    # Wilder smoothing: process gain[period:] (the n-1-period deltas after the seed window)
    # rsi[period+1..n-1] ← n-1-period values
    try:
        from scipy.signal import lfilter
        w = 1.0 - alpha
        ag_arr = lfilter([alpha], [1.0, -w], gain[period:], zi=[ag_seed])[0]
        al_arr = lfilter([alpha], [1.0, -w], loss[period:], zi=[al_seed])[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(al_arr > 0, ag_arr / al_arr, 100.0)
        rsi[period + 1:] = 100.0 - 100.0 / (1.0 + rs)
    except ImportError:
        ag, al = ag_seed, al_seed
        w = 1.0 - alpha
        for i in range(period, n - 1):
            ag = ag * w + gain[i] * alpha
            al = al * w + loss[i] * alpha
            rs = ag / al if al > 0 else 100.0
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average — uses scipy.signal.lfilter when available for
    a fully vectorised O(n) scan; falls back to a Numba-free Python loop."""
    k = 2.0 / (span + 1)
    p = prices.astype(float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    if len(valid) == 0:
        return out
    start = int(valid[0])
    try:
        from scipy.signal import lfilter
        # lfilter implements: y[i] = k*x[i] + (1-k)*y[i-1] exactly
        segment = p[start:]
        zi = np.array([p[start]])
        filtered, _ = lfilter([k], [1.0, -(1.0 - k)], segment, zi=zi)
        out[start:] = filtered
    except ImportError:
        prev = p[start]
        out[start] = prev
        w = 1.0 - k
        for i in range(start + 1, len(p)):
            if not np.isnan(p[i]):
                prev = p[i] * k + prev * w
                out[i] = prev
    return out


def _macd(prices: np.ndarray, fast=12, slow=26, signal=9):
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(prices: np.ndarray, period=20, std_mult=2.0):
    """Bollinger Bands — uses Polars native rolling_mean/rolling_std (C-level)."""
    p = pl.Series(prices.astype(float))
    mid_s = p.rolling_mean(window_size=period)
    std_s = p.rolling_std(window_size=period)
    mid = mid_s.to_numpy()
    std = std_s.to_numpy()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> np.ndarray:
    """Average True Range — vectorised TR then Wilder smoothing."""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < 2:
        return atr
    h = high.astype(float)
    lo = low.astype(float)
    cl = close.astype(float)
    # Vectorised True Range
    prev_close = np.concatenate([[h[0] - lo[0]], cl[:-1]])
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
    tr[0] = h[0] - lo[0]
    atr[period - 1] = float(np.mean(tr[:period]))
    # Wilder smoothing (RMA)
    w = (period - 1) / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * w + tr[i] / period
    return atr


class TradingService:
    def __init__(self, settings: Settings, *, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

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
                table = bio.read_arrow_table(
                    options=CastOptions(row_limit=self.settings.analysis_max_rows + 1)
                )
            return pl.from_arrow(table).lazy()
        except (NotFoundError, ForbiddenError, BadRequestError):
            raise
        except Exception as exc:
            raise BadRequestError(f"Cannot read {path!r} as a table: {exc}")

    # -- async entrypoints --------------------------------------------------

    async def indicators(self, req: IndicatorRequest) -> IndicatorResult:
        return await run_in_threadpool(partial(self._indicators, req))

    async def correlation(self, req: CorrelationRequest) -> CorrelationResult:
        return await run_in_threadpool(partial(self._correlation, req))

    async def portfolio(self, req: PortfolioRequest) -> PortfolioResult:
        return await run_in_threadpool(partial(self._portfolio, req))

    async def var(self, req: VaRRequest) -> VaRResult:
        return await run_in_threadpool(partial(self._var, req))

    async def signals(self, req: SignalRequest) -> SignalResult:
        return await run_in_threadpool(partial(self._signals, req))

    # -- shared loader ------------------------------------------------------

    def _load_series(
        self, path: str, column: str, x: str | None, limit: int
    ) -> tuple[np.ndarray, list, int]:
        """Load a price series, return (values_array, index_list, total_rows)."""
        lf = self._frame(path)
        cols = set(lf.collect_schema().names())
        if column not in cols:
            raise BadRequestError(
                f"column {column!r} not in {path!r}; have {sorted(cols)[:20]}"
            )
        keep = list(dict.fromkeys([column] + ([x] if x and x in cols else [])))
        plan = lf.select(keep)
        if x and x in cols:
            plan = plan.sort(x)
        total = plan.select(pl.len()).collect(engine="streaming").item()
        df = plan.head(limit).collect(engine="streaming")
        prices = (
            df[column]
            .cast(pl.Float64, strict=False)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        index = df[x].to_list() if (x and x in df.columns) else list(range(len(prices)))
        return prices, index, total

    # -- indicators ---------------------------------------------------------

    def _indicators(self, req: IndicatorRequest) -> IndicatorResult:
        inds = {i.lower() for i in req.indicators}
        prices, index, total = self._load_series(req.path, req.column, req.x, req.limit)
        truncated = total > req.limit

        result = IndicatorResult(
            node_id=self.settings.node_id,
            path=req.path,
            column=req.column,
            index=[_safe(v) for v in index],
            value=_safe_list(prices),
            source_rows=total,
            truncated=truncated,
        )
        if "rsi" in inds:
            result.rsi = _safe_list(_rsi(prices, req.rsi_period))
        if "macd" in inds:
            m, s, h = _macd(prices, req.macd_fast, req.macd_slow, req.macd_signal)
            result.macd = _safe_list(m)
            result.macd_signal = _safe_list(s)
            result.macd_hist = _safe_list(h)
        if "bb" in inds:
            u, mid, lo = _bollinger(prices, req.bb_period, req.bb_std)
            result.bb_upper = _safe_list(u)
            result.bb_mid = _safe_list(mid)
            result.bb_lower = _safe_list(lo)
        if "atr" in inds and req.high and req.low:
            lf = self._frame(req.path)
            cols = set(lf.collect_schema().names())
            if req.high in cols and req.low in cols:
                keep = list(dict.fromkeys(
                    [req.column, req.high, req.low]
                    + ([req.x] if req.x and req.x in cols else [])
                ))
                plan = lf.select(keep)
                if req.x and req.x in cols:
                    plan = plan.sort(req.x)
                df = plan.head(req.limit).collect(engine="streaming")
                hi = df[req.high].cast(pl.Float64, strict=False).to_numpy()
                lo_arr = df[req.low].cast(pl.Float64, strict=False).to_numpy()
                cl = df[req.column].cast(pl.Float64, strict=False).to_numpy()
                result.atr = _safe_list(_atr(hi, lo_arr, cl, req.atr_period))
        return result

    # -- correlation --------------------------------------------------------

    def _correlation(self, req: CorrelationRequest) -> CorrelationResult:
        if len(req.paths) < 2:
            raise BadRequestError("Need at least 2 paths for correlation")
        series_list = []
        rows_list = []
        labels = list(req.labels)
        for i, path in enumerate(req.paths):
            prices, _, total = self._load_series(path, req.column, None, req.limit)
            series_list.append(prices)
            rows_list.append(total)
            if i >= len(labels):
                labels.append(os.path.basename(path))

        # Align all series to the same length (trim to shortest, trailing rows).
        min_len = min(len(s) for s in series_list)
        series_list = [s[-min_len:] for s in series_list]

        # Returns for correlation.
        rets = []
        for s in series_list:
            r = np.diff(s) / np.where(s[:-1] != 0, s[:-1], np.nan)
            rets.append(r)

        n = len(rets)
        matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ri, rj = rets[i], rets[j]
                mask = ~(np.isnan(ri) | np.isnan(rj))
                if mask.sum() < 3:
                    matrix[i][j] = None
                    continue
                if req.method == "spearman":
                    try:
                        from scipy.stats import spearmanr

                        corr, _ = spearmanr(ri[mask], rj[mask])
                        matrix[i][j] = float(corr) if np.isfinite(corr) else None
                    except ImportError:
                        # No scipy: rank-transform then Pearson — same result.
                        a = ri[mask].argsort().argsort().astype(float)
                        b = rj[mask].argsort().argsort().astype(float)
                        c = float(np.corrcoef(a, b)[0, 1])
                        matrix[i][j] = c if math.isfinite(c) else None
                    except Exception:
                        matrix[i][j] = None
                else:
                    try:
                        corr = float(np.corrcoef(ri[mask], rj[mask])[0, 1])
                        matrix[i][j] = corr if math.isfinite(corr) else None
                    except Exception:
                        matrix[i][j] = None

        return CorrelationResult(
            node_id=self.settings.node_id,
            labels=labels[:n],
            method=req.method,
            matrix=matrix,
            source_rows=rows_list,
        )

    # -- portfolio ----------------------------------------------------------

    def _portfolio(self, req: PortfolioRequest) -> PortfolioResult:
        if len(req.assets) < 2:
            raise BadRequestError("Portfolio needs at least 2 assets")
        series_list = []
        rows_list = []
        labels = []
        for asset in req.assets:
            prices, _, total = self._load_series(asset.path, asset.column, None, req.limit)
            series_list.append(prices)
            rows_list.append(total)
            labels.append(asset.label or asset.column)

        min_len = min(len(s) for s in series_list)
        series_list = [s[-min_len:] for s in series_list]

        raw_w = np.array([a.weight for a in req.assets[:len(series_list)]], dtype=float)
        total_w = raw_w.sum()
        if total_w == 0:
            raise BadRequestError("Portfolio weights sum to zero; give at least one nonzero weight")
        weights = raw_w / total_w

        # Returns per asset (rows × assets after transpose).
        rets_matrix = np.vstack([
            np.diff(s) / np.where(s[:-1] != 0, s[:-1], np.nan) for s in series_list
        ])
        # Portfolio return = weighted sum across assets.
        port_rets = (weights[:, None] * np.nan_to_num(rets_matrix)).sum(axis=0)

        cum = np.cumprod(1.0 + port_rets) - 1.0
        equity = 1.0 + cum
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0

        # Equal-weight benchmark for beta/alpha.
        bench_rets = np.nanmean(rets_matrix, axis=0)
        bench_rets = np.nan_to_num(bench_rets)

        ind_rets = []
        for s in series_list:
            r = np.diff(s) / np.where(s[:-1] != 0, s[:-1], np.nan)
            cr = list(np.cumprod(1.0 + np.nan_to_num(r)) - 1.0)
            ind_rets.append([_safe(v) for v in cr])

        index = list(range(min_len - 1))

        ppy = max(1, req.periods_per_year)
        r = port_rets[~np.isnan(port_rets)]
        metrics = PortfolioMetrics()
        if len(r) > 1:
            last_v = float(equity[-1])
            total_ret = float(last_v - 1.0)
            years = len(r) / ppy
            cagr = float(last_v ** (1.0 / years) - 1.0) if (years > 0 and last_v > 0) else None
            mean_r = float(np.mean(r))
            std_r = float(np.std(r, ddof=1))
            ann_ret = mean_r * ppy
            ann_vol = std_r * math.sqrt(ppy)
            rf_per = req.risk_free / ppy
            downside = r[r < rf_per]
            down_dev = (
                float(np.sqrt(np.mean((downside - rf_per) ** 2))) * math.sqrt(ppy)
                if len(downside) else 0.0
            )
            max_dd = float(drawdown.min())
            sharpe = (ann_ret - req.risk_free) / ann_vol if ann_vol else None
            sortino = (ann_ret - req.risk_free) / down_dev if down_dev else None

            # Beta/alpha of the portfolio vs the equal-weight benchmark.
            beta = alpha = None
            if len(bench_rets) > 1:
                var_b = float(np.var(bench_rets, ddof=1))
                if var_b > 0:
                    cov = float(np.cov(port_rets, bench_rets, ddof=1)[0, 1])
                    beta = cov / var_b
                    alpha = (ann_ret - req.risk_free) - beta * (
                        float(np.mean(bench_rets)) * ppy - req.risk_free
                    )

            metrics = PortfolioMetrics(
                total_return=total_ret,
                cagr=cagr,
                ann_return=ann_ret,
                ann_volatility=ann_vol,
                sharpe=sharpe,
                sortino=sortino,
                max_drawdown=max_dd,
                calmar=(cagr / abs(max_dd)) if (cagr is not None and max_dd) else None,
                beta=_safe(beta),
                alpha=_safe(alpha),
            )

        n = len(series_list)
        corr_mat: list[list[float | None]] = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                ri = np.diff(series_list[i]) / np.where(
                    series_list[i][:-1] != 0, series_list[i][:-1], np.nan
                )
                rj = np.diff(series_list[j]) / np.where(
                    series_list[j][:-1] != 0, series_list[j][:-1], np.nan
                )
                mask = ~(np.isnan(ri) | np.isnan(rj))
                if mask.sum() < 3:
                    continue
                try:
                    c = float(np.corrcoef(ri[mask], rj[mask])[0, 1])
                    corr_mat[i][j] = c if math.isfinite(c) else None
                except Exception:
                    pass

        return PortfolioResult(
            node_id=self.settings.node_id,
            labels=labels,
            weights=[float(w) for w in weights],
            index=index,
            portfolio_value=_safe_list(equity),
            drawdown=_safe_list(drawdown),
            individual_returns=ind_rets,
            metrics=metrics,
            correlation_matrix=corr_mat,
            source_rows=min(rows_list),
        )

    # -- value at risk ------------------------------------------------------

    def _var(self, req: VaRRequest) -> VaRResult:
        prices, _, total = self._load_series(req.path, req.column, None, req.limit)
        if len(prices) < 10:
            raise BadRequestError("Need at least 10 price observations for VaR")
        rets = np.diff(prices) / np.where(prices[:-1] != 0, prices[:-1], np.nan)
        rets = rets[~np.isnan(rets)]
        ppy = max(1, req.periods_per_year)
        c = max(0.01, min(0.9999, req.confidence))
        h = max(1, req.horizon)
        ann_vol = float(np.std(rets, ddof=1)) * math.sqrt(ppy) if len(rets) > 1 else None

        var_val = cvar_val = None
        if req.method == "historical":
            sorted_r = np.sort(rets)
            idx = int(np.floor((1 - c) * len(sorted_r)))
            idx = max(0, min(idx, len(sorted_r) - 1))
            var_val = float(sorted_r[idx]) * math.sqrt(h)
            cvar_val = (
                float(np.mean(sorted_r[:idx + 1])) * math.sqrt(h) if idx > 0 else var_val
            )
        elif req.method == "parametric":
            try:
                from scipy.stats import norm

                mu = float(np.mean(rets))
                sigma = float(np.std(rets, ddof=1))
                z = float(norm.ppf(1 - c))
                var_val = (mu + z * sigma) * math.sqrt(h)
                cvar_val = (mu - sigma * float(norm.pdf(norm.ppf(c))) / (1 - c)) * math.sqrt(h)
            except ImportError:
                raise BadRequestError(
                    "scipy is required for parametric VaR; install scipy or use method='historical'"
                )
        elif req.method == "cornish_fisher":
            try:
                from scipy.stats import kurtosis, norm, skew
            except ImportError:
                raise BadRequestError(
                    "scipy is required for cornish_fisher VaR; install scipy or use method='historical'"
                )
            mu = float(np.mean(rets))
            sigma = float(np.std(rets, ddof=1))
            s = float(skew(rets))
            k = float(kurtosis(rets))
            z = float(norm.ppf(1 - c))
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            var_val = (mu + z_cf * sigma) * math.sqrt(h)
            sorted_r = np.sort(rets)
            idx = int(np.floor((1 - c) * len(sorted_r)))
            cvar_val = float(np.mean(sorted_r[:max(1, idx)])) * math.sqrt(h)
        else:
            raise BadRequestError(
                f"unknown VaR method {req.method!r}; one of "
                "['historical', 'parametric', 'cornish_fisher']"
            )

        def _pct(v):
            return round(v * 100, 4) if v is not None and math.isfinite(v) else None

        return VaRResult(
            node_id=self.settings.node_id,
            path=req.path,
            column=req.column,
            method=req.method,
            confidence=c,
            horizon=h,
            var=_safe(var_val),
            cvar=_safe(cvar_val),
            var_pct=_pct(var_val),
            cvar_pct=_pct(cvar_val),
            ann_volatility=_safe(ann_vol),
            source_rows=total,
        )

    # -- trade signals ------------------------------------------------------

    def _signals(self, req: SignalRequest) -> SignalResult:
        prices, index, total = self._load_series(req.path, req.column, req.x, req.limit)
        n = len(prices)
        rsi_arr = _rsi(prices, req.rsi_period)
        _, _, macd_hist = _macd(prices, req.macd_fast, req.macd_slow, req.macd_signal_period)
        bb_upper, bb_mid, bb_lower = _bollinger(prices, req.bb_period, req.bb_std)

        signals: list[TradeSignal] = []
        for i in range(1, n):
            reasons = []
            score = 0.0
            count = 0

            rsi_v = rsi_arr[i] if not np.isnan(rsi_arr[i]) else None
            mh = macd_hist[i] if not np.isnan(macd_hist[i]) else None
            mh_prev = macd_hist[i - 1] if i > 0 and not np.isnan(macd_hist[i - 1]) else None

            bb_pos = None
            if not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]):
                rng = bb_upper[i] - bb_lower[i]
                bb_pos = float((prices[i] - bb_lower[i]) / rng) if rng > 0 else 0.5

            if rsi_v is not None:
                if rsi_v < 30:
                    reasons.append("RSI oversold")
                    score += 1.0
                    count += 1
                elif rsi_v > 70:
                    reasons.append("RSI overbought")
                    score -= 1.0
                    count += 1
                else:
                    count += 1

            if mh is not None and mh_prev is not None:
                if mh > 0 and mh_prev <= 0:
                    reasons.append("MACD bullish crossover")
                    score += 1.0
                    count += 1
                elif mh < 0 and mh_prev >= 0:
                    reasons.append("MACD bearish crossover")
                    score -= 1.0
                    count += 1
                else:
                    count += 1

            if bb_pos is not None:
                if bb_pos < 0.1:
                    reasons.append("Price at lower Bollinger Band")
                    score += 0.5
                    count += 1
                elif bb_pos > 0.9:
                    reasons.append("Price at upper Bollinger Band")
                    score -= 0.5
                    count += 1
                else:
                    count += 1

            if count == 0:
                continue

            norm_score = score / count
            if norm_score > 0.25:
                action = "BUY"
            elif norm_score < -0.25:
                action = "SELL"
            else:
                action = "HOLD"

            # Only emit non-HOLD signals OR the last bar (so the UI always has a
            # current read on the latest position).
            if action != "HOLD" or i == n - 1:
                signals.append(TradeSignal(
                    index=_safe(index[i]),
                    action=action,
                    strength=abs(norm_score),
                    reasons=reasons,
                    rsi=_safe(rsi_v),
                    macd_hist=_safe(mh),
                    bb_position=_safe(bb_pos),
                ))

        buy_count = sum(1 for s in signals if s.action == "BUY")
        sell_count = sum(1 for s in signals if s.action == "SELL")
        last_action = signals[-1].action if signals else "HOLD"

        return SignalResult(
            node_id=self.settings.node_id,
            path=req.path,
            column=req.column,
            signals=signals,
            last_action=last_action,
            buy_count=buy_count,
            sell_count=sell_count,
            source_rows=total,
        )
