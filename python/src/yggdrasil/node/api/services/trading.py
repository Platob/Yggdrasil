"""Technical-analysis + backtesting over node-local parquet/arrow files.

Same shape as :class:`AnalysisService`: every read is a polars ``scan_*`` so
projection pushdown reaches the file. ``indicators`` computes the standard TA
suite (EMA/SMA/RSI/MACD/Bollinger/ATR/VWAP) entirely in polars window
expressions — no Python loops over rows. ``signals`` derives crossover signals
from those indicators with vectorized ``diff``/``sign`` expressions. ``backtest``
is the one place a Python loop is justified: position state is inherently
sequential, so we replay the (already vectorized) indicators bar-by-bar.
``correlation``/``portfolio`` reduce multiple assets to a return matrix and the
usual risk metrics.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from .fs import FsService

# Annualization: treat each row as one trading day (252/year), matching
# AnalysisService.finance so the two surfaces report comparable numbers.
_TRADING_DAYS = 252.0


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


class TradingService:
    def __init__(self, settings, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs
        self._root = Path(settings.node_home)

    def _lf(self, relative: str) -> pl.LazyFrame:
        return _scan(self.fs._resolve(relative))

    def _columns(self, relative: str) -> list[str]:
        return self._lf(relative).collect_schema().names()

    async def indicators(self, path: str, column: str, ts_column: str | None = None) -> dict:
        have = self._columns(path)
        if column not in have:
            raise ValueError(
                f"Column {column!r} not in {path!r}. Available: {', '.join(have)}."
            )
        # Pull only what we need: the price column, optional ts, and any of the
        # optional OHLCV columns that actually exist (ATR needs high/low, VWAP
        # needs volume). Conventional names, case-insensitive lookup.
        lower = {c.lower(): c for c in have}
        high_col = lower.get("high")
        low_col = lower.get("low")
        vol_col = lower.get("volume") or lower.get("vol")

        want = [column]
        if ts_column:
            want.append(ts_column)
        for c in (high_col, low_col, vol_col):
            if c and c not in want:
                want.append(c)

        lf = self._lf(path).select(list(dict.fromkeys(want)))
        if ts_column:
            lf = lf.sort(ts_column)

        c = pl.col(column)
        delta = c.diff()
        gain = delta.clip(lower_bound=0).rolling_mean(window_size=14)
        loss = (-delta).clip(lower_bound=0).rolling_mean(window_size=14)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        macd_line = c.ewm_mean(span=12, ignore_nulls=True) - c.ewm_mean(span=26, ignore_nulls=True)
        macd_signal = macd_line.ewm_mean(span=9, ignore_nulls=True)

        bb_mid = c.rolling_mean(window_size=20)
        bb_std = c.rolling_std(window_size=20)

        exprs = [
            c.ewm_mean(span=9, ignore_nulls=True).alias("ema_9"),
            c.ewm_mean(span=21, ignore_nulls=True).alias("ema_21"),
            c.ewm_mean(span=50, ignore_nulls=True).alias("ema_50"),
            c.ewm_mean(span=200, ignore_nulls=True).alias("ema_200"),
            c.rolling_mean(window_size=20).alias("sma_20"),
            c.rolling_mean(window_size=50).alias("sma_50"),
            rsi.alias("rsi_14"),
            macd_line.alias("macd_line"),
            macd_signal.alias("macd_signal"),
            (macd_line - macd_signal).alias("macd_hist"),
            (bb_mid + 2 * bb_std).alias("bb_upper"),
            bb_mid.alias("bb_middle"),
            (bb_mid - 2 * bb_std).alias("bb_lower"),
        ]

        if high_col and low_col:
            # True range = max(high-low, |high-prev_close|, |low-prev_close|).
            prev_close = c.shift(1)
            tr = pl.max_horizontal(
                pl.col(high_col) - pl.col(low_col),
                (pl.col(high_col) - prev_close).abs(),
                (pl.col(low_col) - prev_close).abs(),
            )
            exprs.append(tr.rolling_mean(window_size=14).alias("atr_14"))

        if vol_col:
            # VWAP = cumulative(price*vol) / cumulative(vol).
            exprs.append(
                ((c * pl.col(vol_col)).cum_sum() / pl.col(vol_col).cum_sum()).alias("vwap")
            )

        df = lf.with_columns(exprs).collect()
        n = df.height

        out = {
            "price": df.get_column(column).to_list(),
            "ema_9": df.get_column("ema_9").to_list(),
            "ema_21": df.get_column("ema_21").to_list(),
            "ema_50": df.get_column("ema_50").to_list(),
            "ema_200": df.get_column("ema_200").to_list(),
            "sma_20": df.get_column("sma_20").to_list(),
            "sma_50": df.get_column("sma_50").to_list(),
            "rsi_14": df.get_column("rsi_14").to_list(),
            "macd_line": df.get_column("macd_line").to_list(),
            "macd_signal": df.get_column("macd_signal").to_list(),
            "macd_hist": df.get_column("macd_hist").to_list(),
            "bb_upper": df.get_column("bb_upper").to_list(),
            "bb_middle": df.get_column("bb_middle").to_list(),
            "bb_lower": df.get_column("bb_lower").to_list(),
            "atr_14": df.get_column("atr_14").to_list() if "atr_14" in df.columns else [None] * n,
            "vwap": df.get_column("vwap").to_list() if "vwap" in df.columns else [None] * n,
            "ts": df.get_column(ts_column).to_list() if ts_column else list(range(n)),
        }
        return out

    async def signals(self, path: str, column: str, ts_column: str | None = None) -> dict:
        ind = await self.indicators(path, column, ts_column)
        # Re-derive the crossover signals as a DataFrame so we stay vectorized:
        # a crossover is a sign change in (a - b), i.e. sign flips between rows.
        df = pl.DataFrame({
            "ema_9": ind["ema_9"],
            "ema_21": ind["ema_21"],
            "rsi_14": ind["rsi_14"],
            "macd_line": ind["macd_line"],
            "macd_signal": ind["macd_signal"],
        })

        ema_diff = pl.col("ema_9") - pl.col("ema_21")
        ema_prev = ema_diff.shift(1)
        ema_cross = (
            pl.when((ema_prev <= 0) & (ema_diff > 0)).then(1)
            .when((ema_prev >= 0) & (ema_diff < 0)).then(-1)
            .otherwise(0)
        )

        rsi = pl.col("rsi_14")
        rsi_prev = rsi.shift(1)
        rsi_signal = (
            pl.when((rsi_prev <= 30) & (rsi > 30)).then(1)
            .when((rsi_prev >= 70) & (rsi < 70)).then(-1)
            .otherwise(0)
        )

        macd_diff = pl.col("macd_line") - pl.col("macd_signal")
        macd_prev = macd_diff.shift(1)
        macd_cross = (
            pl.when((macd_prev <= 0) & (macd_diff > 0)).then(1)
            .when((macd_prev >= 0) & (macd_diff < 0)).then(-1)
            .otherwise(0)
        )

        df = df.with_columns(
            ema_cross.alias("ema_cross"),
            rsi_signal.alias("rsi_signal"),
            macd_cross.alias("macd_cross"),
        )
        # Composite: persistent trend state (ema above/below) plus RSI tilt,
        # clamped to [-1, 1]. Mean of the directional components.
        composite = (
            (pl.col("ema_9") > pl.col("ema_21")).cast(pl.Int8) * 2 - 1
        ).cast(pl.Float64)
        rsi_tilt = (
            pl.when(rsi < 30).then(1.0)
            .when(rsi > 70).then(-1.0)
            .otherwise(0.0)
        )
        macd_tilt = (
            pl.when(pl.col("macd_line") > pl.col("macd_signal")).then(1.0)
            .otherwise(-1.0)
        )
        df = df.with_columns(
            ((composite + rsi_tilt + macd_tilt) / 3.0)
            .clip(lower_bound=-1.0, upper_bound=1.0)
            .fill_null(0.0)
            .alias("signal")
        )

        return {
            "signal": df.get_column("signal").to_list(),
            "ema_cross": df.get_column("ema_cross").to_list(),
            "rsi_signal": df.get_column("rsi_signal").to_list(),
            "macd_cross": df.get_column("macd_cross").to_list(),
            "ts": ind["ts"],
        }

    async def backtest(self, path: str, column: str, strategy: str = "ema_cross",
                       initial_cash: float = 10_000.0, ts_column: str | None = None) -> dict:
        valid = {"ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"}
        if strategy not in valid:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Pick one of: {', '.join(sorted(valid))}."
            )

        ind = await self.indicators(path, column, ts_column)
        prices = ind["price"]
        n = len(prices)
        if n < 2:
            raise ValueError(f"Need >=2 rows to backtest, got {n}.")

        ema9, ema21 = ind["ema_9"], ind["ema_21"]
        rsi = ind["rsi_14"]
        macd_line, macd_signal = ind["macd_line"], ind["macd_signal"]
        ts = ind["ts"]

        # Position state is sequential — this loop is the documented exception
        # to the no-Python-loop rule. One share at a time, no costs, no partials.
        cash = float(initial_cash)
        shares = 0.0
        equity_curve: list[float] = []
        trades: list[dict] = []
        entry_price: float | None = None

        def _want_long(i: int) -> bool | None:
            """True=enter long, False=exit, None=hold."""
            if strategy == "buy_and_hold":
                return True if i == 0 else None
            if strategy == "ema_cross":
                a, b = ema9[i], ema21[i]
                if a is None or b is None:
                    return None
                return a > b
            if strategy == "rsi_mean_reversion":
                r = rsi[i]
                if r is None:
                    return None
                if r < 30:
                    return True
                if r > 70:
                    return False
                return None
            # macd
            m, s = macd_line[i], macd_signal[i]
            if m is None or s is None:
                return None
            return m > s

        for i in range(n):
            price = float(prices[i])
            want = _want_long(i)
            if want is True and shares == 0.0:
                shares = cash / price
                cash = 0.0
                entry_price = price
                trades.append({"ts": ts[i], "action": "buy", "price": price,
                               "shares": shares, "cash": cash, "value": shares * price})
            elif want is False and shares > 0.0:
                cash = shares * price
                trades.append({"ts": ts[i], "action": "sell", "price": price,
                               "shares": shares, "cash": cash, "value": cash})
                # Win = sold above entry.
                if entry_price is not None:
                    trades[-1]["win"] = price > entry_price
                shares = 0.0
                entry_price = None
            equity_curve.append(cash + shares * price)

        final_value = equity_curve[-1]
        benchmark_equity = [initial_cash * (float(p) / float(prices[0])) for p in prices]
        benchmark_return = benchmark_equity[-1] / initial_cash - 1.0

        metrics = _equity_metrics(equity_curve, initial_cash)
        sells = [t for t in trades if t["action"] == "sell"]
        wins = sum(1 for t in sells if t.get("win"))
        win_rate = wins / len(sells) if sells else 0.0
        n_trades = sum(1 for t in trades if t["action"] == "buy")

        return {
            "strategy": strategy,
            "initial_cash": float(initial_cash),
            "final_value": round(final_value, 4),
            "total_return": round(final_value / initial_cash - 1.0, 6),
            "ann_return": metrics["ann_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "equity_curve": equity_curve,
            "trades": trades,
            "benchmark_return": round(benchmark_return, 6),
            "benchmark_equity": benchmark_equity,
        }

    async def correlation(self, paths: list[str], column: str = "close") -> dict:
        if len(paths) < 2:
            raise ValueError(f"Need >=2 assets to correlate, got {len(paths)}.")
        # Align by row index (assets may differ in length): truncate to the
        # shortest, build a returns DataFrame, let polars compute corr/std.
        series: dict[str, pl.Series] = {}
        for p in paths:
            df = self._lf(p).select(pl.col(column)).collect()
            series[Path(p).stem] = df.get_column(column)
        m = min(s.len() for s in series.values())
        rets = pl.DataFrame({name: s.head(m) for name, s in series.items()}).select(
            pl.all().pct_change()
        ).drop_nulls()

        assets = list(series.keys())
        matrix = [
            [round(float(rets.select(pl.corr(a, b)).item() or 0.0), 6) for b in assets]
            for a in assets
        ]
        returns_matrix = [
            round(float(rets.get_column(a).mean() or 0.0) * _TRADING_DAYS, 6) for a in assets
        ]
        return {"assets": assets, "matrix": matrix, "returns_matrix": returns_matrix}

    async def portfolio(self, paths: list[str], weights: list[float] | None = None,
                        column: str = "close") -> dict:
        if not paths:
            raise ValueError("portfolio() needs at least one asset path.")
        if weights is None:
            weights = [1.0 / len(paths)] * len(paths)
        if len(weights) != len(paths):
            raise ValueError(
                f"Got {len(weights)} weights for {len(paths)} assets — they must match."
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Weights must sum to a positive number, got {total}.")
        weights = [w / total for w in weights]

        series: dict[str, pl.Series] = {}
        for p in paths:
            df = self._lf(p).select(pl.col(column)).collect()
            series[Path(p).stem] = df.get_column(column)
        m = min(s.len() for s in series.values())
        assets = list(series.keys())
        rets = pl.DataFrame({name: s.head(m) for name, s in series.items()}).select(
            pl.all().pct_change()
        ).drop_nulls()

        # Portfolio return series = weighted sum of component return series.
        port = rets.select(
            sum(pl.col(a) * w for a, w in zip(assets, weights)).alias("port")
        ).get_column("port")

        port_metrics = _return_metrics(port)
        component_sharpes = []
        weighted_vol = 0.0
        for a, w in zip(assets, weights):
            cm = _return_metrics(rets.get_column(a))
            component_sharpes.append(cm["sharpe"])
            weighted_vol += w * cm["ann_volatility"]
        # Diversification ratio = weighted avg of component vols / portfolio vol.
        port_vol = port_metrics["ann_volatility"]
        div_ratio = (weighted_vol / port_vol) if port_vol > 0 else 1.0

        return {
            "assets": assets,
            "weights": [round(w, 6) for w in weights],
            "total_return": port_metrics["total_return"],
            "ann_return": port_metrics["ann_return"],
            "ann_volatility": port_metrics["ann_volatility"],
            "sharpe": port_metrics["sharpe"],
            "sortino": port_metrics["sortino"],
            "max_drawdown": port_metrics["max_drawdown"],
            "weighted_return": round(
                sum(w * float(rets.get_column(a).mean() or 0.0) * _TRADING_DAYS
                    for a, w in zip(assets, weights)), 6),
            "diversification_ratio": round(div_ratio, 4),
            "component_sharpes": component_sharpes,
        }


def _equity_metrics(equity: list[float], initial: float) -> dict:
    """Risk/return metrics from a portfolio value curve."""
    eq = pl.Series("eq", equity, dtype=pl.Float64)
    ret = eq.pct_change().drop_nulls()
    running_max = eq.cum_max()
    drawdown = (eq - running_max) / running_max
    mean_r = float(ret.mean() or 0.0)
    std_r = float(ret.std() or 0.0)
    neg = ret.filter(ret < 0)
    std_neg = float(neg.std() or 0.0)
    return {
        "ann_return": round(mean_r * _TRADING_DAYS, 6),
        "max_drawdown": round(float(drawdown.min() or 0.0), 6),
        "sharpe": round((mean_r / std_r * math.sqrt(_TRADING_DAYS)) if std_r > 0 else 0.0, 4),
        "sortino": round((mean_r / std_neg * math.sqrt(_TRADING_DAYS)) if std_neg > 0 else 0.0, 4),
    }


def _return_metrics(ret_or_price: pl.Series) -> dict:
    """Standard metrics from a *returns* series (already pct_change'd)."""
    ret = ret_or_price
    cum = (1.0 + ret).cum_prod()
    running_max = cum.cum_max()
    drawdown = (cum - running_max) / running_max
    mean_r = float(ret.mean() or 0.0)
    std_r = float(ret.std() or 0.0)
    neg = ret.filter(ret < 0)
    std_neg = float(neg.std() or 0.0)
    total_return = float(cum[-1]) - 1.0 if cum.len() else 0.0
    return {
        "total_return": round(total_return, 6),
        "ann_return": round(mean_r * _TRADING_DAYS, 6),
        "ann_volatility": round(std_r * math.sqrt(_TRADING_DAYS), 6),
        "sharpe": round((mean_r / std_r * math.sqrt(_TRADING_DAYS)) if std_r > 0 else 0.0, 4),
        "sortino": round((mean_r / std_neg * math.sqrt(_TRADING_DAYS)) if std_neg > 0 else 0.0, 4),
        "max_drawdown": round(float(drawdown.min() or 0.0), 6),
    }


STRATEGIES = [
    {"id": "ema_cross", "name": "EMA Crossover",
     "description": "Long when EMA(9) is above EMA(21), flat otherwise."},
    {"id": "rsi_mean_reversion", "name": "RSI Mean Reversion",
     "description": "Buy when RSI(14) < 30 (oversold), sell when RSI(14) > 70 (overbought)."},
    {"id": "macd", "name": "MACD Crossover",
     "description": "Long on a bullish MACD cross (MACD above signal), exit on a bearish cross."},
    {"id": "buy_and_hold", "name": "Buy & Hold",
     "description": "Baseline: buy at the first bar and hold to the end."},
]
