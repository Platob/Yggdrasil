"""Tests for TradingService: indicators, signals, backtest, scan, correlation, portfolio."""
from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.node.config import Settings
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.trading import TradingService, _ds_list, _downsample_dict


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def ohlcv_path(tmp_dir: Path) -> str:
    """Write a small deterministic OHLCV parquet. Prices start at 100 and alternate up/down."""
    n = 200
    prices = [100.0 + (i % 20 - 10) * 0.5 for i in range(n)]  # gentle oscillation
    highs = [p + 0.5 for p in prices]
    lows = [p - 0.5 for p in prices]
    vols = [1000 + i * 10 for i in range(n)]
    pq.write_table(pa.table({
        "ts": list(range(n)),
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": vols,
    }), str(tmp_dir / "ohlcv.parquet"))
    return "ohlcv.parquet"


@pytest.fixture(scope="module")
def svc(tmp_dir: Path) -> TradingService:
    settings = Settings(node_id="test", node_home=tmp_dir, front_home=tmp_dir)
    fs = FsService(settings)
    return TradingService(settings, fs)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# downsample helpers
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_ds_list_passthrough(self):
        lst = list(range(10))
        assert _ds_list(lst, None) is lst
        assert _ds_list(lst, 100) is lst

    def test_ds_list_downsamples(self):
        lst = list(range(100))
        result = _ds_list(lst, 10)
        assert len(result) == 10
        assert result[0] == lst[0]

    def test_downsample_dict(self):
        d = {"a": list(range(100)), "b": list(range(100, 200)), "c": "scalar"}
        out = _downsample_dict(d, 20)
        assert len(out["a"]) == 20
        assert len(out["b"]) == 20
        assert out["c"] == "scalar"

    def test_downsample_dict_passthrough(self):
        d = {"a": list(range(10))}
        assert _downsample_dict(d, 50) is d


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_returns_expected_keys(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close"))
        for key in ("price", "ts", "ema_9", "ema_21", "rsi_14",
                    "macd_line", "macd_signal", "macd_hist",
                    "bb_upper", "bb_middle", "bb_lower", "atr_14", "vwap"):
            assert key in result, f"Missing key: {key}"

    def test_price_length(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close"))
        assert len(result["price"]) == 200
        assert len(result["ts"]) == 200

    def test_max_points_downsamples(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close", max_points=50))
        assert len(result["price"]) == 50
        assert len(result["ema_9"]) == 50

    def test_missing_column_raises(self, svc: TradingService, ohlcv_path: str):
        with pytest.raises(ValueError, match="Column"):
            run(svc.indicators(ohlcv_path, "nonexistent"))

    def test_cache_returns_same_object(self, svc: TradingService, ohlcv_path: str):
        svc._cache.clear()
        a = run(svc._indicators_full(ohlcv_path, "close"))
        b = run(svc._indicators_full(ohlcv_path, "close"))
        assert a is b  # same object from cache

    def test_ema_non_null(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close"))
        # EMA is non-null from the very first row (ewm with ignore_nulls)
        assert all(v is not None for v in result["ema_9"])

    def test_vwap_present_when_volume_exists(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close"))
        vwap = result["vwap"]
        assert any(v is not None for v in vwap)

    def test_atr_present_when_high_low_exist(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.indicators(ohlcv_path, "close"))
        atr = result["atr_14"]
        # First 13 rows may be null (rolling window), rest should be non-null
        non_null = [v for v in atr if v is not None]
        assert len(non_null) > 150


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

class TestSignals:
    def test_returns_expected_keys(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.signals(ohlcv_path, "close"))
        for key in ("signal", "ema_cross", "rsi_signal", "macd_cross", "ts"):
            assert key in result

    def test_signal_range(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.signals(ohlcv_path, "close"))
        for v in result["signal"]:
            assert -1.0 <= v <= 1.0, f"Signal out of range: {v}"

    def test_ema_cross_values(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.signals(ohlcv_path, "close"))
        for v in result["ema_cross"]:
            assert v in (-1, 0, 1)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_buy_and_hold_return(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close", strategy="buy_and_hold"))
        assert result["n_trades"] == 1
        # buy_and_hold: final_value = initial_cash * last_price / first_price
        assert math.isfinite(result["total_return"])

    def test_required_fields(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close"))
        for key in ("strategy", "initial_cash", "final_value", "total_return",
                    "ann_return", "max_drawdown", "sharpe", "sortino", "n_trades",
                    "win_rate", "profit_factor", "avg_win_pct", "avg_loss_pct",
                    "max_consecutive_losses", "equity_curve", "benchmark_equity",
                    "benchmark_return", "trades"):
            assert key in result

    def test_equity_curve_length(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close", max_points=50))
        assert len(result["equity_curve"]) == 50
        assert len(result["benchmark_equity"]) == 50

    def test_final_value_positive(self, svc: TradingService, ohlcv_path: str):
        for strategy in ("ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"):
            result = run(svc.backtest(ohlcv_path, "close", strategy=strategy))
            assert result["final_value"] >= 0, f"Negative final value for {strategy}"

    def test_initial_cash_respected(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close", initial_cash=5000.0))
        assert result["initial_cash"] == 5000.0
        assert result["equity_curve"][0] == pytest.approx(5000.0, rel=0.01)

    def test_stop_loss_exits_recorded(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close", stop_loss_pct=0.001))
        stop_trades = [t for t in result["trades"] if t["action"] == "stop_loss"]
        # Very tight stop loss (0.1%) should trigger on an oscillating price series
        assert len(stop_trades) > 0

    def test_take_profit_exits_recorded(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close", take_profit_pct=0.001))
        tp_trades = [t for t in result["trades"] if t["action"] == "take_profit"]
        assert len(tp_trades) > 0

    def test_half_sizing_keeps_cash(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close",
                                  strategy="buy_and_hold", position_sizing="half"))
        # trade["value"] = cash + shares * price (total equity at that bar)
        buy_trade = next(t for t in result["trades"] if t["action"] == "buy")
        total_equity = buy_trade["value"]  # cash + shares * price
        remaining_cash = buy_trade["cash"]
        # With half sizing, roughly 50% remains as cash
        assert remaining_cash > total_equity * 0.4

    def test_invalid_strategy_raises(self, svc: TradingService, ohlcv_path: str):
        with pytest.raises(ValueError, match="Unknown strategy"):
            run(svc.backtest(ohlcv_path, "close", strategy="bad_strategy"))

    def test_invalid_sizing_raises(self, svc: TradingService, ohlcv_path: str):
        with pytest.raises(ValueError, match="Unknown position_sizing"):
            run(svc.backtest(ohlcv_path, "close", position_sizing="tenth"))

    def test_profit_factor_none_when_no_losses(self, svc: TradingService, ohlcv_path: str):
        # Use buy_and_hold on monotonically increasing prices → no losing trades
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rising.parquet"
            pq.write_table(pa.table({"close": [float(i + 1) for i in range(50)]}), str(p))
            settings = Settings(node_id="t", node_home=Path(d), front_home=Path(d))
            fs = FsService(settings)
            svc2 = TradingService(settings, fs)
            result = run(svc2.backtest("rising.parquet", "close", strategy="buy_and_hold"))
            # buy_and_hold has only 1 trade (buy) — no sells → profit_factor is 0
            assert result["profit_factor"] == 0.0 or result["profit_factor"] is None

    def test_trades_capped_at_500(self, svc: TradingService, ohlcv_path: str):
        # Generate many trades by using a very tight stop
        result = run(svc.backtest(ohlcv_path, "close", stop_loss_pct=0.001))
        assert len(result["trades"]) <= 500

    def test_metrics_finite(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.backtest(ohlcv_path, "close"))
        for key in ("ann_return", "max_drawdown", "sharpe", "sortino", "win_rate"):
            assert math.isfinite(result[key]), f"{key} = {result[key]}"


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

class TestScan:
    def test_returns_one_result_per_path(self, svc: TradingService, ohlcv_path: str):
        results = run(svc.scan([ohlcv_path, ohlcv_path]))
        assert len(results) == 2

    def test_result_fields(self, svc: TradingService, ohlcv_path: str):
        results = run(svc.scan([ohlcv_path]))
        r = results[0]
        assert "path" in r
        assert "price" in r
        assert "signal" in r
        assert -1.0 <= r["signal"] <= 1.0

    def test_error_row_on_bad_path(self, svc: TradingService, ohlcv_path: str):
        results = run(svc.scan(["nonexistent.parquet"]))
        assert len(results) == 1
        assert "error" in results[0]


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_self_correlation_is_one(self, svc: TradingService, ohlcv_path: str):
        # Same file twice: deduplicated to "ohlcv" and "ohlcv_2" — correlation = 1.0
        result = run(svc.correlation([ohlcv_path, ohlcv_path]))
        assert len(result["assets"]) == 2
        assert result["matrix"][0][1] == pytest.approx(1.0, abs=1e-4)

    def test_requires_two_assets(self, svc: TradingService, ohlcv_path: str):
        with pytest.raises(ValueError, match=">=2"):
            run(svc.correlation([ohlcv_path]))

    def test_matrix_shape(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.correlation([ohlcv_path, ohlcv_path]))
        n = len(result["assets"])
        assert len(result["matrix"]) == n
        assert all(len(row) == n for row in result["matrix"])


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_equal_weights_default(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.portfolio([ohlcv_path, ohlcv_path]))
        assert result["weights"] == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_custom_weights_normalized(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.portfolio([ohlcv_path, ohlcv_path], weights=[1.0, 3.0]))
        assert result["weights"][0] == pytest.approx(0.25, rel=1e-4)
        assert result["weights"][1] == pytest.approx(0.75, rel=1e-4)

    def test_metrics_present(self, svc: TradingService, ohlcv_path: str):
        result = run(svc.portfolio([ohlcv_path, ohlcv_path]))
        for key in ("total_return", "ann_return", "ann_volatility", "sharpe",
                    "sortino", "max_drawdown", "diversification_ratio"):
            assert key in result

    def test_weight_mismatch_raises(self, svc: TradingService, ohlcv_path: str):
        with pytest.raises(ValueError, match="weights"):
            run(svc.portfolio([ohlcv_path, ohlcv_path], weights=[1.0]))
