"""Integration tests for the node FastAPI app: HTTP round-trips via ASGI."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.node.config import Settings
from yggdrasil.node.api.app import create_api


@pytest.fixture(scope="module")
def api_setup():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        pq.write_table(pa.table({
            "ts": list(range(50)),
            "close": [100.0 + (i % 10 - 5) * 0.5 for i in range(50)],
            "volume": [1000] * 50,
        }), str(home / "prices.parquet"))
        settings = Settings(node_id="test", node_home=home, front_home=home)
        yield create_api(settings)


@pytest.fixture(scope="module")
def client(api_setup):
    import httpx
    transport = httpx.ASGITransport(app=api_setup)

    async def _get_client():
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    return asyncio.get_event_loop().run_until_complete(_get_client())


def call(client, method, path, json=None):
    coro = client.get(path) if method == "get" else client.post(path, json=json)
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDiscovery:
    def test_ping(self, client):
        r = call(client, "get", "/api/ping")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self, client):
        r = call(client, "get", "/api/v2/health")
        assert r.json()["status"] == "healthy"

    def test_stats(self, client):
        r = call(client, "get", "/api/v2/stats")
        data = r.json()
        assert "uptime_s" in data
        assert "requests" in data

    def test_strategies(self, client):
        r = call(client, "get", "/api/v2/trading/strategies")
        strats = r.json()["strategies"]
        ids = {s["id"] for s in strats}
        assert {"ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"} == ids


class TestTradingEndpoints:
    def test_indicators_200(self, client):
        r = call(client, "post", "/api/v2/trading/indicators",
                 json={"path": "prices.parquet", "column": "close"})
        assert r.status_code == 200
        data = r.json()
        assert "price" in data
        assert "ema_9" in data
        assert "rsi_14" in data

    def test_indicators_missing_column_400(self, client):
        r = call(client, "post", "/api/v2/trading/indicators",
                 json={"path": "prices.parquet", "column": "bad_col"})
        assert r.status_code == 400

    def test_signals_200(self, client):
        r = call(client, "post", "/api/v2/trading/signals",
                 json={"path": "prices.parquet", "column": "close"})
        assert r.status_code == 200
        assert "signal" in r.json()

    def test_backtest_200(self, client):
        r = call(client, "post", "/api/v2/trading/backtest",
                 json={"path": "prices.parquet", "column": "close"})
        assert r.status_code == 200
        data = r.json()
        assert "total_return" in data
        assert "equity_curve" in data
        assert "profit_factor" in data

    def test_backtest_with_stop_loss(self, client):
        r = call(client, "post", "/api/v2/trading/backtest", json={
            "path": "prices.parquet", "column": "close",
            "stop_loss_pct": 0.02, "take_profit_pct": 0.05,
            "position_sizing": "half",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["initial_cash"] == 10000.0

    def test_backtest_invalid_strategy_422(self, client):
        # Invalid strategy literal is caught by Pydantic before reaching the service
        r = call(client, "post", "/api/v2/trading/backtest",
                 json={"path": "prices.parquet", "column": "close", "strategy": "bad"})
        assert r.status_code == 422

    def test_scan_200(self, client):
        r = call(client, "post", "/api/v2/trading/scan",
                 json={"paths": ["prices.parquet"], "column": "close"})
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert "signal" in results[0]

    def test_correlation_200(self, client):
        r = call(client, "post", "/api/v2/trading/correlation",
                 json={"paths": ["prices.parquet", "prices.parquet"], "column": "close"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["assets"]) == 2

    def test_portfolio_200(self, client):
        r = call(client, "post", "/api/v2/trading/portfolio",
                 json={"paths": ["prices.parquet", "prices.parquet"], "column": "close"})
        assert r.status_code == 200
        data = r.json()
        assert "sharpe" in data
        assert "diversification_ratio" in data
