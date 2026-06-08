"""Tests for the ``yggdrasil.node`` HTTP surface.

The market layer is mocked at :func:`yggdrasil.node.api.market.fetch_chart`
(and the underlying httpx client never fires) so every test is offline.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node import Settings, create_app
from yggdrasil.node.transport import CONTENT_TYPE_PICKLE, deserialize_pickle, serialize_pickle


def _fake_chart(symbol: str = "AAPL", n: int = 180) -> dict:
    closes = [150.0 + math.sin(i * 0.2) * 6 + i * 0.05 for i in range(n)]
    return {
        "meta": {
            "symbol": symbol.upper(),
            "regularMarketPrice": closes[-1],
            "chartPreviousClose": closes[-2],
            "currency": "USD",
            "exchangeName": "NASDAQ",
            "regularMarketVolume": 50_000_000,
            "regularMarketTime": 1_700_000_000,
        },
        "timestamp": [1_700_000_000 + i * 86_400 for i in range(n)],
        "indicators": {
            "quote": [
                {
                    "open": [c - 0.5 for c in closes],
                    "high": [c + 1.0 for c in closes],
                    "low": [c - 1.0 for c in closes],
                    "close": closes,
                    "volume": [50_000_000] * n,
                }
            ]
        },
    }


@pytest.fixture
def client(monkeypatch):
    async def _fake_fetch(symbol, interval, range_):
        return _fake_chart(symbol)

    # Patch where the symbol is *used* so trading/ai see the fake too.
    monkeypatch.setattr("yggdrasil.node.api.market.fetch_chart", _fake_fetch)
    monkeypatch.setattr("yggdrasil.node.api.trading.fetch_chart", _fake_fetch)
    # Reset the in-process portfolio between tests.
    from yggdrasil.node.api import trading

    trading._POSITIONS.clear()
    trading._HISTORY.clear()
    return TestClient(create_app(Settings()))


class TestHealth:
    def test_ping(self, client):
        body = client.get("/api/ping").json()
        assert body["pong"] is True and isinstance(body["ts"], int)

    def test_health(self, client):
        body = client.get("/api/v2/health").json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_backend_reports_arrow(self, client):
        body = client.get("/api/v2/backend").json()
        assert "arrow" in body and "python" in body


class TestCall:
    def test_call_scalar_function(self, client):
        from yggdrasil.node.remote import remote

        @remote(name="apitest:add")
        def _add(x, y):
            return x + y

        body = serialize_pickle({"func": "apitest:add", "args": [2, 3], "kwargs": {}})
        resp = client.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        assert resp.status_code == 200
        assert deserialize_pickle(resp.content) == 5

    def test_call_unknown_function_404(self, client):
        body = serialize_pickle({"func": "apitest:nope", "args": [], "kwargs": {}})
        resp = client.post("/api/call", content=body, headers={"Content-Type": CONTENT_TYPE_PICKLE})
        assert resp.status_code == 404

    def test_call_missing_func_400(self, client):
        resp = client.post("/api/call", json={"args": []})
        assert resp.status_code == 400


class TestMarket:
    def test_quote(self, client):
        body = client.get("/api/v2/market/quote/AAPL").json()
        assert body["symbol"] == "AAPL"
        assert body["price"] is not None
        assert body["change_pct"] is not None

    def test_ohlcv(self, client):
        body = client.get("/api/v2/market/ohlcv/AAPL?range=6mo").json()
        assert body["symbol"] == "AAPL"
        assert len(body["data"]) > 0
        assert set(body["data"][0]) >= {"ts", "open", "high", "low", "close", "volume"}

    def test_batch(self, client):
        body = client.get("/api/v2/market/batch?symbols=AAPL,MSFT").json()
        assert set(body["quotes"]) == {"AAPL", "MSFT"}


class TestTrading:
    def test_signals(self, client):
        body = client.get("/api/v2/trading/signals/AAPL").json()
        assert body["symbol"] == "AAPL"
        assert body["signal"] in {"BUY", "SELL", "HOLD"}
        assert "indicators" in body

    def test_scan_ranks_buys_first(self, client):
        body = client.get("/api/v2/trading/scan?symbols=AAPL,MSFT,GOOGL").json()
        signals = [r["signal"] for r in body["scan"]]
        order = {"BUY": 0, "HOLD": 1, "SELL": 2}
        assert signals == sorted(signals, key=lambda s: order[s])

    def test_trade_buy_then_portfolio(self, client):
        r = client.post(
            "/api/v2/trading/portfolio/trade",
            json={"symbol": "AAPL", "action": "BUY", "shares": 10, "price": 150.0},
        )
        assert r.status_code == 200
        assert r.json()["position"]["shares"] == 10

        pf = client.get("/api/v2/trading/portfolio").json()
        assert pf["positions"][0]["symbol"] == "AAPL"
        assert pf["positions"][0]["shares"] == 10

    def test_oversell_rejected(self, client):
        client.post(
            "/api/v2/trading/portfolio/trade",
            json={"symbol": "AAPL", "action": "BUY", "shares": 5, "price": 100.0},
        )
        r = client.post(
            "/api/v2/trading/portfolio/trade",
            json={"symbol": "AAPL", "action": "SELL", "shares": 99, "price": 100.0},
        )
        assert r.status_code == 422

    def test_invalid_action_rejected(self, client):
        r = client.post(
            "/api/v2/trading/portfolio/trade",
            json={"symbol": "AAPL", "action": "HODL", "shares": 1, "price": 1.0},
        )
        assert r.status_code == 400

    def test_history_records_trades(self, client):
        client.post(
            "/api/v2/trading/portfolio/trade",
            json={"symbol": "MSFT", "action": "BUY", "shares": 2, "price": 300.0},
        )
        history = client.get("/api/v2/trading/portfolio/history").json()
        assert history[0]["symbol"] == "MSFT"
        assert history[0]["action"] == "BUY"


class TestAI:
    def test_analyze_falls_back_to_rules(self, client):
        body = client.post("/api/v2/ai/analyze", json={"symbol": "AAPL"}).json()
        assert body["symbol"] == "AAPL"
        assert body["recommendation"] in {"BUY", "SELL", "HOLD"}
        assert isinstance(body["analysis"], str) and body["analysis"]
        assert body["source"] == "rules"


class TestApiKey:
    def test_request_blocked_without_key(self, monkeypatch):
        from yggdrasil.node.api import trading

        trading._POSITIONS.clear()
        app = create_app(Settings(api_key="s3cret"))
        c = TestClient(app)
        # ping stays open for liveness probes
        assert c.get("/api/ping").status_code == 200
        # guarded endpoint requires the header
        assert c.get("/api/v2/health").status_code == 401
        assert c.get("/api/v2/health", headers={"X-API-Key": "s3cret"}).status_code == 200
