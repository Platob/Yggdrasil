"""Smoke tests for the trading service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings
from yggdrasil.node.exceptions import BotError, NotFoundError
from yggdrasil.node.schemas.trading import (
    DEFAULT_SYMBOLS,
    OrderCreate,
    PriceAlertCreate,
)
from yggdrasil.node.services.trading import TradingService


@pytest.fixture
def svc() -> TradingService:
    return TradingService(Settings(allow_remote=True))


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(allow_remote=True)))


# -- service ---------------------------------------------------------------

def test_default_watchlist_seeded(svc):
    syms = {e.symbol for e in svc.get_watchlist()}
    assert syms == set(DEFAULT_SYMBOLS.keys())


def test_get_price_known_symbol(svc):
    q = svc.get_price("AAPL")
    assert q.symbol == "AAPL"
    assert q.price > 0
    assert q.timestamp_ms > 0


def test_get_price_unknown_symbol_works(svc):
    """Unknown symbols still return a deterministic price."""
    q = svc.get_price("WEIRD")
    assert q.symbol == "WEIRD"
    assert q.price > 0


def test_market_buy_fills_immediately(svc):
    o = svc.place_order(OrderCreate(symbol="AAPL", side="buy", qty=5))
    assert o.status == "filled"
    assert o.filled_qty == 5
    assert o.avg_fill_price is not None and o.avg_fill_price > 0
    p = svc.get_portfolio()
    assert len(p.positions) == 1
    assert p.positions[0].symbol == "AAPL"
    assert p.positions[0].qty == 5


def test_buy_then_sell_realises_pnl(svc):
    svc.place_order(OrderCreate(symbol="NVDA", side="buy", qty=2))
    svc.place_order(OrderCreate(symbol="NVDA", side="sell", qty=1))
    p = svc.get_portfolio()
    assert any(pos.symbol == "NVDA" and pos.qty == 1 for pos in p.positions)
    history = svc.get_trade_history()
    assert len(history) == 2


def test_insufficient_cash_rejected(svc):
    with pytest.raises(BotError):
        # BTC at ~$67500 — buying 1000 of those exceeds the $100k starting cash.
        svc.place_order(OrderCreate(symbol="BTC-USD", side="buy", qty=1000))


def test_invalid_qty_rejected(svc):
    with pytest.raises(BotError):
        svc.place_order(OrderCreate(symbol="AAPL", side="buy", qty=0))


def test_limit_order_pending_then_cancel(svc):
    o = svc.place_order(OrderCreate(
        symbol="TSLA", side="buy", qty=1, order_type="limit", limit_price=1.0,
    ))
    assert o.status == "pending"
    cancelled = svc.cancel_order(o.id)
    assert cancelled.status == "cancelled"


def test_cancel_unknown_order_404(svc):
    with pytest.raises(NotFoundError):
        svc.cancel_order(123456789)


def test_signal_has_indicators(svc):
    s = svc.get_signal("NVDA")
    assert s.symbol == "NVDA"
    assert "ma20" in s.indicators
    assert "ma50" in s.indicators
    assert "rsi14" in s.indicators
    assert s.signal in {"strong_buy", "buy", "hold", "sell", "strong_sell"}


def test_watchlist_add_and_remove(svc):
    svc.add_watchlist("META")
    assert any(e.symbol == "META" for e in svc.get_watchlist())
    svc.remove_watchlist("META")
    assert not any(e.symbol == "META" for e in svc.get_watchlist())


def test_alert_triggers(svc):
    # Threshold of $1 will always trigger 'above' for any positive price.
    a = svc.create_alert(PriceAlertCreate(symbol="AAPL", condition="above", threshold=1.0))
    assert not a.triggered
    alerts = svc.list_alerts()
    assert any(a.triggered for a in alerts if a.symbol == "AAPL")


# -- HTTP ------------------------------------------------------------------

def test_http_prices(client):
    r = client.get("/api/trading/prices")
    assert r.status_code == 200
    prices = r.json()["prices"]
    assert len(prices) >= 9


def test_http_place_market_order(client):
    r = client.post("/api/trading/orders", json={"symbol": "AAPL", "side": "buy", "qty": 1})
    assert r.status_code == 200
    assert r.json()["order"]["status"] == "filled"


def test_http_signals(client):
    r = client.get("/api/trading/signals")
    assert r.status_code == 200
    assert len(r.json()["signals"]) >= 9
