"""Tests for the yggdrasil.bot FastAPI application."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yggdrasil.bot import BotSettings, create_app


@pytest.fixture()
def client():
    app = create_app(BotSettings())
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

def test_ping(client):
    r = client.get("/api/ping")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "version" in data
    assert data["service"] == "ygg-bot"


def test_health(client):
    r = client.get("/api/v2/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "uptime_s" in data
    assert isinstance(data["market_cache_size"], int)
    assert isinstance(data["ws_connections"], int)


def test_stats(client):
    r = client.get("/api/v2/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["requests_total"] >= 0
    assert data["cache_hits"] >= 0
    assert data["cache_misses"] >= 0


# ---------------------------------------------------------------------------
# Market endpoints
# ---------------------------------------------------------------------------

def test_prices_valid_zone(client):
    r = client.get("/api/v2/market/prices?zone=DE_LU&days=1")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["zone"] == "DE_LU"
    assert data["days"] == 1
    assert isinstance(data["prices"], list)
    assert data["count"] == len(data["prices"])


def test_prices_unknown_zone(client):
    r = client.get("/api/v2/market/prices?zone=MARS")
    assert r.status_code == 400


def test_prices_unknown_series(client):
    r = client.get("/api/v2/market/prices?zone=DE_LU&series=nonsense")
    assert r.status_code == 400


def test_prices_days_bounds(client):
    r = client.get("/api/v2/market/prices?zone=DE_LU&days=0")
    assert r.status_code == 422      # FastAPI query validation

    r = client.get("/api/v2/market/prices?zone=DE_LU&days=91")
    assert r.status_code == 422


def test_fx_default(client):
    r = client.get("/api/v2/market/fx")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["base"] == "EUR"
    assert isinstance(data["rates"], list)


def test_fx_empty_targets(client):
    r = client.get("/api/v2/market/fx?targets=")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Signals endpoint
# ---------------------------------------------------------------------------

def test_signals_valid(client):
    r = client.get("/api/v2/signals?zone=DE_LU")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert isinstance(data["signals"], list)
    # With no ENTSOE token, prices are empty → no signals
    for sig in data["signals"]:
        assert sig["kind"] in ("BUY", "SELL", "HOLD")


def test_signals_unknown_zone(client):
    r = client.get("/api/v2/signals?zone=MARS")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Cache and performance
# ---------------------------------------------------------------------------

def test_cache_hit_on_repeated_prices(client):
    """Second prices call should be a cache hit (miss count stays at 1)."""
    from yggdrasil.bot.market import cache_stats

    _, _, misses_before = cache_stats()
    client.get("/api/v2/market/prices?zone=FR&days=2")
    _, _, misses_after_1 = cache_stats()
    client.get("/api/v2/market/prices?zone=FR&days=2")
    _, _, misses_after_2 = cache_stats()

    # First call is a miss, second call should be a hit (miss count unchanged)
    assert misses_after_1 == misses_before + 1
    assert misses_after_2 == misses_after_1


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------

def test_signal_computation():
    import datetime as dt
    from yggdrasil.bot.signals import compute_signals

    base = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
    # Flat prices → HOLD
    flat = [{"timestamp": base + dt.timedelta(hours=i), "value": 50.0,
             "unit": "MWh", "currency": "EUR"} for i in range(24)]
    sigs = compute_signals(flat, "DE_LU", "day_ahead_prices")
    assert len(sigs) == 1
    assert sigs[0].kind == "HOLD"
    assert sigs[0].zscore == 0.0

    # Spike → SELL
    spike = [{"timestamp": base + dt.timedelta(hours=i), "value": float(50 + i * 0.1),
              "unit": "MWh", "currency": "EUR"} for i in range(24)]
    spike[-1]["value"] = 500.0   # massive spike
    sigs = compute_signals(spike, "DE_LU", "day_ahead_prices")
    assert sigs[0].kind == "SELL"

    # Crash → BUY
    crash = [{"timestamp": base + dt.timedelta(hours=i), "value": float(50 + i * 0.1),
              "unit": "MWh", "currency": "EUR"} for i in range(24)]
    crash[-1]["value"] = -500.0
    sigs = compute_signals(crash, "DE_LU", "day_ahead_prices")
    assert sigs[0].kind == "BUY"


def test_signal_empty_prices():
    from yggdrasil.bot.signals import compute_signals
    assert compute_signals([], "DE_LU", "day_ahead_prices") == []


# ---------------------------------------------------------------------------
# Market service
# ---------------------------------------------------------------------------

def test_ttl_cache():
    import time
    from yggdrasil.bot.market import _TTLCache

    cache = _TTLCache()
    assert cache.get("x") is None
    cache.set("x", [1, 2, 3], ttl=100)
    assert cache.get("x") == [1, 2, 3]
    assert cache.size() == 1
    # Expired
    cache.set("y", "val", ttl=0.001)
    time.sleep(0.01)
    assert cache.get("y") is None


def test_root_redirects(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
