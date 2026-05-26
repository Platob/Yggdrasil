"""Trading service benchmarks.

Measures:
1. Price fetch latency (against live node if available, else direct service)
2. Portfolio P&L computation speed
3. Technical indicator computation speed (SMA, EMA, RSI on 200 data points)
4. Alert checking speed

Usage:
    python benchmarks/bot/bench_trading.py

If a node is running at $YGG_BENCH_URL, benchmarks hit the live API.
Otherwise, they test the service directly (no network).
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
import urllib.request

BASE_URL = os.environ.get("YGG_BENCH_URL", "http://127.0.0.1:8100")


def _timed(label: str, fn, n: int = 1):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = statistics.mean(times)
    med = statistics.median(times)
    p99 = sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else times[0]
    print(f"  {label}: avg={avg*1000:.1f}ms  med={med*1000:.1f}ms  p99={p99*1000:.1f}ms  (n={n})")
    return result


def _node_available() -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/hello")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _post(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _delete(path: str) -> None:
    req = urllib.request.Request(f"{BASE_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=10):
        pass


# -- Live node benchmarks ------------------------------------------------------


def bench_price_fetch_live():
    print("\n=== Price Fetch (Live API) ===")
    _timed("GET /api/trading (default prices)", lambda: _get("/api/trading"), n=20)
    _timed("GET /api/trading/prices?symbols=EUR/USD,BTC-USD",
           lambda: _get("/api/trading/prices?symbols=EUR/USD,BTC-USD"), n=20)


def bench_portfolio_live():
    print("\n=== Portfolio (Live API) ===")
    # Seed positions
    for i in range(10):
        _post("/api/trading/portfolio/position", {
            "symbol": f"TEST-{i}",
            "quantity": random.uniform(1.0, 100.0),
            "avg_cost": random.uniform(10.0, 1000.0),
            "currency": "USD",
        })
    _timed("GET /api/trading/portfolio (10 positions)", lambda: _get("/api/trading/portfolio"), n=50)
    # Cleanup
    for i in range(10):
        try:
            _delete(f"/api/trading/portfolio/position/TEST-{i}")
        except Exception:
            pass


def bench_alerts_live():
    print("\n=== Alerts (Live API) ===")
    alert_ids = []
    for i in range(20):
        resp = _post("/api/trading/alerts", {
            "symbol": "BTC-USD",
            "condition": "above",
            "price": 999999.0 + i,
        })
        alert_ids.append(resp["id"])

    _timed("GET /api/trading/alerts (20 alerts)", lambda: _get("/api/trading/alerts"), n=50)

    # Cleanup
    for aid in alert_ids:
        try:
            _delete(f"/api/trading/alerts/{aid}")
        except Exception:
            pass


# -- Direct service benchmarks (no network) ------------------------------------


def _make_service():
    from yggdrasil.node.config import Settings
    from yggdrasil.node.services.trading import TradingService
    return TradingService(Settings())


def _seed_price_history(service, symbol: str, n: int = 200):
    """Inject synthetic price history for benchmark."""
    import datetime as dt
    from collections import deque

    prices = deque(maxlen=200)
    base_price = 100.0
    now = time.time()
    for i in range(n):
        ts = dt.datetime.fromtimestamp(now - (n - i) * 60, tz=dt.timezone.utc).isoformat()
        price = base_price + random.uniform(-5.0, 5.0)
        prices.append((ts, price))
        base_price = price
    service._price_history[symbol] = prices


def bench_portfolio_pnl_direct():
    print("\n=== Portfolio P&L Computation (Direct, no network) ===")
    service = _make_service()

    # Add positions -- use symbols with seeded price history
    for i in range(50):
        sym = f"SYN-{i}"
        service.upsert_position(sym, random.uniform(1.0, 100.0), random.uniform(10.0, 500.0))
        _seed_price_history(service, sym)
        # Warm price cache so get_portfolio doesn't hit the network
        service._price_cache[sym] = _make_cached_value(100.0 + random.uniform(-10, 10))

    def compute_portfolio():
        return service.get_portfolio()

    _timed("get_portfolio (50 positions, cached prices)", compute_portfolio, n=1000)


def _make_cached_value(price: float):
    from yggdrasil.node.services.trading import _CachedValue
    return _CachedValue(price, time.time())


def bench_indicators_direct():
    print("\n=== Technical Indicators (Direct, 200 data points) ===")
    service = _make_service()
    symbol = "BENCH-IND"
    _seed_price_history(service, symbol, 200)
    # Warm cache so get_indicators doesn't try network calls
    service._price_cache[symbol] = _make_cached_value(105.0)

    def compute_indicators():
        return service.get_indicators(symbol)

    _timed("get_indicators (SMA20, SMA50, EMA20, RSI14)", compute_indicators, n=1000)


def bench_sma_direct():
    print("\n=== SMA/EMA/RSI raw computation ===")
    prices = [100.0 + random.uniform(-5.0, 5.0) for _ in range(200)]

    from yggdrasil.node.services.trading import TradingService

    _timed("SMA(20) on 200 prices", lambda: TradingService._sma(prices, 20), n=1000)
    _timed("SMA(50) on 200 prices", lambda: TradingService._sma(prices, 50), n=1000)
    _timed("EMA(20) on 200 prices", lambda: TradingService._ema(prices, 20), n=1000)
    _timed("RSI(14) on 200 prices", lambda: TradingService._rsi(prices, 14), n=1000)


def bench_alert_checking_direct():
    print("\n=== Alert Checking (Direct) ===")
    from yggdrasil.node.schemas.trading import AlertCreate

    service = _make_service()

    # Seed alerts
    for i in range(100):
        service.set_alert(AlertCreate(
            symbol="BENCH-ALERT",
            condition="above" if i % 2 == 0 else "below",
            price=float(50 + i),
        ))

    # Seed price cache so check_alerts doesn't hit network
    service._price_cache["BENCH-ALERT"] = _make_cached_value(75.0)

    _timed("check_alerts (100 alerts, cached price)", lambda: service.check_alerts(), n=1000)


if __name__ == "__main__":
    print(f"Trading Service Benchmarks")
    print(f"Node URL: {BASE_URL}")

    live = _node_available()
    if live:
        print("Node is reachable -- running live API benchmarks.")
        bench_price_fetch_live()
        bench_portfolio_live()
        bench_alerts_live()
    else:
        print("Node not reachable -- skipping live API benchmarks.")

    print("\n--- Direct Service Benchmarks (no network) ---")
    bench_portfolio_pnl_direct()
    bench_indicators_direct()
    bench_sma_direct()
    bench_alert_checking_direct()

    print("\nDone.")
