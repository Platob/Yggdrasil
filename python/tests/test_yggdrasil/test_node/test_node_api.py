"""Tests for the yggdrasil.node trading API."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.node.app import create_api
from yggdrasil.node.config import Settings
from yggdrasil.node.api.services.market import MarketDataService, ASSETS
from yggdrasil.node.api.services.portfolio import PortfolioService, DEMO_PORTFOLIO_ID
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.schemas.analysis import (
    AggregateRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    AggMeasure,
    SeriesRequest,
)
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    write_arrow_stream,
    read_arrow_stream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market_svc():
    return MarketDataService()


@pytest.fixture(scope="module")
def portfolio_svc(market_svc):
    return PortfolioService(market_svc)


@pytest.fixture(scope="module")
def tmp_home(tmp_path_factory):
    return tmp_path_factory.mktemp("node_home")


@pytest.fixture(scope="module")
def analysis_svc(tmp_home):
    return AnalysisService(tmp_home)


@pytest.fixture(scope="module")
def parquet_file(tmp_home):
    n = 1000
    table = pa.table({
        "id": list(range(n)),
        "price": [100.0 + (i % 50) * 0.5 for i in range(n)],
        "sector": [["Tech", "Energy", "Finance"][i % 3] for i in range(n)],
    })
    path = tmp_home / "test.parquet"
    pq.write_table(table, str(path))
    return "test.parquet"


# ---------------------------------------------------------------------------
# Market service tests
# ---------------------------------------------------------------------------


def test_market_assets(market_svc):
    assets = asyncio.run(market_svc.get_assets())
    assert len(assets) == 8
    symbols = {a.symbol for a in assets}
    assert "BTC/USD" in symbols
    assert "EUR/USD" in symbols


def test_market_candles_btc(market_svc):
    candles = asyncio.run(market_svc.get_candles("BTC/USD", "1h", 50))
    assert len(candles) == 50
    for c in candles:
        assert c.high >= c.close >= 0
        assert c.high >= c.open >= 0
        assert c.high >= c.low >= 0
        assert c.volume > 0
        assert c.symbol == "BTC/USD"
        assert c.interval == "1h"


def test_market_candles_cache(market_svc):
    # Two calls should return identical data (cache hit within TTL).
    c1 = asyncio.run(market_svc.get_candles("ETH/USD", "1h", 10))
    c2 = asyncio.run(market_svc.get_candles("ETH/USD", "1h", 10))
    assert [c.ts for c in c1] == [c.ts for c in c2]


def test_market_candles_cached_json(market_svc):
    json_bytes, arrow_table = market_svc.get_candles_cached("SPY", "1d", 30)
    import orjson
    data = orjson.loads(json_bytes)
    assert data["symbol"] == "SPY"
    assert len(data["candles"]) == 30
    assert arrow_table.num_rows == 30


def test_market_tick(market_svc):
    tick = asyncio.run(market_svc.get_tick("BTC/USD"))
    assert tick.symbol == "BTC/USD"
    assert tick.price > 0
    assert tick.side in ("buy", "sell")


def test_market_book(market_svc):
    book = asyncio.run(market_svc.get_book("ETH/USD", depth=5))
    assert book.symbol == "ETH/USD"
    assert len(book.bids) == 5
    assert len(book.asks) == 5
    # Bids should be below asks.
    assert book.bids[0][0] < book.asks[0][0]


def test_market_unknown_symbol(market_svc):
    with pytest.raises(KeyError):
        asyncio.run(market_svc.get_candles("FAKE/XXX", "1h", 10))


# ---------------------------------------------------------------------------
# Portfolio service tests
# ---------------------------------------------------------------------------


def test_portfolio_get(portfolio_svc):
    p = asyncio.run(portfolio_svc.get_portfolio(DEMO_PORTFOLIO_ID))
    assert p.name == "Demo Book"
    assert p.equity > 0
    assert len(p.positions) > 0


def test_portfolio_summary(portfolio_svc):
    s = asyncio.run(portfolio_svc.get_summary(DEMO_PORTFOLIO_ID))
    assert s.equity > 0
    assert 0.0 <= s.win_rate <= 1.0
    assert s.position_count >= 0


def test_portfolio_trades(portfolio_svc):
    trades = asyncio.run(portfolio_svc.get_trades(DEMO_PORTFOLIO_ID, limit=10))
    assert isinstance(trades, list)
    assert len(trades) <= 10


def test_portfolio_unknown(portfolio_svc):
    with pytest.raises(KeyError):
        asyncio.run(portfolio_svc.get_portfolio(999999))


# ---------------------------------------------------------------------------
# Analysis service tests
# ---------------------------------------------------------------------------


def test_analysis_aggregate_grouped(analysis_svc, parquet_file):
    req = AggregateRequest(path=parquet_file, column="price", agg="mean", group_by="sector")
    res = asyncio.run(analysis_svc.aggregate(req))
    assert len(res.rows) == 3  # Tech, Energy, Finance
    assert res.group_by == "sector"
    assert all(r.value is not None for r in res.rows)


def test_analysis_aggregate_ungrouped(analysis_svc, parquet_file):
    req = AggregateRequest(path=parquet_file, column="price", agg="sum")
    res = asyncio.run(analysis_svc.aggregate(req))
    assert len(res.rows) == 1
    assert res.rows[0].value > 0


def test_analysis_aggregate_schema_discovery(analysis_svc, parquet_file):
    req = AggregateRequest(path=parquet_file, column="*", agg="count")
    res = asyncio.run(analysis_svc.aggregate(req))
    assert "price" in res.columns
    assert "sector" in res.columns
    assert len(res.rows) == 0  # schema discovery returns no rows


def test_analysis_aggregate_series(analysis_svc, parquet_file):
    req = AggregateRequest(path=parquet_file, column="price", agg="series")
    res = asyncio.run(analysis_svc.aggregate(req))
    assert len(res.rows) == 1000
    assert all(r.value is not None for r in res.rows)


def test_analysis_series_downsample(analysis_svc, parquet_file):
    req = SeriesRequest(path=parquet_file, column="price", points=100)
    res = asyncio.run(analysis_svc.series(req))
    assert res.points <= 100
    assert len(res.x) == res.points
    assert len(res.y) == res.points


def test_analysis_ohlc(analysis_svc, parquet_file):
    req = OhlcRequest(path=parquet_file, column="price", buckets=50)
    res = asyncio.run(analysis_svc.ohlc(req))
    assert res.bars <= 50
    assert len(res.open) == res.bars
    for i in range(res.bars):
        assert res.high[i] >= res.open[i]
        assert res.high[i] >= res.close[i]
        assert res.low[i] <= res.open[i]
        assert res.low[i] <= res.close[i]


def test_analysis_pivot(analysis_svc, parquet_file):
    req = PivotRequest(
        path=parquet_file,
        rows=["sector"],
        columns=["sector"],
        measures=[AggMeasure(column="price", agg="mean")],
    )
    res = asyncio.run(analysis_svc.pivot(req))
    assert res.row_count == 3
    assert res.col_count >= 3


def test_analysis_forecast(analysis_svc, parquet_file):
    req = ForecastRequest(path=parquet_file, column="price", x="id", horizon=5)
    res = asyncio.run(analysis_svc.forecast(req))
    assert res.model_used == "ridge"
    assert len(res.series) == 1
    assert len(res.series[0].forecast) == 5
    assert res.series[0].rmse >= 0


def test_analysis_path_escape(analysis_svc):
    req = AggregateRequest(path="../../../etc/passwd", column="x", agg="count")
    with pytest.raises(PermissionError):
        asyncio.run(analysis_svc.aggregate(req))


# ---------------------------------------------------------------------------
# Transport tests
# ---------------------------------------------------------------------------


def test_transport_arrow_round_trip():
    table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    buf = write_arrow_stream(table)
    assert isinstance(buf, bytes)
    assert len(buf) > 0
    recovered = read_arrow_stream(buf)
    assert recovered.num_rows == 3
    assert recovered.column_names == ["x", "y"]


# ---------------------------------------------------------------------------
# FastAPI integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_ping():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/ping")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "ts" in body
    assert body["node_id"] == "ygg"


@pytest.mark.asyncio
async def test_api_health():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v2/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_api_candles_json():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v2/market/candles?symbol=BTC%2FUSD&interval=1h&limit=20")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "BTC/USD"
    assert len(data["candles"]) == 20


@pytest.mark.asyncio
async def test_api_candles_arrow():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(
            "/api/v2/market/candles?symbol=ETH%2FUSD&interval=1h&limit=10",
            headers={"Accept": CONTENT_TYPE_ARROW_STREAM},
        )
    assert r.status_code == 200
    assert r.headers["content-type"] == CONTENT_TYPE_ARROW_STREAM
    tbl = read_arrow_stream(r.content)
    assert tbl.num_rows == 10
    assert "close" in tbl.column_names


@pytest.mark.asyncio
async def test_api_assets():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v2/market/assets")
    assert r.status_code == 200
    assets = r.json()
    assert len(assets) == 8


@pytest.mark.asyncio
async def test_api_portfolio():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/api/v2/portfolio/{DEMO_PORTFOLIO_ID}")
    assert r.status_code == 200
    p = r.json()
    assert p["name"] == "Demo Book"
    assert p["equity"] > 0


@pytest.mark.asyncio
async def test_api_unknown_symbol_404():
    import httpx

    app = create_api()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/v2/market/candles?symbol=FAKE&interval=1h&limit=5")
    assert r.status_code == 404
