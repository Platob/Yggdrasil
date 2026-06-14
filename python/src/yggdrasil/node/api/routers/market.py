"""Market data endpoints — FX (Frankfurter), crypto (CoinGecko), energy (ENTSO-E)."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()

_DEFAULT_FX = "EUR/USD,EUR/GBP,EUR/JPY,USD/JPY,GBP/USD,USD/CHF"
_DEFAULT_CRYPTO = "bitcoin,ethereum,solana,cardano"


@router.get("/fx")
async def get_fx(
    request: Request,
    pairs: str = Query(_DEFAULT_FX),
    start: str | None = Query(None, description="ISO date for historical range start"),
    end: str | None = Query(None, description="ISO date for historical range end"),
    sampling: str = Query("1d"),
):
    pair_list = [p.strip() for p in pairs.split(",") if "/" in p]
    return await request.app.state.market.get_fx(pair_list, start=start, end=end)


@router.get("/energy")
async def get_energy(
    request: Request,
    zone: str = Query("DE_LU"),
    series: str = Query("day_ahead_prices"),
    start: str | None = Query(None),
    end: str | None = Query(None),
):
    return await request.app.state.market.get_energy(zone, series=series, start=start, end=end)


@router.get("/crypto")
async def get_crypto(
    coins: str = Query(_DEFAULT_CRYPTO, description="Comma-separated CoinGecko IDs"),
    vs: str = Query("usd", description="Quote currency"),
):
    """Live crypto prices via CoinGecko public API (no auth required)."""
    import httpx

    ids = ",".join(c.strip() for c in coins.split(",") if c.strip())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies={vs}&include_24hr_change=true"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, headers={"Accept": "application/json"})
            data = r.json()
        results = [
            {
                "id": coin_id,
                "price": vals.get(vs),
                "change_24h": vals.get(f"{vs}_24h_change"),
                "vs": vs,
            }
            for coin_id, vals in data.items()
        ]
        return {"prices": results}
    except Exception as exc:
        return {"error": str(exc), "hint": "CoinGecko public API may be rate-limited; retry shortly"}


@router.get("/summary")
async def get_market_summary(request: Request):
    """Single-call market snapshot: key FX rates + crypto + node stats."""
    import asyncio

    fx_task = asyncio.create_task(
        request.app.state.market.get_fx(
            ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"],
        )
    )

    import httpx
    crypto_url = (
        "https://api.coingecko.com/api/v3/simple/price"
        "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
    )
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            cr = await client.get(crypto_url)
            crypto_data = cr.json()
    except Exception:
        crypto_data = {}

    fx_result = await fx_task

    crypto = [
        {
            "id": cid,
            "price": vals.get("usd"),
            "change_24h": vals.get("usd_24h_change"),
        }
        for cid, vals in crypto_data.items()
    ]

    snap = request.app.state.monitor.snapshot()

    return {
        "fx": fx_result.get("rates", []),
        "crypto": crypto,
        "node": {
            "cpu_percent": snap.cpu_percent,
            "mem_percent": snap.mem_percent,
        },
    }
