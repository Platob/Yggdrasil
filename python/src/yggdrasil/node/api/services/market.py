"""Market data service — FX via Frankfurter (free, no auth), energy via ENTSO-E.

FX uses the public Frankfurter API so it works without credentials. Results
are cached in-process for ``_FX_TTL`` seconds so repeated dashboard refreshes
don't hammer the upstream. Energy goes through ``yggdrasil.loki.entsoe`` and
needs ``ENTSOE_API_TOKEN``, returning a helpful hint when the token is absent.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Any

_FX_TTL = 30.0  # seconds — Frankfurter publishes daily rates so 30s is fine


class MarketService:
    """FX + energy market data, with a short TTL in-process cache for FX."""

    def __init__(self, settings: object | None = None) -> None:
        self._settings = settings
        self._fx_cache: dict[str, tuple[float, dict]] = {}  # pair → (expires_at, result)
        self._pending: dict[str, asyncio.Task] = {}  # coalesce concurrent same-pair fetches

    async def get_fx(
        self,
        pairs: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        if start or end:
            # Historical path — no caching, call Frankfurter time series
            return await self._get_fx_history(pairs, start, end)

        parsed = [p.strip().split("/") for p in pairs if "/" in p and len(p.strip().split("/")) == 2]
        if not parsed:
            return {"error": "no valid pairs", "hint": "pass pairs like 'EUR/USD'"}

        now = time.monotonic()
        hits = {f"{s}/{t}": v for (s, t) in parsed if (k := f"{s}/{t}") in self._fx_cache and self._fx_cache[k][0] > now for v in [self._fx_cache[k][1]]}
        misses = [(s, t) for s, t in parsed if f"{s}/{t}" not in hits]

        if misses:
            await self._fetch_fx_batch(misses)
            now = time.monotonic()
            hits = {f"{s}/{t}": self._fx_cache[f"{s}/{t}"][1] for s, t in parsed if f"{s}/{t}" in self._fx_cache}

        results = [hits.get(f"{s}/{t}", {"pair": f"{s}/{t}", "error": "fetch failed"}) for s, t in parsed]
        return {"rates": results}

    async def _fetch_fx_batch(self, pairs: list[tuple[str, str]]) -> None:
        import httpx

        expires = time.monotonic() + _FX_TTL
        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks = {}
            for source, target in pairs:
                key = f"{source}/{target}"
                if key not in tasks:
                    tasks[key] = asyncio.create_task(
                        client.get(f"https://api.frankfurter.app/latest?from={source}&to={target}")
                    )

            for (source, target), task in zip(pairs, tasks.values()):
                key = f"{source}/{target}"
                try:
                    r = await task
                    data = r.json()
                    self._fx_cache[key] = (
                        expires,
                        {
                            "pair": key,
                            "rate": data.get("rates", {}).get(target),
                            "date": data.get("date"),
                            "base": source,
                        },
                    )
                except Exception as exc:
                    self._fx_cache[key] = (expires, {"pair": key, "error": str(exc)})

    async def _get_fx_history(
        self,
        pairs: list[str],
        start: str | None,
        end: str | None,
    ) -> dict:
        import httpx

        parsed = [p.strip().split("/") for p in pairs if "/" in p and len(p.strip().split("/")) == 2]
        if not parsed:
            return {"error": "no valid pairs"}

        now = dt.date.today()
        start_date = start or (now - dt.timedelta(days=30)).isoformat()
        end_date = end or now.isoformat()

        results: list[dict[str, Any]] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for source, target in parsed:
                url = f"https://api.frankfurter.app/{start_date}..{end_date}?from={source}&to={target}"
                try:
                    r = await client.get(url)
                    data = r.json()
                    rates_ts = [
                        {"date": d, "rate": vals.get(target)}
                        for d, vals in sorted(data.get("rates", {}).items())
                    ]
                    results.append({"pair": f"{source}/{target}", "base": source, "history": rates_ts})
                except Exception as exc:
                    results.append({"pair": f"{source}/{target}", "error": str(exc)})
        return {"rates": results}

    async def get_energy(
        self,
        zone: str,
        series: str = "day_ahead_prices",
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        from yggdrasil.loki import entsoe

        token = entsoe.token()
        if not token:
            return {
                "error": "ENTSOE_API_TOKEN not set",
                "hint": "Set the env var to fetch energy data.",
            }

        now = dt.datetime.now(tz=dt.timezone.utc)
        if start is None:
            start = (now - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        if end is None:
            end = now.strftime("%Y-%m-%d")

        try:
            frame = entsoe.fetch_frame(series=series, zone=zone, start=start, end=end)
            return {"zone": zone, "series": series, "data": frame.to_dicts()}
        except Exception as exc:
            return {"error": str(exc), "zone": zone, "series": series}
