"""Market data service — FX via Frankfurter (free, no auth), energy via ENTSO-E.

FX uses the public Frankfurter API so it works without credentials; energy goes
through ``yggdrasil.loki.entsoe`` and needs ``ENTSOE_API_TOKEN``, returning a
helpful hint when the token is absent rather than failing opaquely.
"""
from __future__ import annotations

import datetime as dt


class MarketService:
    """FX + energy market data."""

    def __init__(self, settings: object | None = None) -> None:
        self._settings = settings

    async def get_fx(
        self,
        pairs: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        import httpx

        parsed = [p.strip().split("/") for p in pairs if "/" in p]
        if not parsed:
            return {"error": "no valid pairs", "hint": "pass pairs like 'EUR/USD'"}

        results: list[dict] = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for source, target in parsed:
                url = f"https://api.frankfurter.app/latest?from={source}&to={target}"
                try:
                    r = await client.get(url)
                    data = r.json()
                    results.append(
                        {
                            "pair": f"{source}/{target}",
                            "rate": data.get("rates", {}).get(target),
                            "date": data.get("date"),
                            "base": source,
                        }
                    )
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
