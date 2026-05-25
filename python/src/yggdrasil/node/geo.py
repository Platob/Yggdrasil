"""Best-effort IP geolocation — no external dependencies."""
from __future__ import annotations

import json
import logging
import urllib.request

LOGGER = logging.getLogger(__name__)

_cache: tuple[float, float] | None = None


def get_location() -> tuple[float | None, float | None]:
    """Return (lat, lon) by querying a free IP geolocation API. Cached after first call."""
    global _cache
    if _cache is not None:
        return _cache
    for url in (
        "http://ip-api.com/json/?fields=lat,lon",
        "https://ipapi.co/json/",
    ):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "yggdrasil-bot/0.1"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            lat = data.get("lat") or data.get("latitude")
            lon = data.get("lon") or data.get("longitude")
            if lat is not None and lon is not None:
                _cache = (float(lat), float(lon))
                return _cache
        except Exception as exc:
            LOGGER.debug("Geo lookup failed (%s): %s", url, exc)
    return None, None
