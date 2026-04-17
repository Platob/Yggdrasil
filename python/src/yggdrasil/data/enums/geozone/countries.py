from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from yggdrasil.io.http_ import HTTPSession

from .base import GeoZone, GeoZoneType, normalize_geozone_name

__all__ = [
    "REST_COUNTRIES_URL",
    "fetch_country_geozones",
]


REST_COUNTRIES_URL = "https://restcountries.com/v3.1/all"
_REST_COUNTRIES_FIELDS = "cca2,cca3,name,latlng,altSpellings,region,subregion,currencies"


def _coerce_country_zone(payload: Mapping[str, Any]) -> GeoZone | None:
    cca2 = payload.get("cca2")
    if not isinstance(cca2, str) or not cca2.strip():
        return None

    latlng = payload.get("latlng")
    if not isinstance(latlng, list | tuple) or len(latlng) < 2:
        return None

    try:
        lat = float(latlng[0])
        lon = float(latlng[1])
    except (TypeError, ValueError):
        return None

    name = payload.get("name")
    common_name = name.get("common") if isinstance(name, Mapping) else None
    official_name = name.get("official") if isinstance(name, Mapping) else None

    aliases: list[str] = []
    cca3 = payload.get("cca3")
    if isinstance(cca3, str) and cca3.strip():
        aliases.append(cca3)

    for value in payload.get("altSpellings", ()) or ():
        if isinstance(value, str) and value.strip():
            aliases.append(value)

    if isinstance(official_name, str) and official_name.strip():
        aliases.append(official_name)

    region = normalize_geozone_name(payload.get("region"))
    subregion = normalize_geozone_name(payload.get("subregion"))
    region_iso = subregion or region
    currencies = payload.get("currencies")
    ccy = None
    if isinstance(currencies, Mapping):
        for currency_code in currencies.keys():
            if isinstance(currency_code, str) and currency_code.strip():
                ccy = currency_code
                break

    return GeoZone(
        gtype=GeoZoneType.COUNTRY,
        key=cca2,
        name=common_name or official_name or cca2,
        country_iso=cca2,
        region_iso=region_iso,
        ccy=ccy,
        lat=lat,
        lon=lon,
        aliases=tuple(aliases),
    )


def fetch_country_geozones(
    *,
    session: HTTPSession | None = None,
    url: str = REST_COUNTRIES_URL,
) -> tuple[GeoZone, ...]:
    owns_session = session is None
    http = session or HTTPSession()

    try:
        response = http.get(
            url,
            params={"fields": _REST_COUNTRIES_FIELDS},
            headers={"Accept": "application/json"},
        )
        payload = response.json()
    finally:
        if owns_session:
            http.__exit__(None, None, None)

    if not isinstance(payload, list):
        raise ValueError("REST Countries response must be a list")

    zones: list[GeoZone] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        zone = _coerce_country_zone(item)
        if zone is not None:
            zones.append(zone)

    zones.sort(key=lambda zone: (zone.name or "", zone.country_iso or ""))
    return tuple(zones)
