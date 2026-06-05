from __future__ import annotations

from typing import Iterable, Mapping, Any

from .base import GeoZone, GeoZoneType
from .catalog import GeoZoneCatalog
from .continents import load_continent_geozones
from .countries import fetch_country_geozones
from .entsoe import fetch_entsoe_bidding_zones

__all__ = ["DEFAULT_GEOZONES", "load_geozones"]

DEFAULT_GEOZONES: tuple[GeoZone, ...] = (
    GeoZone(
        gtype=GeoZoneType.EARTH,
        key="WORLD",
        name="World",
        country_iso=None,
        region_iso="WORLD",
        ccy=None,
        lat=0.0,
        lon=0.0,
        aliases=("EARTH", "GLOBAL"),
    ),
    *load_continent_geozones(),
    GeoZone(
        gtype=GeoZoneType.COUNTRY,
        key="FR",
        name="France",
        country_iso="FR",
        region_iso="FR-IDF",
        sub_iso="FR-IDF",
        ccy="EUR",
        lat=46.2276,
        lon=2.2137,
        aliases=("FRA",),
    ),
    GeoZone(
        gtype=GeoZoneType.COUNTRY,
        key="DE",
        name="Germany",
        country_iso="DE",
        region_iso="DE-BE",
        sub_iso="DE-BE",
        ccy="EUR",
        lat=51.1657,
        lon=10.4515,
        aliases=("DEU", "GERMANY"),
    ),
    GeoZone(
        gtype=GeoZoneType.CITY,
        key="CH-ZH",
        name="Zurich",
        country_iso="CH",
        region_iso="CH-ZH",
        sub_iso="CH-ZH",
        ccy="CHF",
        lat=47.3769,
        lon=8.5417,
        aliases=("ZURICH", "ZUERICH"),
    ),
)

_DEFAULT_CATALOG: GeoZoneCatalog | None = None


def load_geozones(
    values: Iterable[GeoZone | Mapping[str, Any]] | None = None,
    *,
    include_countries: bool = False,
    include_entsoe_bidding_zones: bool = False,
) -> GeoZoneCatalog:
    global _DEFAULT_CATALOG

    if values is None and not include_countries and not include_entsoe_bidding_zones:
        if _DEFAULT_CATALOG is None:
            _DEFAULT_CATALOG = GeoZoneCatalog.from_values(DEFAULT_GEOZONES)
        return _DEFAULT_CATALOG

    catalog_values: list[GeoZone | Mapping[str, Any]] = list(DEFAULT_GEOZONES if values is None else values)
    if include_countries:
        catalog_values.extend(fetch_country_geozones())
    if include_entsoe_bidding_zones:
        catalog_values.extend(fetch_entsoe_bidding_zones())

    return GeoZoneCatalog.from_values(catalog_values)
