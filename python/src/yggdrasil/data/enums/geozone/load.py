from __future__ import annotations

from .cities import load_cities
from .core import load_core_geozones
from .country_zones import load_country_zones
from .countries import load_countries
from .geozone import GeoZone
from .mountains import load_mountains
from .zones_entsoe import load_entsoe_zones
from .zones_markets import load_market_zones

__all__ = ['load_geozones']

_LOADERS = (
    load_core_geozones,
    load_countries,
    load_cities,
    load_entsoe_zones,
    load_market_zones,
    load_country_zones,
    load_mountains,
)


def load_geozones() -> None:
    GeoZone.clear_cache()
    for loader in _LOADERS:
        loader()
    GeoZone.UK = GeoZone.UNITED_KINGDOM
    GeoZone.USA = GeoZone.UNITED_STATES
    GeoZone.US = GeoZone.UNITED_STATES
