from __future__ import annotations

from .base import GeoZone, GeoZoneType

__all__ = ["DEFAULT_CONTINENTS", "load_continent_geozones"]


DEFAULT_CONTINENTS: tuple[GeoZone, ...] = (
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="AF",
        name="Africa",
        region_iso="AF",
        lat=1.6508,
        lon=17.6791,
        aliases=("AFRICA",),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="AN",
        name="Antarctica",
        region_iso="AN",
        lat=-82.8628,
        lon=135.0,
        aliases=("ANTARCTICA",),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="AS",
        name="Asia",
        region_iso="AS",
        lat=34.0479,
        lon=100.6197,
        aliases=("ASIA",),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="EU",
        name="Europe",
        region_iso="EU",
        ccy="EUR",
        lat=54.5260,
        lon=15.2551,
        aliases=("EUROPE",),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="NA",
        name="North America",
        region_iso="NA",
        lat=54.5260,
        lon=-105.2551,
        aliases=("NORTH AMERICA", "NORTHAMERICA"),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="OC",
        name="Oceania",
        region_iso="OC",
        lat=-22.7359,
        lon=140.0188,
        aliases=("OCEANIA",),
    ),
    GeoZone(
        gtype=GeoZoneType.CONTINENT,
        key="SA",
        name="South America",
        region_iso="SA",
        lat=-8.7832,
        lon=-55.4915,
        aliases=("SOUTH AMERICA", "SOUTHAMERICA"),
    ),
)


def load_continent_geozones() -> tuple[GeoZone, ...]:
    return DEFAULT_CONTINENTS
