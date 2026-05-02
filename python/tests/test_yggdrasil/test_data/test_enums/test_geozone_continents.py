"""Seeded continent geozones — the static catalog.

The continent set is hardcoded: Europe / Africa / Asia / etc., with
canonical key + currency. Pinning the seed makes downstream catalog
lookups stable across runs that don't pull from any live API.
"""
from __future__ import annotations

from yggdrasil.data.enums.geozone import GeoZoneType, load_continent_geozones


class TestSeededContinents:

    def test_at_least_seven_continents(self) -> None:
        zones = load_continent_geozones()

        assert len(zones) >= 7

    def test_europe_is_present_and_typed(self) -> None:
        zones = load_continent_geozones()

        europe = next((zone for zone in zones if zone.key == "EU"), None)

        assert europe is not None
        assert europe.gtype == GeoZoneType.CONTINENT
        assert europe.name == "Europe"
        assert europe.ccy == "EUR"
