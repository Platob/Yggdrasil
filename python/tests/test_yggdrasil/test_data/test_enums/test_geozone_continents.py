from __future__ import annotations

from yggdrasil.data.enums.geozone import GeoZoneType, load_continent_geozones


def test_load_continent_geozones_returns_seeded_continents() -> None:
    zones = load_continent_geozones()

    assert len(zones) >= 7

    europe = next((zone for zone in zones if zone.key == "EU"), None)
    assert europe is not None
    assert europe.gtype == GeoZoneType.CONTINENT
    assert europe.name == "Europe"
    assert europe.ccy == "EUR"
