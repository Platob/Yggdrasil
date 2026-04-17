from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import fetch_country_geozones, load_geozones


@pytest.mark.integration
def test_fetch_country_geozones_real_api() -> None:
    try:
        zones = fetch_country_geozones()
    except Exception as exc:
        pytest.skip(f"live country API unavailable: {exc}")

    assert len(zones) >= 200

    france = next((zone for zone in zones if zone.country_iso == "FR"), None)
    assert france is not None
    assert france.name == "France"
    assert -90.0 <= france.lat <= 90.0
    assert -180.0 <= france.lon <= 180.0


@pytest.mark.integration
def test_load_geozones_with_real_country_enrichment() -> None:
    try:
        catalog = load_geozones(include_countries=True)
    except Exception as exc:
        pytest.skip(f"live country API unavailable: {exc}")

    germany = catalog.parse("DEU")
    assert germany is not None
    assert germany.country_iso == "DE"
    assert germany.name == "Germany"
