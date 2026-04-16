from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import fetch_entsoe_bidding_zones, load_geozones


@pytest.mark.integration
def test_fetch_entsoe_bidding_zones_real_api() -> None:
    try:
        zones = fetch_entsoe_bidding_zones()
    except Exception as exc:
        pytest.skip(f"live ENTSO-E bidding-zone API unavailable: {exc}")

    assert len(zones) >= 10

    france = next((zone for zone in zones if zone.country_iso == "FR"), None)
    assert france is not None
    assert france.eic is not None
    assert france.key is not None
    assert france.region_iso is not None


@pytest.mark.integration
def test_load_geozones_with_real_entsoe_enrichment() -> None:
    try:
        catalog = load_geozones(include_entsoe_bidding_zones=True)
    except Exception as exc:
        pytest.skip(f"live ENTSO-E bidding-zone API unavailable: {exc}")

    zone = catalog.find_by_str("10Y1001A1001A63L")
    assert zone is not None
    assert zone.eic == "10Y1001A1001A63L"
    assert zone.country_iso is not None
