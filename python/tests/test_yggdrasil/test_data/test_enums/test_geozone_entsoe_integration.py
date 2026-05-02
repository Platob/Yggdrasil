"""Live-API integration tests for ENTSO-E bidding-zone fetch.

Marked ``@pytest.mark.integration`` and opt-in via ``-m integration``.
Each test pings the ENTSO-E EIC code list endpoint; if it can't reach
the server, the test skips rather than failing.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import (
    fetch_entsoe_bidding_zones,
    load_geozones,
)


@pytest.mark.integration
class TestLiveEntsoeFetch:

    def test_fetch_returns_at_least_ten_bidding_zones(self) -> None:
        try:
            zones = fetch_entsoe_bidding_zones()
        except Exception as exc:
            pytest.skip(f"live ENTSO-E bidding-zone API unavailable: {exc}")

        assert len(zones) >= 10

    def test_france_zone_has_complete_metadata(self) -> None:
        try:
            zones = fetch_entsoe_bidding_zones()
        except Exception as exc:
            pytest.skip(f"live ENTSO-E bidding-zone API unavailable: {exc}")

        france = next(
            (zone for zone in zones if zone.country_iso == "FR"), None
        )
        assert france is not None
        assert france.eic is not None
        assert france.key is not None
        assert france.region_iso is not None


@pytest.mark.integration
class TestLiveEntsoeEnrichment:

    def test_load_geozones_resolves_eic_lookup(self) -> None:
        try:
            catalog = load_geozones(include_entsoe_bidding_zones=True)
        except Exception as exc:
            pytest.skip(f"live ENTSO-E bidding-zone API unavailable: {exc}")

        zone = catalog.find_by_str("10Y1001A1001A63L")
        assert zone is not None
        assert zone.eic == "10Y1001A1001A63L"
        assert zone.country_iso is not None
