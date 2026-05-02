"""Live-API integration tests for ``fetch_country_geozones``.

Marked with ``@pytest.mark.integration`` so the default ``pytest`` run
skips them — they're opt-in via ``-m integration``. Each test pings
the live REST Countries endpoint; if it can't reach the server, the
test skips rather than failing.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import fetch_country_geozones, load_geozones


@pytest.mark.integration
class TestLiveCountryFetch:

    def test_fetch_returns_at_least_two_hundred_zones(self) -> None:
        try:
            zones = fetch_country_geozones()
        except Exception as exc:
            pytest.skip(f"live country API unavailable: {exc}")

        assert len(zones) >= 200

    def test_france_record_is_well_formed(self) -> None:
        try:
            zones = fetch_country_geozones()
        except Exception as exc:
            pytest.skip(f"live country API unavailable: {exc}")

        france = next(
            (zone for zone in zones if zone.country_iso == "FR"), None
        )
        assert france is not None
        assert france.name == "France"
        assert -90.0 <= france.lat <= 90.0
        assert -180.0 <= france.lon <= 180.0


@pytest.mark.integration
class TestLiveLoadGeozonesEnrichment:

    def test_country_enrichment_resolves_alpha3_alias(self) -> None:
        try:
            catalog = load_geozones(include_countries=True)
        except Exception as exc:
            pytest.skip(f"live country API unavailable: {exc}")

        germany = catalog.parse("DEU")
        assert germany is not None
        assert germany.country_iso == "DE"
        assert germany.name == "Germany"
