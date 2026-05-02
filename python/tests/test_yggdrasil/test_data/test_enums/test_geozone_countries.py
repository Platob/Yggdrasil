"""``fetch_country_geozones`` — REST Countries API ingestion.

Fake-session-driven tests pin the parser shape: ``cca2`` becomes
``country_iso``, ``subregion`` becomes ``region_iso`` (uppercased),
the first currency key becomes ``ccy``, and ``cca3`` is added as the
first alias. The live integration variant lives in
``test_geozone_countries_integration.py``.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import (
    GeoZone,
    GeoZoneType,
    fetch_country_geozones,
    load_geozones,
)


class _FakeResponse:

    def __init__(self, payload) -> None:
        self._payload = payload

    def json(self):
        return self._payload


class _FakeSession:

    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.payload)


def _country_record(
    cca2: str,
    cca3: str,
    common_name: str,
    official_name: str,
    latlng: list,
    region: str,
    subregion: str,
    currency_code: str = "EUR",
) -> dict:
    return {
        "cca2": cca2,
        "cca3": cca3,
        "name": {"common": common_name, "official": official_name},
        "latlng": latlng,
        "altSpellings": [cca2, official_name],
        "region": region,
        "subregion": subregion,
        "currencies": {currency_code: {"name": "Euro", "symbol": "€"}},
    }


class TestFetchCountriesParsesPayload:

    def test_two_country_payload(self) -> None:
        session = _FakeSession(
            [
                _country_record(
                    "FR", "FRA", "France", "French Republic",
                    [46.0, 2.0], "Europe", "Western Europe",
                ),
                _country_record(
                    "DE", "DEU", "Germany", "Federal Republic of Germany",
                    [51.0, 10.0], "Europe", "Western Europe",
                ),
            ]
        )

        zones = fetch_country_geozones(session=session)

        assert [zone.country_iso for zone in zones] == ["FR", "DE"]
        assert all(zone.gtype == GeoZoneType.COUNTRY for zone in zones)

    def test_alpha3_iso_is_first_alias(self) -> None:
        session = _FakeSession(
            [
                _country_record(
                    "FR", "FRA", "France", "French Republic",
                    [46.0, 2.0], "Europe", "Western Europe",
                )
            ]
        )

        zones = fetch_country_geozones(session=session)

        assert zones[0].aliases[0] == "FRA"

    def test_currency_code_propagates(self) -> None:
        session = _FakeSession(
            [
                _country_record(
                    "FR", "FRA", "France", "French Republic",
                    [46.0, 2.0], "Europe", "Western Europe",
                )
            ]
        )

        zones = fetch_country_geozones(session=session)

        assert zones[0].ccy == "EUR"

    def test_subregion_is_uppercased_into_region_iso(self) -> None:
        session = _FakeSession(
            [
                _country_record(
                    "FR", "FRA", "France", "French Republic",
                    [46.0, 2.0], "Europe", "Western Europe",
                )
            ]
        )

        zones = fetch_country_geozones(session=session)

        assert zones[0].region_iso == "WESTERN EUROPE"

    def test_request_targets_expected_field_set(self) -> None:
        session = _FakeSession([])

        fetch_country_geozones(session=session)

        assert session.calls[0][1]["params"]["fields"] == (
            "cca2,cca3,name,latlng,altSpellings,region,subregion,currencies"
        )


class TestLoadGeozonesWithCountries:

    def test_include_countries_weaves_fetched_records_into_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "yggdrasil.data.enums.geozone.load.fetch_country_geozones",
            lambda: (
                GeoZone(
                    gtype=GeoZoneType.COUNTRY,
                    key="ES",
                    name="Spain",
                    country_iso="ES",
                    region_iso="Southern Europe",
                    ccy="EUR",
                    lat=40.4637,
                    lon=-3.7492,
                    aliases=("ESP",),
                ),
            ),
        )

        catalog = load_geozones(include_countries=True)

        zone = catalog.parse("esp")
        assert zone is not None
        assert zone.name == "Spain"
