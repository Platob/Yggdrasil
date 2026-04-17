from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import GeoZone, GeoZoneType, fetch_country_geozones, load_geozones


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


def test_fetch_country_geozones_from_http_source() -> None:
    session = _FakeSession(
        [
            {
                "cca2": "FR",
                "cca3": "FRA",
                "name": {"common": "France", "official": "French Republic"},
                "latlng": [46.0, 2.0],
                "altSpellings": ["FR", "French Republic"],
                "region": "Europe",
                "subregion": "Western Europe",
                "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
            },
            {
                "cca2": "DE",
                "cca3": "DEU",
                "name": {"common": "Germany", "official": "Federal Republic of Germany"},
                "latlng": [51.0, 10.0],
                "altSpellings": ["DE", "Federal Republic of Germany"],
                "region": "Europe",
                "subregion": "Western Europe",
                "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
            },
        ]
    )

    zones = fetch_country_geozones(session=session)

    assert [zone.country_iso for zone in zones] == ["FR", "DE"]
    assert [zone.gtype for zone in zones] == [GeoZoneType.COUNTRY, GeoZoneType.COUNTRY]
    assert zones[0].aliases[0] == "FRA"
    assert zones[0].ccy == "EUR"
    assert zones[0].region_iso == "WESTERN EUROPE"
    assert session.calls[0][1]["params"]["fields"] == "cca2,cca3,name,latlng,altSpellings,region,subregion,currencies"


def test_load_geozones_can_include_fetched_countries(monkeypatch: pytest.MonkeyPatch) -> None:
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
