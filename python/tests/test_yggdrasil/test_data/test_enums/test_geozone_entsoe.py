from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import GeoZone, GeoZoneCatalog, GeoZoneType, fetch_entsoe_bidding_zones, load_geozones


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeSession:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.text)


_CSV_TEXT = """EicCode;EicDisplayName;EicLongName;EicParent;EicResponsibleParty;EicStatus;MarketParticipantPostalCode;MarketParticipantIsoCountryCode;MarketParticipantVatCode;EicTypeFunctionList;type
10Y1001A1001A44P;FR;France Bidding Zone;;10X1001A1001A39I;Active;;FR;;Bidding Zone;Y
10YDOM-CZ-DE-SKK;DE_LU;Germany-Luxembourg Bidding Zone;;10X1001A1001A83F;Active;;DE;;Bidding Zone;Y
11YNOT-BIDDING;FR_GRID;France Grid Area;;10X1001A1001A39I;Active;;FR;;Control Area;Y
"""


def test_fetch_entsoe_bidding_zones_from_csv() -> None:
    session = _FakeSession(_CSV_TEXT)
    countries = {
        "FR": GeoZone(gtype=GeoZoneType.COUNTRY, key="FR", name="France", country_iso="FR", region_iso="FR", lat=46.0, lon=2.0),
        "DE": GeoZone(gtype=GeoZoneType.COUNTRY, key="DE", name="Germany", country_iso="DE", region_iso="DE", lat=51.0, lon=10.0),
    }

    zones = fetch_entsoe_bidding_zones(session=session, countries=countries)

    assert [zone.key for zone in zones] == ["DE LU", "FR"]
    assert [zone.gtype for zone in zones] == [GeoZoneType.EIC, GeoZoneType.EIC]
    assert zones[0].eic == "10YDOM CZ DE SKK"
    assert zones[0].country_iso == "DE"
    assert zones[1].aliases[0] == "10Y1001A1001A44P"
    assert session.calls[0][0].endswith("Y_eiccodes.csv")


def test_catalog_find_by_str_returns_best_match() -> None:
    catalog = GeoZoneCatalog.from_values(
        [
            GeoZone(gtype=GeoZoneType.COUNTRY, key="FR", name="France", country_iso="FR", region_iso="FR", lat=46.0, lon=2.0, aliases=("FRA",)),
            GeoZone(gtype=GeoZoneType.EIC, key="DE_LU", name="Germany Luxembourg Bidding Zone", country_iso="DE", region_iso="DE_LU", lat=51.0, lon=10.0, aliases=("DELU",)),
        ]
    )

    zone = catalog.find_by_str("germany lux bidding")
    assert zone is not None
    assert zone.key == "DE LU"


def test_load_geozones_can_include_entsoe_bidding_zones(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "yggdrasil.data.enums.geozone.load.fetch_entsoe_bidding_zones",
        lambda: (
            GeoZone(
                gtype=GeoZoneType.EIC,
                key="SE1",
                name="Sweden Bidding Zone 1",
                country_iso="SE",
                region_iso="SE1",
                eic="10Y1001A1001A44Q",
                lat=62.0,
                lon=15.0,
                aliases=("SE1",),
            ),
        ),
    )

    catalog = load_geozones(include_entsoe_bidding_zones=True)

    zone = catalog.find_by_str("10Y1001A1001A44Q")
    assert zone is not None
    assert zone.region_iso == "SE1"
