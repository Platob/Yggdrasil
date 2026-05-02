"""``fetch_entsoe_bidding_zones`` — ENTSO-E EIC code ingestion.

Bidding zones come as a CSV from the ENTSO-E EIC code list. The
parser keeps only ``Bidding Zone`` rows (drops ``Control Area`` etc.),
joins each zone to a known country, and normalises EIC / display
strings. Live integration variant in
``test_geozone_entsoe_integration.py``.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import (
    GeoZone,
    GeoZoneCatalog,
    GeoZoneType,
    fetch_entsoe_bidding_zones,
    load_geozones,
)


_CSV_TEXT = """EicCode;EicDisplayName;EicLongName;EicParent;EicResponsibleParty;EicStatus;MarketParticipantPostalCode;MarketParticipantIsoCountryCode;MarketParticipantVatCode;EicTypeFunctionList;type
10Y1001A1001A44P;FR;France Bidding Zone;;10X1001A1001A39I;Active;;FR;;Bidding Zone;Y
10YDOM-CZ-DE-SKK;DE_LU;Germany-Luxembourg Bidding Zone;;10X1001A1001A83F;Active;;DE;;Bidding Zone;Y
11YNOT-BIDDING;FR_GRID;France Grid Area;;10X1001A1001A39I;Active;;FR;;Control Area;Y
"""


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


def _country_seed() -> dict[str, GeoZone]:
    return {
        "FR": GeoZone(
            gtype=GeoZoneType.COUNTRY,
            key="FR", name="France",
            country_iso="FR", region_iso="FR",
            lat=46.0, lon=2.0,
        ),
        "DE": GeoZone(
            gtype=GeoZoneType.COUNTRY,
            key="DE", name="Germany",
            country_iso="DE", region_iso="DE",
            lat=51.0, lon=10.0,
        ),
    }


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------


class TestFetchEntsoeBiddingZones:

    def test_parses_only_bidding_zones_and_drops_control_areas(self) -> None:
        session = _FakeSession(_CSV_TEXT)

        zones = fetch_entsoe_bidding_zones(
            session=session, countries=_country_seed()
        )

        # France grid area is dropped (Control Area row); only bidding zones remain.
        assert [zone.key for zone in zones] == ["DE LU", "FR"]
        assert all(zone.gtype == GeoZoneType.EIC for zone in zones)

    def test_eic_field_normalises_hyphens_to_spaces(self) -> None:
        session = _FakeSession(_CSV_TEXT)

        zones = fetch_entsoe_bidding_zones(
            session=session, countries=_country_seed()
        )

        assert zones[0].eic == "10YDOM CZ DE SKK"

    def test_country_iso_attaches_from_country_seed(self) -> None:
        session = _FakeSession(_CSV_TEXT)

        zones = fetch_entsoe_bidding_zones(
            session=session, countries=_country_seed()
        )

        assert zones[0].country_iso == "DE"

    def test_eic_code_kept_as_first_alias_for_lookup(self) -> None:
        session = _FakeSession(_CSV_TEXT)

        zones = fetch_entsoe_bidding_zones(
            session=session, countries=_country_seed()
        )

        # Ordered DE_LU first, FR second (alphabetical on key).
        assert zones[1].aliases[0] == "10Y1001A1001A44P"

    def test_request_targets_eic_code_csv_endpoint(self) -> None:
        session = _FakeSession(_CSV_TEXT)

        fetch_entsoe_bidding_zones(
            session=session, countries=_country_seed()
        )

        assert session.calls[0][0].endswith("Y_eiccodes.csv")


# ---------------------------------------------------------------------------
# Catalog.find_by_str — fuzzy search across name + aliases
# ---------------------------------------------------------------------------


class TestFindByStr:

    def test_resolves_to_best_partial_match(self) -> None:
        catalog = GeoZoneCatalog.from_values(
            [
                GeoZone(
                    gtype=GeoZoneType.COUNTRY,
                    key="FR", name="France",
                    country_iso="FR", region_iso="FR",
                    lat=46.0, lon=2.0,
                    aliases=("FRA",),
                ),
                GeoZone(
                    gtype=GeoZoneType.EIC,
                    key="DE_LU", name="Germany Luxembourg Bidding Zone",
                    country_iso="DE", region_iso="DE_LU",
                    lat=51.0, lon=10.0,
                    aliases=("DELU",),
                ),
            ]
        )

        zone = catalog.find_by_str("germany lux bidding")

        assert zone is not None
        assert zone.key == "DE LU"


# ---------------------------------------------------------------------------
# load_geozones with ENTSO-E enrichment
# ---------------------------------------------------------------------------


class TestLoadGeozonesWithEntsoe:

    def test_include_entsoe_bidding_zones_weaves_records_into_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "yggdrasil.data.enums.geozone.load.fetch_entsoe_bidding_zones",
            lambda: (
                GeoZone(
                    gtype=GeoZoneType.EIC,
                    key="SE1", name="Sweden Bidding Zone 1",
                    country_iso="SE", region_iso="SE1",
                    eic="10Y1001A1001A44Q",
                    lat=62.0, lon=15.0,
                    aliases=("SE1",),
                ),
            ),
        )

        catalog = load_geozones(include_entsoe_bidding_zones=True)

        zone = catalog.find_by_str("10Y1001A1001A44Q")
        assert zone is not None
        assert zone.region_iso == "SE1"
