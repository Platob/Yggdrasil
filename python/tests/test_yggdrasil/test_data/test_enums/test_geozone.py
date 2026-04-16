from __future__ import annotations

import pytest

from yggdrasil.data.enums.geozone import (
    GeoZone,
    GeoZoneCatalog,
    GeoZoneType,
    join_geozones,
    load_geozones,
)
from yggdrasil.polars.tests import PolarsTestCase


@pytest.fixture(scope="module")
def catalog() -> GeoZoneCatalog:
    return load_geozones()


def test_geozone_normalizes_and_validates() -> None:
    zone = GeoZone(
        gtype=GeoZoneType.REGION,
        key=" fr ",
        name=" Ile   de   France ",
        country_iso=" fr ",
        region_iso=" fr-idf ",
        sub_iso=" fr-idf ",
        ccy=" eur ",
        lat="48.8566",
        lon="2.3522",
        aliases=("fra", "paris-region"),
    )

    assert zone.key == "FR"
    assert zone.gtype == GeoZoneType.REGION
    assert zone.name == "Ile de France"
    assert zone.country_iso == "FR"
    assert zone.region_iso == "FR IDF"
    assert zone.sub_iso == "FR IDF"
    assert zone.ccy == "EUR"
    assert zone.aliases == ("FRA", "PARIS REGION")


@pytest.mark.parametrize(
    ("value", "expected_name"),
    [
        ("FR", "France"),
        ("fra", "France"),
        ("Europe", "Europe"),
        ("CH-ZH", "Zurich"),
        ("zuerich", "Zurich"),
    ],
)
def test_catalog_parse_str_resolves_codes_and_aliases(catalog: GeoZoneCatalog, value: str, expected_name: str) -> None:
    zone = catalog.parse_str(value)
    assert zone is not None
    assert zone.name == expected_name


@pytest.mark.parametrize(
    "value",
    ["47.3769, 8.5417", "47.3769 8.5417", "47.3769|8.5417", "47.3769;8.5417"],
)
def test_catalog_parse_str_resolves_known_coordinates(catalog: GeoZoneCatalog, value: str) -> None:
    zone = catalog.parse_str(value)
    assert zone is not None
    assert zone.name == "Zurich"


def test_catalog_parse_returns_ad_hoc_zone_for_unknown_coordinates(catalog: GeoZoneCatalog) -> None:
    zone = catalog.parse_str("10.5, 20.5")
    assert zone == GeoZone(gtype=GeoZoneType.CUSTOM, lat=10.5, lon=20.5)


def test_catalog_parse_accepts_mappings_and_tuples(catalog: GeoZoneCatalog) -> None:
    assert catalog.parse({"country_iso": "DE"}).name == "Germany"
    assert catalog.parse({"lat": 47.3769, "lon": 8.5417}).name == "Zurich"

    tuple_zone = catalog.parse((1.5, 2.5))
    assert tuple_zone == GeoZone(gtype=GeoZoneType.CUSTOM, lat=1.5, lon=2.5)


def test_load_geozones_accepts_custom_records() -> None:
    catalog = load_geozones(
        [
            {
                "key": "CA-QC",
                "name": "Quebec",
                "country_iso": "CA",
                "region_iso": "CA-QC",
                "lat": 52.9399,
                "lon": -73.5491,
                "aliases": ["QUEBEC"],
            }
        ]
    )

    zone = catalog.parse("quebec")
    assert zone is not None
    assert zone.region_iso == "CA QC"
    assert zone.country_iso == "CA"


class TestGeoZonePolars(PolarsTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.catalog = load_geozones()

    def test_join_geozones_enriches_polars_frames(self) -> None:
        frame = self.pl.DataFrame({"zone": ["fr", "CH-ZH", "unknown"]})
        enriched = join_geozones(frame, "zone", catalog=self.catalog)

        self.assertEqual(enriched["geozone_country_iso"].to_list(), ["FR", "CH", None])
        self.assertEqual(enriched["geozone_gtype"].to_list(), ["COUNTRY", "CITY", None])
        self.assertEqual(enriched["geozone_name"].to_list(), ["France", "Zurich", None])
        self.assertEqual(enriched["geozone_ccy"].to_list(), ["EUR", "CHF", None])

    def test_catalog_to_polars_supports_filters(self) -> None:
        frame = self.catalog.to_polars(country_iso="FR")

        self.assertIsInstance(frame, self.pl.DataFrame)
        self.assertEqual(frame["name"].to_list(), ["France"])
        self.assertEqual(frame["country_iso"].to_list(), ["FR"])

    def test_catalog_to_polars_supports_text_match(self) -> None:
        frame = self.catalog.to_polars(text="zuerich")

        self.assertEqual(frame["key"].to_list(), ["CH ZH"])
        self.assertEqual(frame["name"].to_list(), ["Zurich"])

    def test_catalog_to_polars_supports_sub_iso_and_currency_filters(self) -> None:
        frame = self.catalog.to_polars(sub_iso="CH-ZH", ccy="CHF")

        self.assertEqual(frame["name"].to_list(), ["Zurich"])
        self.assertEqual(frame["sub_iso"].to_list(), ["CH ZH"])
        self.assertEqual(frame["ccy"].to_list(), ["CHF"])
