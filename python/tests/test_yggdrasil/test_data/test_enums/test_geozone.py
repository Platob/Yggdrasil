"""
Comprehensive unit tests for yggdrasil.data.enums.geozone.GeoZone.

Coverage areas
--------------
* GeoZoneType constants
* WKB helpers (_point_wkb / _parse_point_wkb round-trip)
* GeoZone constructor – validation, field normalisation, frozen semantics
* GeoZone.from_coordinates – happy path, boundary values, invalid input
* GeoZone.parse_coordinates – strings, sequences, dicts, objects, edge cases
* Cache operations – put, get_by_*, clear_cache, alias resolution
* GeoZone.parse_str – key, name, EIC, compact key (dash/space), coordinates
* GeoZone.parse – None, self, bytes, str, dict, arbitrary objects
* polars_parse_str – return_value wkb / country_iso / city_iso / point / struct, Expr, Series, lazy, invalid type/value
* polars_parse_bin – return_value wkb / country_iso / city_iso / point / struct, Expr, Series, lazy, invalid type/value
* py_parse_str – scalar pure-Python mirror of polars_parse_str (all return_value modes)
* py_parse_bin – scalar pure-Python mirror of polars_parse_bin (all return_value modes)
"""

from __future__ import annotations

import struct
from typing import Any

import polars as pl
import pytest

from yggdrasil.data.enums.geozone import GeoZone, GeoZoneType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_point_wkb(lat: float, lon: float) -> bytes:
    """Build a minimal little-endian WKB POINT (type=1)."""
    return struct.pack("<BIdd", 1, 1, float(lon), float(lat))


def _zone(
    *,
    gtype: int = GeoZoneType.COUNTRY,
    lat: float = 0.0,
    lon: float = 0.0,
    key: str = "TEST",
    name: str = "Test Zone",
    **kwargs: Any,
) -> GeoZone:
    """Convenience factory that wraps GeoZone.from_coordinates."""
    return GeoZone.from_coordinates(
        gtype=gtype,
        lat=lat,
        lon=lon,
        key=key,
        name=name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_geozone_cache() -> None:
    """Wipe the global GeoZone cache before (and after) every test."""
    GeoZone.clear_cache()
    yield
    GeoZone.clear_cache()


# ---------------------------------------------------------------------------
# GeoZoneType
# ---------------------------------------------------------------------------

class TestGeoZoneType:
    def test_constants_are_distinct_ints(self) -> None:
        values = [
            GeoZoneType.UNKNOWN,
            GeoZoneType.WORLD,
            GeoZoneType.CONTINENT,
            GeoZoneType.COUNTRY,
            GeoZoneType.CITY,
            GeoZoneType.ZONE,
        ]
        assert len(set(values)) == 6
        assert all(isinstance(v, int) for v in values)

    def test_ordering(self) -> None:
        assert GeoZoneType.UNKNOWN < GeoZoneType.WORLD < GeoZoneType.CONTINENT < GeoZoneType.COUNTRY
        assert GeoZoneType.COUNTRY < GeoZoneType.CITY < GeoZoneType.ZONE

    def test_unknown_is_negative(self) -> None:
        assert GeoZoneType.UNKNOWN < 0


# ---------------------------------------------------------------------------
# WKB round-trip
# ---------------------------------------------------------------------------

class TestWkbRoundTrip:
    @pytest.mark.parametrize("lat,lon", [
        (0.0, 0.0),
        (46.8182, 8.2275),
        (-33.8688, 151.2093),
        (90.0, 180.0),
        (-90.0, -180.0),
    ])
    def test_point_wkb_roundtrip(self, lat: float, lon: float) -> None:
        wkb = _make_point_wkb(lat, lon)
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=wkb, srid=4326, key="X")
        assert zone.lat == pytest.approx(lat)
        assert zone.lon == pytest.approx(lon)
        assert zone.point == pytest.approx((lat, lon))

    def test_big_endian_wkb_roundtrip(self) -> None:
        lat, lon = 51.5074, -0.1278
        # big-endian WKB POINT
        wkb = struct.pack(">BIdd", 0, 1, lon, lat)
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=wkb, srid=4326, key="X")
        assert zone.lat == pytest.approx(lat)
        assert zone.lon == pytest.approx(lon)

    def test_invalid_wkb_geometry_type_raises(self) -> None:
        wkb = struct.pack("<BIdd", 1, 2, 0.0, 0.0)  # type 2 = LineString
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=wkb, srid=4326, key="X")
        with pytest.raises(ValueError, match="geometry type"):
            _ = zone.point

    def test_invalid_wkb_byte_order_raises(self) -> None:
        wkb = bytes([2]) + b"\x00" * 20  # byte order = 2 (invalid)
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=wkb, srid=4326, key="X")
        with pytest.raises(ValueError, match="byte order"):
            _ = zone.point

    def test_wkb_wrong_length_raises(self) -> None:
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=b"\x01" * 5, srid=4326, key="X")
        with pytest.raises(ValueError, match="WKB length"):
            _ = zone.point


# ---------------------------------------------------------------------------
# Constructor validation & normalisation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_rejects_empty_wkb(self) -> None:
        with pytest.raises(ValueError, match="wkb must not be empty"):
            GeoZone(gtype=GeoZoneType.COUNTRY, wkb=b"")

    def test_rejects_non_bytes_wkb(self) -> None:
        with pytest.raises(TypeError, match="wkb must be bytes"):
            GeoZone(gtype=GeoZoneType.COUNTRY, wkb="not bytes")  # type: ignore[arg-type]

    def test_rejects_negative_srid(self) -> None:
        with pytest.raises(ValueError, match="srid must be >= 0"):
            GeoZone(gtype=GeoZoneType.COUNTRY, wkb=_make_point_wkb(1, 2), srid=-1)

    def test_accepts_zero_srid(self) -> None:
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=_make_point_wkb(1, 2), srid=0)
        assert zone.srid == 0

    def test_accepts_bytearray_wkb(self) -> None:
        zone = GeoZone(
            gtype=GeoZoneType.COUNTRY,
            wkb=bytearray(_make_point_wkb(1, 2)),
            srid=4326,
            key="X",
        )
        assert isinstance(zone.wkb, bytes)

    def test_accepts_memoryview_wkb(self) -> None:
        zone = GeoZone(
            gtype=GeoZoneType.COUNTRY,
            wkb=memoryview(_make_point_wkb(1, 2)),
            srid=4326,
            key="X",
        )
        assert isinstance(zone.wkb, bytes)

    def test_is_frozen(self) -> None:
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=_make_point_wkb(1, 2), srid=4326, key="X")
        with pytest.raises((AttributeError, TypeError)):
            zone.key = "Y"  # type: ignore[misc]


class TestConstructorNormalisation:
    def test_string_fields_stripped_and_upcased(self) -> None:
        zone = GeoZone(
            gtype=GeoZoneType.CITY,
            wkb=_make_point_wkb(1.0, 2.0),
            srid=4326,
            country_iso=" ch ",
            country_name=" Switzerland ",
            city_iso=" zrh ",
            city_name=" Zurich ",
            key=" zrh ",
            name=" Zurich ",
            eic=" 10ych-swissgridz ",
            tz=" Europe/Zurich ",
            ccy=" chf ",
        )
        assert zone.country_iso == "CH"
        assert zone.country_name == "Switzerland"
        assert zone.city_iso == "ZRH"
        assert zone.city_name == "Zurich"
        assert zone.key == "ZRH"
        assert zone.name == "Zurich"
        assert zone.eic == "10YCH-SWISSGRIDZ"
        assert zone.tz == "Europe/Zurich"
        assert zone.ccy == "CHF"

    def test_whitespace_only_fields_become_none(self) -> None:
        zone = GeoZone(
            gtype=GeoZoneType.COUNTRY,
            wkb=_make_point_wkb(1, 2),
            srid=4326,
            country_iso="   ",
            name="   ",
            eic="   ",
        )
        assert zone.country_iso is None
        assert zone.name is None
        assert zone.eic is None

    def test_aliases_normalised(self) -> None:
        zone = GeoZone(
            gtype=GeoZoneType.COUNTRY,
            wkb=_make_point_wkb(1, 2),
            srid=4326,
            key="DE",
            aliases=(" germany ", " deu "),
        )
        assert "GERMANY" in zone.aliases
        assert "DEU" in zone.aliases

    def test_none_fields_stay_none(self) -> None:
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=_make_point_wkb(1, 2))
        assert zone.key is None
        assert zone.name is None
        assert zone.eic is None
        assert zone.tz is None
        assert zone.ccy is None

    def test_geom_key_property(self) -> None:
        wkb = _make_point_wkb(10.0, 20.0)
        zone = GeoZone(gtype=GeoZoneType.COUNTRY, wkb=wkb, srid=4326)
        assert zone.geom_key == (4326, wkb)


# ---------------------------------------------------------------------------
# from_coordinates
# ---------------------------------------------------------------------------

class TestFromCoordinates:
    def test_basic_country(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.8182,
            lon=8.2275,
            key="CH",
            name="Switzerland",
            country_iso="CH",
            country_name="Switzerland",
            tz="Europe/Zurich",
            ccy="CHF",
        )
        assert zone.gtype == GeoZoneType.COUNTRY
        assert zone.srid == 4326
        assert zone.lat == pytest.approx(46.8182)
        assert zone.lon == pytest.approx(8.2275)
        assert zone.key == "CH"
        assert zone.ccy == "CHF"

    def test_basic_city(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769,
            lon=8.5417,
            key="ZRH",
            name="Zurich",
            country_iso="CH",
            country_name="Switzerland",
            city_iso="ZRH",
            city_name="Zurich",
            tz="Europe/Zurich",
            ccy="CHF",
        )
        assert zone.gtype == GeoZoneType.CITY
        assert zone.city_iso == "ZRH"

    @pytest.mark.parametrize("lat,lon", [
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 180.0),
        (0.0, -180.0),
    ])
    def test_boundary_coordinates_accepted(self, lat: float, lon: float) -> None:
        zone = GeoZone.from_coordinates(gtype=GeoZoneType.ZONE, lat=lat, lon=lon, key="B")
        assert zone.lat == pytest.approx(lat)
        assert zone.lon == pytest.approx(lon)

    @pytest.mark.parametrize("lat", [90.001, -90.001, 100.0, -100.0])
    def test_invalid_latitude_raises(self, lat: float) -> None:
        with pytest.raises(ValueError, match="latitude"):
            GeoZone.from_coordinates(gtype=GeoZoneType.CITY, lat=lat, lon=0.0, key="X")

    @pytest.mark.parametrize("lon", [180.001, -180.001, 200.0, -200.0])
    def test_invalid_longitude_raises(self, lon: float) -> None:
        with pytest.raises(ValueError, match="longitude"):
            GeoZone.from_coordinates(gtype=GeoZoneType.CITY, lat=0.0, lon=lon, key="X")

    def test_custom_srid(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE, lat=0.0, lon=0.0, key="S", srid=3857
        )
        assert zone.srid == 3857


# ---------------------------------------------------------------------------
# parse_coordinates
# ---------------------------------------------------------------------------

class TestParseCoordinates:
    @pytest.mark.parametrize("s,expected", [
        ("46.8182, 8.2275", (46.8182, 8.2275)),
        ("46.8182 8.2275", (46.8182, 8.2275)),
        ("46.8182;8.2275", (46.8182, 8.2275)),
        ("46.8182|8.2275", (46.8182, 8.2275)),
        ("-33.8688,151.2093", (-33.8688, 151.2093)),
        (" 1.0 , -2.5 ", (1.0, -2.5)),
    ])
    def test_valid_string(self, s: str, expected: tuple) -> None:
        assert GeoZone.parse_coordinates(s) == pytest.approx(expected)

    @pytest.mark.parametrize("seq,expected", [
        ((46.8182, 8.2275), (46.8182, 8.2275)),
        ([46.8182, 8.2275], (46.8182, 8.2275)),
        ((0, 0), (0.0, 0.0)),
    ])
    def test_valid_sequence(self, seq: Any, expected: tuple) -> None:
        assert GeoZone.parse_coordinates(seq) == pytest.approx(expected)

    @pytest.mark.parametrize("d,expected", [
        ({"lat": 46.8182, "lon": 8.2275}, (46.8182, 8.2275)),
        ({"latitude": 46.8182, "longitude": 8.2275}, (46.8182, 8.2275)),
    ])
    def test_valid_dict(self, d: dict, expected: tuple) -> None:
        assert GeoZone.parse_coordinates(d) == pytest.approx(expected)

    def test_none_returns_none(self) -> None:
        assert GeoZone.parse_coordinates(None) is None

    def test_non_coordinate_string_returns_none(self) -> None:
        assert GeoZone.parse_coordinates("Switzerland") is None

    def test_empty_string_returns_none(self) -> None:
        assert GeoZone.parse_coordinates("") is None

    def test_sequence_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="length 2"):
            GeoZone.parse_coordinates((1.0, 2.0, 3.0))

    def test_sequence_single_element_raises(self) -> None:
        with pytest.raises(ValueError, match="length 2"):
            GeoZone.parse_coordinates([1.0])

    def test_dict_missing_keys_returns_none(self) -> None:
        assert GeoZone.parse_coordinates({"x": 1, "y": 2}) is None

    def test_object_with_lat_lon_attrs(self) -> None:
        class Pt:
            lat = 10.0
            lon = 20.0

        assert GeoZone.parse_coordinates(Pt()) == pytest.approx((10.0, 20.0))

    def test_object_without_lat_lon_returns_none(self) -> None:
        class NoAttrs:
            pass

        assert GeoZone.parse_coordinates(NoAttrs()) is None


# ---------------------------------------------------------------------------
# Cache: put / get_by_* / clear_cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_put_and_get_by_key(self) -> None:
        zone = _zone(key="FR", name="France")
        GeoZone.put(zone)
        assert GeoZone.get_by_key("FR") is zone
        assert GeoZone.get_by_key("fr") is zone  # case-insensitive

    def test_put_and_get_by_name(self) -> None:
        zone = _zone(key="DE", name="Germany")
        GeoZone.put(zone)
        assert GeoZone.get_by_name("Germany") is zone
        assert GeoZone.get_by_name("germany") is zone
        assert GeoZone.get_by_name("GERMANY") is zone

    def test_put_and_get_by_eic(self) -> None:
        zone = _zone(key="FR", name="France", eic="10YFR-RTE------C")
        GeoZone.put(zone)
        assert GeoZone.get_by_eic("10YFR-RTE------C") is zone
        assert GeoZone.get_by_eic("10yfr-rte------c") is zone

    def test_put_and_get_by_geom(self) -> None:
        zone = _zone(lat=46.2276, lon=2.2137, key="FR", name="France")
        GeoZone.put(zone)
        assert GeoZone.get_by_geom(zone.wkb, srid=4326) is zone

    def test_put_and_get_by_coordinates(self) -> None:
        zone = _zone(lat=47.3769, lon=8.5417, key="ZRH", name="Zurich")
        GeoZone.put(zone)
        assert GeoZone.get_by_coordinates(47.3769, 8.5417, srid=4326) is zone

    def test_country_iso_registered_as_key(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276,
            lon=2.2137,
            key="FR",
            name="France",
            country_iso="FR",
            country_name="France",
        )
        GeoZone.put(zone)
        assert GeoZone.get_by_key("FR") is zone

    def test_city_iso_registered_as_key(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769,
            lon=8.5417,
            key="ZRH",
            name="Zurich",
            city_iso="ZRH",
            city_name="Zurich",
        )
        GeoZone.put(zone)
        assert GeoZone.get_by_key("ZRH") is zone

    def test_aliases_registered(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=50.9,
            lon=8.7,
            key="DE_LU",
            name="Germany-Luxembourg",
            aliases=("GER_LUX", "DELU"),
        )
        GeoZone.put(zone)
        assert GeoZone.get_by_key("GER_LUX") is zone
        assert GeoZone.get_by_key("DELU") is zone

    def test_clear_cache_removes_all_entries(self) -> None:
        zone = _zone(lat=20.0, lon=30.0, key="CLR", name="Clear Me", eic="10YCLR-TEST-----")
        GeoZone.put(zone)

        GeoZone.clear_cache()

        assert GeoZone.get_by_key("CLR") is None
        assert GeoZone.get_by_name("Clear Me") is None
        assert GeoZone.get_by_eic("10YCLR-TEST-----") is None
        assert GeoZone.get_by_geom(zone.wkb, 4326) is None

    def test_put_returns_same_zone(self) -> None:
        zone = _zone(key="RET", name="Return")
        result = GeoZone.put(zone)
        assert result is zone

    def test_get_by_key_missing_returns_none(self) -> None:
        assert GeoZone.get_by_key("MISSING") is None

    def test_get_by_name_missing_returns_none(self) -> None:
        assert GeoZone.get_by_name("No Such Zone") is None

    def test_get_by_eic_missing_returns_none(self) -> None:
        assert GeoZone.get_by_eic("10YNONE---------") is None

    def test_get_by_geom_missing_returns_none(self) -> None:
        assert GeoZone.get_by_geom(_make_point_wkb(99.0, 99.0)) is None

    def test_overwrite_key_updates_cache(self) -> None:
        zone1 = _zone(lat=1.0, lon=2.0, key="OVR", name="First")
        zone2 = _zone(lat=3.0, lon=4.0, key="OVR", name="Second")
        GeoZone.put(zone1)
        GeoZone.put(zone2)
        assert GeoZone.get_by_key("OVR") is zone2


# ---------------------------------------------------------------------------
# parse_str
# ---------------------------------------------------------------------------

class TestParseStr:
    def test_by_exact_key(self) -> None:
        zone = _zone(key="CH", name="Switzerland")
        GeoZone.put(zone)
        assert GeoZone.parse_str("CH") is zone

    def test_by_key_case_insensitive(self) -> None:
        zone = _zone(key="CH", name="Switzerland")
        GeoZone.put(zone)
        assert GeoZone.parse_str("ch") is zone

    def test_by_name(self) -> None:
        zone = _zone(key="CH_TEST", name="Switzerland Test")
        GeoZone.put(zone)
        assert GeoZone.parse_str("Switzerland Test") is zone

    def test_by_name_case_insensitive(self) -> None:
        zone = _zone(key="CH_TEST2", name="Switzerland Two")
        GeoZone.put(zone)
        assert GeoZone.parse_str("SWITZERLAND TWO") is zone

    def test_by_eic(self) -> None:
        zone = _zone(key="FR2", name="France 2", eic="10YFR2-TEST-----")
        GeoZone.put(zone)
        assert GeoZone.parse_str("10YFR2-TEST-----") is zone

    def test_compact_key_dash_separator(self) -> None:
        zone = _zone(key="DE_LU", name="Germany-Luxembourg", gtype=GeoZoneType.ZONE)
        GeoZone.put(zone)
        assert GeoZone.parse_str("DE-LU") is zone

    def test_compact_key_space_separator(self) -> None:
        zone = _zone(key="DE_LU", name="Germany-Luxembourg", gtype=GeoZoneType.ZONE)
        GeoZone.put(zone)
        assert GeoZone.parse_str("DE LU") is zone

    def test_by_coordinates(self) -> None:
        zone = _zone(lat=58.1467, lon=7.9956, key="NO2_COORD", name="Norway NO2 Coord")
        GeoZone.put(zone)
        assert GeoZone.parse_str("58.1467,7.9956") is zone

    def test_empty_string_returns_none(self) -> None:
        assert GeoZone.parse_str("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert GeoZone.parse_str("   ") is None

    def test_unknown_string_returns_none(self) -> None:
        assert GeoZone.parse_str("definitely_unknown_zone_xyz_123") is None


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

class TestParse:
    def test_none_returns_none(self) -> None:
        assert GeoZone.parse(None) is None

    def test_geozone_instance_returns_self(self) -> None:
        zone = _zone(key="SELF", name="Self")
        assert GeoZone.parse(zone) is zone

    def test_bytes_wkb_lookup(self) -> None:
        zone = _zone(lat=10.0, lon=20.0, key="WKB", name="WKB Zone")
        GeoZone.put(zone)
        assert GeoZone.parse(zone.wkb) is zone

    def test_bytearray_wkb_lookup(self) -> None:
        zone = _zone(lat=10.0, lon=20.0, key="WKB2", name="WKB Zone 2")
        GeoZone.put(zone)
        assert GeoZone.parse(bytearray(zone.wkb)) is zone

    def test_string_key(self) -> None:
        zone = _zone(key="STR", name="String")
        GeoZone.put(zone)
        assert GeoZone.parse("STR") is zone

    def test_dict_key(self) -> None:
        zone = _zone(lat=11.0, lon=21.0, key="DICT_K", name="Dict Key")
        GeoZone.put(zone)
        assert GeoZone.parse({"key": "DICT_K"}) is zone

    def test_dict_eic(self) -> None:
        zone = _zone(lat=12.0, lon=22.0, key="DICT_E", name="Dict EIC", eic="10YTEST-EIC-----")
        GeoZone.put(zone)
        assert GeoZone.parse({"eic": "10YTEST-EIC-----"}) is zone

    def test_dict_name(self) -> None:
        zone = _zone(lat=13.0, lon=23.0, key="DICT_N", name="Dict Name Zone")
        GeoZone.put(zone)
        assert GeoZone.parse({"name": "Dict Name Zone"}) is zone

    def test_dict_country_iso(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=14.0,
            lon=24.0,
            key="DICT_ISO",
            name="Dict Iso",
            country_iso="DI",
            country_name="Dict Iso",
        )
        GeoZone.put(zone)
        assert GeoZone.parse({"country_iso": "DI"}) is zone

    def test_dict_city_iso(self) -> None:
        zone = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=15.0,
            lon=25.0,
            key="DICT_CI",
            name="Dict City",
            city_iso="DCI",
            city_name="Dict City",
        )
        GeoZone.put(zone)
        assert GeoZone.parse({"city_iso": "DCI"}) is zone

    def test_dict_wkb_srid(self) -> None:
        zone = _zone(lat=16.0, lon=26.0, key="DICT_WKB", name="Dict WKB")
        GeoZone.put(zone)
        assert GeoZone.parse({"wkb": zone.wkb, "srid": 4326}) is zone

    def test_object_with_key_attr(self) -> None:
        zone = _zone(lat=17.0, lon=27.0, key="OBJ_K", name="Obj Key")
        GeoZone.put(zone)

        class Obj:
            key = "OBJ_K"

        assert GeoZone.parse(Obj()) is zone

    def test_object_with_eic_attr(self) -> None:
        zone = _zone(lat=18.0, lon=28.0, key="OBJ_E", name="Obj EIC", eic="10YOBJ-EIC------")
        GeoZone.put(zone)

        class Obj:
            eic = "10YOBJ-EIC------"

        assert GeoZone.parse(Obj()) is zone

    def test_object_with_name_attr(self) -> None:
        zone = _zone(lat=19.0, lon=29.0, key="OBJ_N", name="Obj Name Zone")
        GeoZone.put(zone)

        class Obj:
            name = "Obj Name Zone"

        assert GeoZone.parse(Obj()) is zone

    def test_object_with_wkb_attr(self) -> None:
        zone = _zone(lat=20.0, lon=30.0, key="OBJ_WKB", name="Obj WKB")
        GeoZone.put(zone)

        class Obj:
            wkb = zone.wkb
            srid = 4326

        assert GeoZone.parse(Obj()) is zone

    def test_dict_unknown_key_returns_none(self) -> None:
        assert GeoZone.parse({"key": "DEFINITELY_UNKNOWN_ZONE"}) is None

    def test_arbitrary_unknown_object_returns_none(self) -> None:
        class NoAttrs:
            pass

        assert GeoZone.parse(NoAttrs()) is None

    def test_int_returns_none(self) -> None:
        # int does not match any supported type path
        assert GeoZone.parse(42) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# polars_parse_str
# ---------------------------------------------------------------------------

class TestPolarsParseStr:
    """Tests for GeoZone.polars_parse_str covering all four return_value modes."""

    @pytest.fixture(autouse=True)
    def _register_zones(self) -> None:
        fr = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276,
            lon=2.2137,
            key="FR",
            name="France",
            country_iso="FR",
            country_name="France",
            eic="10YFR-RTE------C",
        )
        zrh = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769,
            lon=8.5417,
            key="ZRH",
            name="Zurich",
            country_iso="CH",
            country_name="Switzerland",
            city_iso="ZRH",
            city_name="Zurich",
        )
        de_lu = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=50.9,
            lon=8.7,
            key="DE_LU",
            name="Germany-Luxembourg",
            country_name="Germany-Luxembourg",
        )
        no2 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=58.1467,
            lon=7.9956,
            key="NO2",
            name="Norway NO2",
            country_iso="NO",
            country_name="Norway",
            eic="10YNO-2--------T",
        )
        for z in (fr, zrh, de_lu, no2):
            GeoZone.put(z)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        self._fr = fr
        self._zrh = zrh
        self._de_lu = de_lu
        self._no2 = no2

    # ------------------------------------------------------------------
    # return_value="wkb"  (default)
    # ------------------------------------------------------------------

    def test_wkb_expr_known_zones(self) -> None:
        df = pl.DataFrame(
            {
                "zone": [
                    "FR", "France", "fr",
                    "DE-LU", "de lu", "Germany Luxembourg",
                    "10YFR-RTE------C",
                    "unknown",
                ]
            }
        )
        out = df.select(GeoZone.polars_parse_str(pl.col("zone")).alias("wkb"))
        values = out["wkb"].to_list()

        assert values[0] == self._fr.wkb
        assert values[1] == self._fr.wkb
        assert values[2] == self._fr.wkb
        assert values[3] == self._de_lu.wkb
        assert values[4] == self._de_lu.wkb
        assert values[5] == self._de_lu.wkb
        assert values[6] == self._fr.wkb
        assert values[7] is None

    def test_wkb_series_returns_wkb(self) -> None:
        s = pl.Series("zone", ["NO2", "norway no2", "10YNO-2--------T", "bad"])
        out = GeoZone.polars_parse_str(s)

        assert isinstance(out, pl.Series)
        assert out.name == "zone"
        assert out.to_list() == [self._no2.wkb, self._no2.wkb, self._no2.wkb, None]

    def test_wkb_dtype_is_binary(self) -> None:
        s = pl.Series("zone", ["FR", "unknown"])
        assert GeoZone.polars_parse_str(s).dtype == pl.Binary

    def test_wkb_expr_dtype_is_binary(self) -> None:
        df = pl.DataFrame({"zone": ["FR", "unknown"]})
        out = df.select(GeoZone.polars_parse_str(pl.col("zone")).alias("wkb"))
        assert out["wkb"].dtype == pl.Binary

    def test_wkb_series_unknown_all_none(self) -> None:
        s = pl.Series("zone", ["TOTALLY_UNKNOWN_1", "TOTALLY_UNKNOWN_2"])
        assert GeoZone.polars_parse_str(s).to_list() == [None, None]

    def test_wkb_series_empty(self) -> None:
        s = pl.Series("zone", [], dtype=pl.Utf8)
        out = GeoZone.polars_parse_str(s)
        assert isinstance(out, pl.Series)
        assert len(out) == 0

    def test_wkb_eic_lookup(self) -> None:
        df = pl.DataFrame({"zone": ["10YNO-2--------T"]})
        out = df.select(GeoZone.polars_parse_str(pl.col("zone")).alias("wkb"))
        assert out["wkb"][0] == self._no2.wkb

    def test_wkb_case_insensitive(self) -> None:
        s = pl.Series("zone", ["france", "FRANCE", "FrAnCe"])
        out = GeoZone.polars_parse_str(s)
        assert all(v == self._fr.wkb for v in out.to_list())

    # ------------------------------------------------------------------
    # Substring / free-text matching
    # e.g. "Sweden SE1 wind power" should resolve to the SE1 zone
    # ------------------------------------------------------------------

    def test_wkb_substring_key_in_free_text(self) -> None:
        """A zone key embedded anywhere in a longer string is resolved."""
        se1 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=65.0,
            lon=17.0,
            key="SE1",
            name="Sweden SE1",
            country_iso="SE",
            country_name="Sweden",
        )
        GeoZone.put(se1)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        s = pl.Series("zone", [
            "Sweden SE1 wind power",
            "SE1 production",
            "wind power SE1",
            "nordic SE1 area",
        ])
        out = GeoZone.polars_parse_str(s)
        assert all(v == se1.wkb for v in out.to_list()), out.to_list()

    def test_wkb_substring_longer_alias_wins(self) -> None:
        """When two zones could match, the longer alias (more specific) wins."""
        se1 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=65.0, lon=17.0,
            key="SE1", name="Sweden SE1",
            country_iso="SE", country_name="Sweden",
        )
        se = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=60.1, lon=18.6,
            key="SE", name="Sweden",
            country_iso="SE", country_name="Sweden",
        )
        GeoZone.put(se)
        GeoZone.put(se1)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        s = pl.Series("zone", ["Sweden SE1 area"])
        out = GeoZone.polars_parse_str(s)
        # SE1 (longer) must beat SE
        assert out[0] == se1.wkb

    def test_country_iso_substring_free_text(self) -> None:
        """country_iso is returned for a zone found via free-text substring."""
        se1 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=65.0, lon=17.0,
            key="SE1", name="Sweden SE1",
            country_iso="SE", country_name="Sweden",
        )
        GeoZone.put(se1)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        s = pl.Series("zone", ["Sweden SE1 wind power"])
        out = GeoZone.polars_parse_str(s, return_value="country_iso")
        assert out[0] == "SE"

    def test_point_substring_free_text(self) -> None:
        """point is returned for a zone found via free-text substring."""
        se1 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=65.0, lon=17.0,
            key="SE1", name="Sweden SE1",
            country_iso="SE", country_name="Sweden",
        )
        GeoZone.put(se1)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        s = pl.Series("zone", ["Sweden SE1 wind power"])
        out = GeoZone.polars_parse_str(s, return_value="point")
        assert out[0]["lat"] == pytest.approx(65.0, abs=1e-4)
        assert out[0]["lon"] == pytest.approx(17.0, abs=1e-4)

    def test_wkb_no_false_positive_on_unrelated_text(self) -> None:
        """A string with no embedded zone key returns None."""
        s = pl.Series("zone", ["some completely unrelated text here"])
        out = GeoZone.polars_parse_str(s)
        assert out[0] is None

    # ------------------------------------------------------------------
    # return_value="country_iso"
    # ------------------------------------------------------------------

    def test_country_iso_expr(self) -> None:
        df = pl.DataFrame({"zone": ["FR", "france", "NO2", "DE-LU", "unknown"]})
        out = df.select(
            GeoZone.polars_parse_str(pl.col("zone"), return_value="country_iso").alias("iso")
        )
        values = out["iso"].to_list()

        assert values[0] == "FR"   # country_iso of FR
        assert values[1] == "FR"   # name lookup → same zone
        assert values[2] == "NO"   # NO2 has country_iso=NO
        assert values[3] is None   # DE_LU has no country_iso
        assert values[4] is None   # unknown

    def test_country_iso_series(self) -> None:
        s = pl.Series("zone", ["FR", "NO2", "unknown"])
        out = GeoZone.polars_parse_str(s, return_value="country_iso")

        assert isinstance(out, pl.Series)
        assert out.name == "zone"
        assert out.to_list() == ["FR", "NO", None]

    def test_country_iso_dtype_is_utf8(self) -> None:
        s = pl.Series("zone", ["FR"])
        out = GeoZone.polars_parse_str(s, return_value="country_iso")
        assert out.dtype == pl.Utf8

    def test_country_iso_unknown_is_none(self) -> None:
        s = pl.Series("zone", ["DEFINITELY_UNKNOWN"])
        out = GeoZone.polars_parse_str(s, return_value="country_iso")
        assert out[0] is None

    # ------------------------------------------------------------------
    # return_value="city_iso"
    # ------------------------------------------------------------------

    def test_city_iso_expr(self) -> None:
        df = pl.DataFrame({"zone": ["ZRH", "Zurich", "FR", "unknown"]})
        out = df.select(
            GeoZone.polars_parse_str(pl.col("zone"), return_value="city_iso").alias("iso")
        )
        values = out["iso"].to_list()

        assert values[0] == "ZRH"  # city_iso of ZRH zone
        assert values[1] == "ZRH"  # name lookup → same zone
        assert values[2] is None   # FR has no city_iso
        assert values[3] is None   # unknown

    def test_city_iso_series(self) -> None:
        s = pl.Series("zone", ["ZRH", "FR", "unknown"])
        out = GeoZone.polars_parse_str(s, return_value="city_iso")

        assert isinstance(out, pl.Series)
        assert out.name == "zone"
        assert out.to_list() == ["ZRH", None, None]

    def test_city_iso_dtype_is_utf8(self) -> None:
        s = pl.Series("zone", ["ZRH"])
        out = GeoZone.polars_parse_str(s, return_value="city_iso")
        assert out.dtype == pl.Utf8

    def test_city_iso_unknown_is_none(self) -> None:
        s = pl.Series("zone", ["DEFINITELY_UNKNOWN"])
        out = GeoZone.polars_parse_str(s, return_value="city_iso")
        assert out[0] is None

    # ------------------------------------------------------------------
    # return_value="point"
    # ------------------------------------------------------------------

    def test_point_expr(self) -> None:
        df = pl.DataFrame({"zone": ["FR", "NO2", "unknown"]})
        out = df.select(
            GeoZone.polars_parse_str(pl.col("zone"), return_value="point").alias("pt")
        )
        rows = out["pt"].to_list()

        assert rows[0]["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert rows[0]["lon"] == pytest.approx(self._fr.lon, abs=1e-4)
        assert rows[1]["lat"] == pytest.approx(self._no2.lat, abs=1e-4)
        assert rows[1]["lon"] == pytest.approx(self._no2.lon, abs=1e-4)
        assert rows[2]["lat"] is None
        assert rows[2]["lon"] is None

    def test_point_series(self) -> None:
        s = pl.Series("zone", ["FR", "ZRH", "unknown"])
        out = GeoZone.polars_parse_str(s, return_value="point")

        assert isinstance(out, pl.Series)
        assert out.name == "zone"
        rows = out.to_list()
        assert rows[0]["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert rows[0]["lon"] == pytest.approx(self._fr.lon, abs=1e-4)
        assert rows[1]["lat"] == pytest.approx(self._zrh.lat, abs=1e-4)
        assert rows[1]["lon"] == pytest.approx(self._zrh.lon, abs=1e-4)
        assert rows[2]["lat"] is None
        assert rows[2]["lon"] is None

    def test_point_dtype_is_struct(self) -> None:
        s = pl.Series("zone", ["FR"])
        out = GeoZone.polars_parse_str(s, return_value="point")
        assert out.dtype == pl.Struct({"lat": pl.Float64, "lon": pl.Float64})

    def test_point_unknown_lat_lon_none(self) -> None:
        s = pl.Series("zone", ["DEFINITELY_UNKNOWN"])
        out = GeoZone.polars_parse_str(s, return_value="point")
        row = out[0]
        assert row["lat"] is None
        assert row["lon"] is None

    # ------------------------------------------------------------------
    # return_value="struct"
    # ------------------------------------------------------------------

    def test_struct_series_fields(self) -> None:
        s = pl.Series("zone", ["FR", "ZRH", "unknown"])
        out = GeoZone.polars_parse_str(s, return_value="struct")

        assert isinstance(out, pl.Series)
        fr_row = out[0]
        assert fr_row["key"] == "FR"
        assert fr_row["name"] == "France"
        assert fr_row["country_iso"] == "FR"
        assert fr_row["country_name"] == "France"
        assert fr_row["city_iso"] is None
        assert fr_row["city_name"] is None
        assert fr_row["eic"] == "10YFR-RTE------C"
        assert fr_row["gtype"] == GeoZoneType.COUNTRY
        assert fr_row["srid"] == 4326
        assert fr_row["wkb"] == self._fr.wkb
        assert fr_row["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert fr_row["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

        zrh_row = out[1]
        assert zrh_row["key"] == "ZRH"
        assert zrh_row["city_iso"] == "ZRH"
        assert zrh_row["country_iso"] == "CH"
        assert zrh_row["gtype"] == GeoZoneType.CITY

        none_row = out[2]
        assert none_row["key"] is None
        assert none_row["wkb"] is None
        assert none_row["lat"] is None

    def test_struct_expr(self) -> None:
        df = pl.DataFrame({"zone": ["FR"]})
        out = df.select(GeoZone.polars_parse_str(pl.col("zone"), return_value="struct").alias("s"))
        row = out["s"][0]
        assert row["key"] == "FR"
        assert row["country_iso"] == "FR"

    def test_struct_dtype_has_all_fields(self) -> None:
        s = pl.Series("zone", ["FR"])
        out = GeoZone.polars_parse_str(s, return_value="struct")
        expected_fields = {
            "gtype", "wkb", "srid", "key", "name",
            "country_iso", "country_name", "city_iso", "city_name",
            "eic", "tz", "ccy", "lat", "lon",
        }
        actual_fields = {f.name for f in out.dtype.fields}  # type: ignore[attr-defined]
        assert actual_fields == expected_fields

    def test_struct_no2_zone_fields(self) -> None:
        s = pl.Series("zone", ["NO2"])
        out = GeoZone.polars_parse_str(s, return_value="struct")
        row = out[0]
        assert row["key"] == "NO2"
        assert row["country_iso"] == "NO"
        assert row["gtype"] == GeoZoneType.ZONE
        assert row["eic"] == "10YNO-2--------T"
        assert row["lat"] == pytest.approx(self._no2.lat, abs=1e-4)

    def test_struct_free_text_resolves_fields(self) -> None:
        """Free-text substring match via parse_str also populates all struct fields."""
        s = pl.Series("zone", ["Norway NO2 wind power"])
        out = GeoZone.polars_parse_str(s, return_value="struct")
        row = out[0]
        assert row["key"] == "NO2"
        assert row["country_iso"] == "NO"

    # ------------------------------------------------------------------
    # Invalid inputs (shared across all modes)
    # ------------------------------------------------------------------

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="expected pl.Series | pl.Expr"):
            GeoZone.polars_parse_str(123)  # type: ignore[arg-type]

    def test_invalid_type_list_raises(self) -> None:
        with pytest.raises(TypeError, match="expected pl.Series | pl.Expr"):
            GeoZone.polars_parse_str(["FR", "DE"])  # type: ignore[arg-type]

    def test_invalid_return_value_raises(self) -> None:
        with pytest.raises(ValueError, match="return_value must be one of"):
            df = pl.DataFrame({"zone": ["FR"]})
            df.select(
                GeoZone.polars_parse_str(pl.col("zone"), return_value="bad_mode").alias("x")  # type: ignore[arg-type]
            )

    # ------------------------------------------------------------------
    # lazy=True — Series input returns Expr
    # ------------------------------------------------------------------

    def test_lazy_series_returns_expr(self) -> None:
        s = pl.Series("zone", ["FR", "unknown"])
        result = GeoZone.polars_parse_str(s, lazy=True)
        assert isinstance(result, pl.Expr)

    def test_lazy_expr_still_returns_expr(self) -> None:
        result = GeoZone.polars_parse_str(pl.col("zone"), lazy=True)
        assert isinstance(result, pl.Expr)

    def test_lazy_false_series_returns_series(self) -> None:
        s = pl.Series("zone", ["FR", "unknown"])
        result = GeoZone.polars_parse_str(s, lazy=False)
        assert isinstance(result, pl.Series)

    def test_lazy_expr_is_usable_in_select(self) -> None:
        s = pl.Series("zone", ["FR", "NO2"])
        expr = GeoZone.polars_parse_str(s, lazy=True)
        out = pl.DataFrame({"zone": ["FR", "NO2"]}).select(expr.alias("wkb"))
        assert out["wkb"][0] == self._fr.wkb
        assert out["wkb"][1] == self._no2.wkb

    def test_lazy_country_iso_expr_is_usable(self) -> None:
        s = pl.Series("zone", ["FR", "NO2"])
        expr = GeoZone.polars_parse_str(s, return_value="country_iso", lazy=True)
        out = pl.DataFrame({"zone": ["FR", "NO2"]}).select(expr.alias("iso"))
        assert out["iso"][0] == "FR"
        assert out["iso"][1] == "NO"


# ---------------------------------------------------------------------------
# polars_parse_bin
# ---------------------------------------------------------------------------

class TestPolarsParseBin:
    """Tests for GeoZone.polars_parse_bin covering all four return_value modes."""

    @pytest.fixture(autouse=True)
    def _register_zones(self) -> None:
        fr = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276, lon=2.2137,
            key="FR", name="France",
            country_iso="FR", country_name="France",
            eic="10YFR-RTE------C",
        )
        zrh = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769, lon=8.5417,
            key="ZRH", name="Zurich",
            country_iso="CH", country_name="Switzerland",
            city_iso="ZRH", city_name="Zurich",
        )
        no2 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=58.1467, lon=7.9956,
            key="NO2", name="Norway NO2",
            country_iso="NO", country_name="Norway",
            eic="10YNO-2--------T",
        )
        for z in (fr, zrh, no2):
            GeoZone.put(z)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()
        GeoZone._build_bin_lookup_cache.cache_clear()

        self._fr = fr
        self._zrh = zrh
        self._no2 = no2

    # ---------------------------------------------------------------
    # return_value="wkb"  (default) — pass-through canonical WKB
    # ---------------------------------------------------------------

    def test_wkb_series_known(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb, self._no2.wkb, b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s)
        assert isinstance(out, pl.Series)
        assert out[0] == self._fr.wkb
        assert out[1] == self._no2.wkb
        assert out[2] is None  # unknown WKB → None

    def test_wkb_expr_known(self) -> None:
        df = pl.DataFrame({"wkb": [self._fr.wkb, self._no2.wkb, b"\x00" * 21]})
        out = df.select(GeoZone.polars_parse_bin(pl.col("wkb")).alias("out"))
        assert out["out"][0] == self._fr.wkb
        assert out["out"][1] == self._no2.wkb
        assert out["out"][2] is None

    def test_wkb_dtype_is_binary(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        assert GeoZone.polars_parse_bin(s).dtype == pl.Binary

    def test_wkb_series_empty(self) -> None:
        s = pl.Series("wkb", [], dtype=pl.Binary)
        out = GeoZone.polars_parse_bin(s)
        assert isinstance(out, pl.Series)
        assert len(out) == 0

    def test_wkb_all_unknown_returns_none(self) -> None:
        s = pl.Series("wkb", [b"\x00" * 21, b"\xff" * 21])
        out = GeoZone.polars_parse_bin(s)
        assert out.to_list() == [None, None]

    # ---------------------------------------------------------------
    # return_value="country_iso"
    # ---------------------------------------------------------------

    def test_country_iso_series(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb, self._no2.wkb, self._zrh.wkb, b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s, return_value="country_iso")
        assert out.to_list() == ["FR", "NO", "CH", None]

    def test_country_iso_expr(self) -> None:
        df = pl.DataFrame({"wkb": [self._fr.wkb, self._no2.wkb]})
        out = df.select(GeoZone.polars_parse_bin(pl.col("wkb"), return_value="country_iso").alias("iso"))
        assert out["iso"].to_list() == ["FR", "NO"]

    def test_country_iso_dtype_is_utf8(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        assert GeoZone.polars_parse_bin(s, return_value="country_iso").dtype == pl.Utf8

    def test_country_iso_no_country_iso_returns_none(self) -> None:
        # ZRH has country_iso=CH — just sanity check zone-without-country-iso gives None
        no_country = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE, lat=1.0, lon=1.0, key="NOCOUNTRY", name="No Country",
        )
        GeoZone.put(no_country)
        GeoZone._build_bin_lookup_cache.cache_clear()
        s = pl.Series("wkb", [no_country.wkb])
        out = GeoZone.polars_parse_bin(s, return_value="country_iso")
        assert out[0] is None

    # ---------------------------------------------------------------
    # return_value="city_iso"
    # ---------------------------------------------------------------

    def test_city_iso_series(self) -> None:
        s = pl.Series("wkb", [self._zrh.wkb, self._fr.wkb, b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s, return_value="city_iso")
        assert out.to_list() == ["ZRH", None, None]

    def test_city_iso_dtype_is_utf8(self) -> None:
        s = pl.Series("wkb", [self._zrh.wkb])
        assert GeoZone.polars_parse_bin(s, return_value="city_iso").dtype == pl.Utf8

    # ---------------------------------------------------------------
    # return_value="point"
    # ---------------------------------------------------------------

    def test_point_series(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb, self._no2.wkb, b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s, return_value="point")
        rows = out.to_list()
        assert rows[0]["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert rows[0]["lon"] == pytest.approx(self._fr.lon, abs=1e-4)
        assert rows[1]["lat"] == pytest.approx(self._no2.lat, abs=1e-4)
        assert rows[1]["lon"] == pytest.approx(self._no2.lon, abs=1e-4)
        assert rows[2]["lat"] is None
        assert rows[2]["lon"] is None

    def test_point_expr(self) -> None:
        df = pl.DataFrame({"wkb": [self._zrh.wkb]})
        out = df.select(GeoZone.polars_parse_bin(pl.col("wkb"), return_value="point").alias("pt"))
        row = out["pt"][0]
        assert row["lat"] == pytest.approx(self._zrh.lat, abs=1e-4)
        assert row["lon"] == pytest.approx(self._zrh.lon, abs=1e-4)

    def test_point_dtype_is_struct(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        out = GeoZone.polars_parse_bin(s, return_value="point")
        assert out.dtype == pl.Struct({"lat": pl.Float64, "lon": pl.Float64})

    def test_point_unknown_returns_none_fields(self) -> None:
        s = pl.Series("wkb", [b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s, return_value="point")
        row = out[0]
        assert row["lat"] is None
        assert row["lon"] is None

    # ---------------------------------------------------------------
    # lazy=True
    # ---------------------------------------------------------------

    def test_lazy_series_returns_expr(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        result = GeoZone.polars_parse_bin(s, lazy=True)
        assert isinstance(result, pl.Expr)

    def test_lazy_expr_still_returns_expr(self) -> None:
        result = GeoZone.polars_parse_bin(pl.col("wkb"), lazy=True)
        assert isinstance(result, pl.Expr)

    def test_lazy_false_series_returns_series(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        result = GeoZone.polars_parse_bin(s, lazy=False)
        assert isinstance(result, pl.Series)

    def test_lazy_expr_usable_in_select(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb, self._no2.wkb])
        expr = GeoZone.polars_parse_bin(s, return_value="country_iso", lazy=True)
        out = pl.DataFrame({"wkb": [self._fr.wkb, self._no2.wkb]}).select(expr.alias("iso"))
        assert out["iso"].to_list() == ["FR", "NO"]

    # ---------------------------------------------------------------
    # return_value="struct"
    # ---------------------------------------------------------------

    def test_struct_series_fields(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb, self._zrh.wkb, b"\x00" * 21])
        out = GeoZone.polars_parse_bin(s, return_value="struct")

        assert isinstance(out, pl.Series)
        fr_row = out[0]
        assert fr_row["key"] == "FR"
        assert fr_row["name"] == "France"
        assert fr_row["country_iso"] == "FR"
        assert fr_row["country_name"] == "France"
        assert fr_row["city_iso"] is None
        assert fr_row["eic"] == "10YFR-RTE------C"
        assert fr_row["gtype"] == GeoZoneType.COUNTRY
        assert fr_row["srid"] == 4326
        assert fr_row["wkb"] == self._fr.wkb
        assert fr_row["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert fr_row["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

        zrh_row = out[1]
        assert zrh_row["key"] == "ZRH"
        assert zrh_row["city_iso"] == "ZRH"
        assert zrh_row["gtype"] == GeoZoneType.CITY

        none_row = out[2]
        assert none_row["key"] is None
        assert none_row["wkb"] is None
        assert none_row["lat"] is None

    def test_struct_expr(self) -> None:
        df = pl.DataFrame({"wkb": [self._no2.wkb]})
        out = df.select(GeoZone.polars_parse_bin(pl.col("wkb"), return_value="struct").alias("s"))
        row = out["s"][0]
        assert row["key"] == "NO2"
        assert row["country_iso"] == "NO"
        assert row["eic"] == "10YNO-2--------T"

    def test_struct_dtype_has_all_fields(self) -> None:
        s = pl.Series("wkb", [self._fr.wkb])
        out = GeoZone.polars_parse_bin(s, return_value="struct")
        expected_fields = {
            "gtype", "wkb", "srid", "key", "name",
            "country_iso", "country_name", "city_iso", "city_name",
            "eic", "tz", "ccy", "lat", "lon",
        }
        actual_fields = {f.name for f in out.dtype.fields}  # type: ignore[attr-defined]
        assert actual_fields == expected_fields

    def test_struct_roundtrip_str_to_bin_to_struct(self) -> None:
        """Full round-trip: string → WKB → struct."""
        s_str = pl.Series("zone", ["France", "NO2"])
        wkb_series = GeoZone.polars_parse_str(s_str)
        struct_series = GeoZone.polars_parse_bin(wkb_series, return_value="struct")
        assert struct_series[0]["key"] == "FR"
        assert struct_series[1]["key"] == "NO2"
        assert struct_series[0]["country_iso"] == "FR"
        assert struct_series[1]["country_iso"] == "NO"

    # ---------------------------------------------------------------
    # Round-trip: polars_parse_str → polars_parse_bin
    # ---------------------------------------------------------------

    def test_roundtrip_str_to_bin_to_country_iso(self) -> None:
        """str → wkb via parse_str, then wkb → country_iso via parse_bin."""
        s_str = pl.Series("zone", ["France", "NO2"])
        wkb_series = GeoZone.polars_parse_str(s_str)
        iso_series = GeoZone.polars_parse_bin(wkb_series, return_value="country_iso")
        assert iso_series.to_list() == ["FR", "NO"]

    def test_roundtrip_str_to_bin_to_point(self) -> None:
        s_str = pl.Series("zone", ["FR", "NO2"])
        wkb_series = GeoZone.polars_parse_str(s_str)
        point_series = GeoZone.polars_parse_bin(wkb_series, return_value="point")
        rows = point_series.to_list()
        assert rows[0]["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert rows[1]["lat"] == pytest.approx(self._no2.lat, abs=1e-4)

    # ---------------------------------------------------------------
    # Invalid inputs
    # ---------------------------------------------------------------

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="expected pl.Series | pl.Expr"):
            GeoZone.polars_parse_bin(123)  # type: ignore[arg-type]

    def test_invalid_type_list_raises(self) -> None:
        with pytest.raises(TypeError, match="expected pl.Series | pl.Expr"):
            GeoZone.polars_parse_bin([b"\x01"])  # type: ignore[arg-type]

    def test_invalid_return_value_raises(self) -> None:
        with pytest.raises(ValueError, match="return_value must be one of"):
            df = pl.DataFrame({"wkb": [self._fr.wkb]})
            df.select(
                GeoZone.polars_parse_bin(pl.col("wkb"), return_value="bad_mode").alias("x")  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# py_parse_str  (pure-Python scalar mirror of polars_parse_str)
# ---------------------------------------------------------------------------

class TestPyParseStr:
    """Tests for GeoZone.py_parse_str covering all return_value modes."""

    @pytest.fixture(autouse=True)
    def _register_zones(self) -> None:
        fr = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276, lon=2.2137,
            key="FR", name="France",
            country_iso="FR", country_name="France",
            eic="10YFR-RTE------C",
        )
        zrh = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769, lon=8.5417,
            key="ZRH", name="Zurich",
            country_iso="CH", country_name="Switzerland",
            city_iso="ZRH", city_name="Zurich",
        )
        de_lu = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=50.9, lon=8.7,
            key="DE_LU", name="Germany-Luxembourg",
            country_name="Germany-Luxembourg",
        )
        no2 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=58.1467, lon=7.9956,
            key="NO2", name="Norway NO2",
            country_iso="NO", country_name="Norway",
            eic="10YNO-2--------T",
        )
        for z in (fr, zrh, de_lu, no2):
            GeoZone.put(z)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        self._fr = fr
        self._zrh = zrh
        self._de_lu = de_lu
        self._no2 = no2

    # ------------------------------------------------------------------
    # None input
    # ------------------------------------------------------------------

    def test_none_returns_none_wkb(self) -> None:
        assert GeoZone.py_parse_str(None) is None

    def test_none_returns_point_nulls(self) -> None:
        result = GeoZone.py_parse_str(None, return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_none_returns_struct_nulls(self) -> None:
        result = GeoZone.py_parse_str(None, return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    # ------------------------------------------------------------------
    # return_value="wkb"  (default)
    # ------------------------------------------------------------------

    def test_wkb_exact_key(self) -> None:
        assert GeoZone.py_parse_str("FR") == self._fr.wkb

    def test_wkb_case_insensitive(self) -> None:
        assert GeoZone.py_parse_str("fr") == self._fr.wkb
        assert GeoZone.py_parse_str("France") == self._fr.wkb
        assert GeoZone.py_parse_str("france") == self._fr.wkb

    def test_wkb_by_eic(self) -> None:
        assert GeoZone.py_parse_str("10YFR-RTE------C") == self._fr.wkb

    def test_wkb_compact_dash(self) -> None:
        assert GeoZone.py_parse_str("DE-LU") == self._de_lu.wkb

    def test_wkb_compact_space(self) -> None:
        assert GeoZone.py_parse_str("DE LU") == self._de_lu.wkb

    def test_wkb_free_text(self) -> None:
        """Free-text string with embedded zone key resolves via token scan."""
        assert GeoZone.py_parse_str("Norway NO2 wind power") == self._no2.wkb

    def test_wkb_unknown_returns_none(self) -> None:
        assert GeoZone.py_parse_str("TOTALLY_UNKNOWN_XYZ") is None

    def test_wkb_empty_returns_none(self) -> None:
        assert GeoZone.py_parse_str("") is None

    def test_wkb_whitespace_returns_none(self) -> None:
        assert GeoZone.py_parse_str("   ") is None

    def test_wkb_is_bytes(self) -> None:
        result = GeoZone.py_parse_str("FR")
        assert isinstance(result, bytes)

    # ------------------------------------------------------------------
    # return_value="country_iso"
    # ------------------------------------------------------------------

    def test_country_iso_known(self) -> None:
        assert GeoZone.py_parse_str("FR", return_value="country_iso") == "FR"
        assert GeoZone.py_parse_str("NO2", return_value="country_iso") == "NO"

    def test_country_iso_none_when_missing(self) -> None:
        # DE_LU has no country_iso set
        assert GeoZone.py_parse_str("DE-LU", return_value="country_iso") is None

    def test_country_iso_unknown(self) -> None:
        assert GeoZone.py_parse_str("UNKNOWN_XYZ", return_value="country_iso") is None

    def test_country_iso_none_input(self) -> None:
        assert GeoZone.py_parse_str(None, return_value="country_iso") is None

    def test_country_iso_free_text(self) -> None:
        assert GeoZone.py_parse_str("Norway NO2 wind power", return_value="country_iso") == "NO"

    # ------------------------------------------------------------------
    # return_value="city_iso"
    # ------------------------------------------------------------------

    def test_city_iso_known(self) -> None:
        assert GeoZone.py_parse_str("ZRH", return_value="city_iso") == "ZRH"
        assert GeoZone.py_parse_str("Zurich", return_value="city_iso") == "ZRH"

    def test_city_iso_none_for_country(self) -> None:
        assert GeoZone.py_parse_str("FR", return_value="city_iso") is None

    def test_city_iso_unknown(self) -> None:
        assert GeoZone.py_parse_str("UNKNOWN_XYZ", return_value="city_iso") is None

    def test_city_iso_none_input(self) -> None:
        assert GeoZone.py_parse_str(None, return_value="city_iso") is None

    # ------------------------------------------------------------------
    # return_value="point"
    # ------------------------------------------------------------------

    def test_point_known(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="point")
        assert result["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

    def test_point_unknown(self) -> None:
        result = GeoZone.py_parse_str("UNKNOWN_XYZ", return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_point_none_input(self) -> None:
        result = GeoZone.py_parse_str(None, return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_point_free_text(self) -> None:
        result = GeoZone.py_parse_str("Norway NO2 wind power", return_value="point")
        assert result["lat"] == pytest.approx(self._no2.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._no2.lon, abs=1e-4)

    def test_point_has_lat_lon_keys(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="point")
        assert set(result.keys()) == {"lat", "lon"}

    # ------------------------------------------------------------------
    # return_value="struct"
    # ------------------------------------------------------------------

    def test_struct_known_fr(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="struct")
        assert result["key"] == "FR"
        assert result["name"] == "France"
        assert result["country_iso"] == "FR"
        assert result["country_name"] == "France"
        assert result["city_iso"] is None
        assert result["eic"] == "10YFR-RTE------C"
        assert result["gtype"] == GeoZoneType.COUNTRY
        assert result["srid"] == 4326
        assert result["wkb"] == self._fr.wkb
        assert result["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

    def test_struct_known_zrh(self) -> None:
        result = GeoZone.py_parse_str("ZRH", return_value="struct")
        assert result["key"] == "ZRH"
        assert result["city_iso"] == "ZRH"
        assert result["country_iso"] == "CH"
        assert result["gtype"] == GeoZoneType.CITY

    def test_struct_unknown_all_none(self) -> None:
        result = GeoZone.py_parse_str("UNKNOWN_XYZ", return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    def test_struct_none_input_all_none(self) -> None:
        result = GeoZone.py_parse_str(None, return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    def test_struct_has_all_geozone_fields(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="struct")
        expected_fields = {
            "gtype", "wkb", "srid", "key", "name",
            "country_iso", "country_name", "city_iso", "city_name",
            "eic", "tz", "ccy", "lat", "lon",
        }
        assert set(result.keys()) == expected_fields

    def test_struct_field_order_matches_geozone(self) -> None:
        """Field order in the returned dict matches GeoZone dataclass order."""
        result = GeoZone.py_parse_str("FR", return_value="struct")
        expected_order = [
            "gtype", "wkb", "srid",
            "country_iso", "country_name",
            "city_iso", "city_name",
            "key", "name", "eic",
            "tz", "ccy", "lat", "lon",
        ]
        assert list(result.keys()) == expected_order

    def test_struct_free_text(self) -> None:
        result = GeoZone.py_parse_str("Norway NO2 wind power", return_value="struct")
        assert result["key"] == "NO2"
        assert result["country_iso"] == "NO"

    # ------------------------------------------------------------------
    # Longer-alias-wins (specificity)
    # ------------------------------------------------------------------

    def test_longer_alias_wins_over_shorter(self) -> None:
        se1 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE, lat=65.0, lon=17.0,
            key="SE1", name="Sweden SE1",
            country_iso="SE", country_name="Sweden",
        )
        se = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY, lat=60.1, lon=18.6,
            key="SE", name="Sweden",
            country_iso="SE", country_name="Sweden",
        )
        GeoZone.put(se)
        GeoZone.put(se1)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()

        # "Sweden SE1 area" must resolve to SE1 (longer token wins)
        assert GeoZone.py_parse_str("Sweden SE1 area") == se1.wkb

    # ------------------------------------------------------------------
    # Consistency with polars_parse_str
    # ------------------------------------------------------------------

    def test_consistent_with_polars_parse_str_wkb(self) -> None:
        for zone_str in ["FR", "france", "10YFR-RTE------C", "DE-LU", "NO2", "UNKNOWN"]:
            py_result = GeoZone.py_parse_str(zone_str)
            pl_result = GeoZone.polars_parse_str(pl.Series("z", [zone_str]))[0]
            assert py_result == pl_result, f"mismatch for {zone_str!r}"

    def test_consistent_with_polars_parse_str_country_iso(self) -> None:
        for zone_str in ["FR", "NO2", "ZRH", "UNKNOWN"]:
            py_result = GeoZone.py_parse_str(zone_str, return_value="country_iso")
            pl_result = GeoZone.polars_parse_str(
                pl.Series("z", [zone_str]), return_value="country_iso"
            )[0]
            assert py_result == pl_result, f"mismatch for {zone_str!r}"

    def test_consistent_with_polars_parse_str_city_iso(self) -> None:
        for zone_str in ["ZRH", "FR", "UNKNOWN"]:
            py_result = GeoZone.py_parse_str(zone_str, return_value="city_iso")
            pl_result = GeoZone.polars_parse_str(
                pl.Series("z", [zone_str]), return_value="city_iso"
            )[0]
            assert py_result == pl_result, f"mismatch for {zone_str!r}"

    def test_consistent_with_polars_parse_str_point(self) -> None:
        for zone_str in ["FR", "NO2", "UNKNOWN"]:
            py_result = GeoZone.py_parse_str(zone_str, return_value="point")
            pl_row = GeoZone.polars_parse_str(
                pl.Series("z", [zone_str]), return_value="point"
            )[0]
            assert py_result["lat"] == pl_row["lat"], f"lat mismatch for {zone_str!r}"
            assert py_result["lon"] == pl_row["lon"], f"lon mismatch for {zone_str!r}"


# ---------------------------------------------------------------------------
# py_parse_bin  (pure-Python scalar mirror of polars_parse_bin)
# ---------------------------------------------------------------------------

class TestPyParseBin:
    """Tests for GeoZone.py_parse_bin covering all return_value modes."""

    @pytest.fixture(autouse=True)
    def _register_zones(self) -> None:
        fr = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276, lon=2.2137,
            key="FR", name="France",
            country_iso="FR", country_name="France",
            eic="10YFR-RTE------C",
        )
        zrh = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769, lon=8.5417,
            key="ZRH", name="Zurich",
            country_iso="CH", country_name="Switzerland",
            city_iso="ZRH", city_name="Zurich",
        )
        no2 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=58.1467, lon=7.9956,
            key="NO2", name="Norway NO2",
            country_iso="NO", country_name="Norway",
            eic="10YNO-2--------T",
        )
        for z in (fr, zrh, no2):
            GeoZone.put(z)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()
        GeoZone._build_bin_lookup_cache.cache_clear()

        self._fr = fr
        self._zrh = zrh
        self._no2 = no2

    _UNKNOWN_WKB = b"\x00" * 21

    # ------------------------------------------------------------------
    # None input
    # ------------------------------------------------------------------

    def test_none_returns_none_wkb(self) -> None:
        assert GeoZone.py_parse_bin(None) is None

    def test_none_returns_point_nulls(self) -> None:
        result = GeoZone.py_parse_bin(None, return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_none_returns_struct_nulls(self) -> None:
        result = GeoZone.py_parse_bin(None, return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    # ------------------------------------------------------------------
    # return_value="wkb"  (default)
    # ------------------------------------------------------------------

    def test_wkb_known(self) -> None:
        assert GeoZone.py_parse_bin(self._fr.wkb) == self._fr.wkb
        assert GeoZone.py_parse_bin(self._no2.wkb) == self._no2.wkb

    def test_wkb_accepts_bytearray(self) -> None:
        assert GeoZone.py_parse_bin(bytearray(self._fr.wkb)) == self._fr.wkb

    def test_wkb_accepts_memoryview(self) -> None:
        assert GeoZone.py_parse_bin(memoryview(self._fr.wkb)) == self._fr.wkb

    def test_wkb_unknown_returns_none(self) -> None:
        assert GeoZone.py_parse_bin(self._UNKNOWN_WKB) is None

    def test_wkb_result_is_bytes(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb)
        assert isinstance(result, bytes)

    # ------------------------------------------------------------------
    # return_value="country_iso"
    # ------------------------------------------------------------------

    def test_country_iso_known(self) -> None:
        assert GeoZone.py_parse_bin(self._fr.wkb, return_value="country_iso") == "FR"
        assert GeoZone.py_parse_bin(self._no2.wkb, return_value="country_iso") == "NO"
        assert GeoZone.py_parse_bin(self._zrh.wkb, return_value="country_iso") == "CH"

    def test_country_iso_unknown(self) -> None:
        assert GeoZone.py_parse_bin(self._UNKNOWN_WKB, return_value="country_iso") is None

    def test_country_iso_none_input(self) -> None:
        assert GeoZone.py_parse_bin(None, return_value="country_iso") is None

    def test_country_iso_missing_field(self) -> None:
        no_country = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE, lat=1.0, lon=1.0,
            key="NOCOUNTRY", name="No Country",
        )
        GeoZone.put(no_country)
        GeoZone._build_bin_lookup_cache.cache_clear()
        assert GeoZone.py_parse_bin(no_country.wkb, return_value="country_iso") is None

    # ------------------------------------------------------------------
    # return_value="city_iso"
    # ------------------------------------------------------------------

    def test_city_iso_known(self) -> None:
        assert GeoZone.py_parse_bin(self._zrh.wkb, return_value="city_iso") == "ZRH"

    def test_city_iso_none_for_country(self) -> None:
        assert GeoZone.py_parse_bin(self._fr.wkb, return_value="city_iso") is None

    def test_city_iso_unknown(self) -> None:
        assert GeoZone.py_parse_bin(self._UNKNOWN_WKB, return_value="city_iso") is None

    def test_city_iso_none_input(self) -> None:
        assert GeoZone.py_parse_bin(None, return_value="city_iso") is None

    # ------------------------------------------------------------------
    # return_value="point"
    # ------------------------------------------------------------------

    def test_point_known(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="point")
        assert result["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

    def test_point_unknown(self) -> None:
        result = GeoZone.py_parse_bin(self._UNKNOWN_WKB, return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_point_none_input(self) -> None:
        result = GeoZone.py_parse_bin(None, return_value="point")
        assert result == {"lat": None, "lon": None}

    def test_point_has_lat_lon_keys(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="point")
        assert set(result.keys()) == {"lat", "lon"}

    def test_point_no2(self) -> None:
        result = GeoZone.py_parse_bin(self._no2.wkb, return_value="point")
        assert result["lat"] == pytest.approx(self._no2.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._no2.lon, abs=1e-4)

    # ------------------------------------------------------------------
    # return_value="struct"
    # ------------------------------------------------------------------

    def test_struct_known_fr(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="struct")
        assert result["key"] == "FR"
        assert result["name"] == "France"
        assert result["country_iso"] == "FR"
        assert result["country_name"] == "France"
        assert result["city_iso"] is None
        assert result["eic"] == "10YFR-RTE------C"
        assert result["gtype"] == GeoZoneType.COUNTRY
        assert result["srid"] == 4326
        assert result["wkb"] == self._fr.wkb
        assert result["lat"] == pytest.approx(self._fr.lat, abs=1e-4)
        assert result["lon"] == pytest.approx(self._fr.lon, abs=1e-4)

    def test_struct_known_zrh(self) -> None:
        result = GeoZone.py_parse_bin(self._zrh.wkb, return_value="struct")
        assert result["key"] == "ZRH"
        assert result["city_iso"] == "ZRH"
        assert result["country_iso"] == "CH"
        assert result["gtype"] == GeoZoneType.CITY

    def test_struct_unknown_all_none(self) -> None:
        result = GeoZone.py_parse_bin(self._UNKNOWN_WKB, return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    def test_struct_none_input_all_none(self) -> None:
        result = GeoZone.py_parse_bin(None, return_value="struct")
        assert result["key"] is None
        assert result["wkb"] is None
        assert result["lat"] is None

    def test_struct_has_all_geozone_fields(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="struct")
        expected_fields = {
            "gtype", "wkb", "srid", "key", "name",
            "country_iso", "country_name", "city_iso", "city_name",
            "eic", "tz", "ccy", "lat", "lon",
        }
        assert set(result.keys()) == expected_fields

    def test_struct_field_order_matches_geozone(self) -> None:
        """Field order in the returned dict matches GeoZone dataclass order."""
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="struct")
        expected_order = [
            "gtype", "wkb", "srid",
            "country_iso", "country_name",
            "city_iso", "city_name",
            "key", "name", "eic",
            "tz", "ccy", "lat", "lon",
        ]
        assert list(result.keys()) == expected_order

    # ------------------------------------------------------------------
    # Invalid return_value
    # ------------------------------------------------------------------

    def test_invalid_return_value_raises(self) -> None:
        with pytest.raises(ValueError, match="return_value must be one of"):
            GeoZone.py_parse_bin(self._fr.wkb, return_value="bad_mode")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Round-trip: py_parse_str → py_parse_bin
    # ------------------------------------------------------------------

    def test_roundtrip_str_to_bin_wkb(self) -> None:
        wkb = GeoZone.py_parse_str("France")
        assert GeoZone.py_parse_bin(wkb) == self._fr.wkb

    def test_roundtrip_str_to_bin_country_iso(self) -> None:
        wkb = GeoZone.py_parse_str("NO2")
        assert GeoZone.py_parse_bin(wkb, return_value="country_iso") == "NO"

    def test_roundtrip_str_to_bin_struct(self) -> None:
        wkb = GeoZone.py_parse_str("France")
        result = GeoZone.py_parse_bin(wkb, return_value="struct")
        assert result["key"] == "FR"
        assert result["country_iso"] == "FR"

    # ------------------------------------------------------------------
    # Consistency with polars_parse_bin
    # ------------------------------------------------------------------

    def test_consistent_with_polars_parse_bin_wkb(self) -> None:
        for zone in (self._fr, self._zrh, self._no2):
            py_result = GeoZone.py_parse_bin(zone.wkb)
            pl_result = GeoZone.polars_parse_bin(pl.Series("w", [zone.wkb]))[0]
            assert py_result == pl_result

    def test_consistent_with_polars_parse_bin_country_iso(self) -> None:
        for zone in (self._fr, self._zrh, self._no2):
            py_result = GeoZone.py_parse_bin(zone.wkb, return_value="country_iso")
            pl_result = GeoZone.polars_parse_bin(
                pl.Series("w", [zone.wkb]), return_value="country_iso"
            )[0]
            assert py_result == pl_result

    def test_consistent_with_polars_parse_bin_point(self) -> None:
        for zone in (self._fr, self._no2):
            py_result = GeoZone.py_parse_bin(zone.wkb, return_value="point")
            pl_row = GeoZone.polars_parse_bin(
                pl.Series("w", [zone.wkb]), return_value="point"
            )[0]
            assert py_result["lat"] == pytest.approx(pl_row["lat"], abs=1e-9)
            assert py_result["lon"] == pytest.approx(pl_row["lon"], abs=1e-9)

    def test_consistent_with_polars_parse_bin_unknown(self) -> None:
        py_result = GeoZone.py_parse_bin(self._UNKNOWN_WKB)
        pl_result = GeoZone.polars_parse_bin(pl.Series("w", [self._UNKNOWN_WKB]))[0]
        assert py_result == pl_result

    # ------------------------------------------------------------------
    # return_value="dataclass"
    # ------------------------------------------------------------------

    def test_dataclass_known_returns_singleton(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="dataclass")
        assert result is self._fr

    def test_dataclass_zrh_singleton(self) -> None:
        result = GeoZone.py_parse_bin(self._zrh.wkb, return_value="dataclass")
        assert result is self._zrh

    def test_dataclass_no2_singleton(self) -> None:
        result = GeoZone.py_parse_bin(self._no2.wkb, return_value="dataclass")
        assert result is self._no2

    def test_dataclass_unknown_returns_none(self) -> None:
        result = GeoZone.py_parse_bin(self._UNKNOWN_WKB, return_value="dataclass")
        assert result is None

    def test_dataclass_none_input_returns_none(self) -> None:
        result = GeoZone.py_parse_bin(None, return_value="dataclass")
        assert result is None

    def test_dataclass_bytearray_input(self) -> None:
        result = GeoZone.py_parse_bin(bytearray(self._fr.wkb), return_value="dataclass")
        assert result is self._fr

    def test_dataclass_memoryview_input(self) -> None:
        result = GeoZone.py_parse_bin(memoryview(self._no2.wkb), return_value="dataclass")
        assert result is self._no2

    def test_dataclass_is_geozone_instance(self) -> None:
        result = GeoZone.py_parse_bin(self._fr.wkb, return_value="dataclass")
        assert isinstance(result, GeoZone)

    def test_dataclass_invalid_return_value_still_raises(self) -> None:
        with pytest.raises(ValueError, match="return_value must be one of"):
            GeoZone.py_parse_bin(self._fr.wkb, return_value="bad_mode")  # type: ignore[arg-type]

    def test_dataclass_roundtrip_str_to_bin(self) -> None:
        """str → wkb → GeoZone singleton round-trip."""
        wkb = GeoZone.py_parse_str("France")
        result = GeoZone.py_parse_bin(wkb, return_value="dataclass")
        assert result is self._fr


# ---------------------------------------------------------------------------
# py_parse_str  — return_value="dataclass"
# ---------------------------------------------------------------------------

class TestPyParseStrDataclass:
    """Tests for GeoZone.py_parse_str with return_value='dataclass'."""

    @pytest.fixture(autouse=True)
    def _register_zones(self) -> None:
        fr = GeoZone.from_coordinates(
            gtype=GeoZoneType.COUNTRY,
            lat=46.2276, lon=2.2137,
            key="FR", name="France",
            country_iso="FR", country_name="France",
            eic="10YFR-RTE------C",
        )
        zrh = GeoZone.from_coordinates(
            gtype=GeoZoneType.CITY,
            lat=47.3769, lon=8.5417,
            key="ZRH", name="Zurich",
            country_iso="CH", country_name="Switzerland",
            city_iso="ZRH", city_name="Zurich",
        )
        no2 = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=58.1467, lon=7.9956,
            key="NO2", name="Norway NO2",
            country_iso="NO", country_name="Norway",
            eic="10YNO-2--------T",
        )
        de_lu = GeoZone.from_coordinates(
            gtype=GeoZoneType.ZONE,
            lat=50.9, lon=8.7,
            key="DE_LU", name="Germany-Luxembourg",
            country_name="Germany-Luxembourg",
        )
        for z in (fr, zrh, no2, de_lu):
            GeoZone.put(z)
        GeoZone._build_bidding_zone_regex_cache.cache_clear()
        GeoZone._build_bin_lookup_cache.cache_clear()

        self._fr = fr
        self._zrh = zrh
        self._no2 = no2
        self._de_lu = de_lu

    def test_exact_key_returns_singleton(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="dataclass")
        assert result is self._fr

    def test_case_insensitive_returns_singleton(self) -> None:
        assert GeoZone.py_parse_str("fr", return_value="dataclass") is self._fr
        assert GeoZone.py_parse_str("france", return_value="dataclass") is self._fr
        assert GeoZone.py_parse_str("FRANCE", return_value="dataclass") is self._fr

    def test_by_name_returns_singleton(self) -> None:
        assert GeoZone.py_parse_str("Zurich", return_value="dataclass") is self._zrh

    def test_by_eic_returns_singleton(self) -> None:
        assert GeoZone.py_parse_str("10YFR-RTE------C", return_value="dataclass") is self._fr

    def test_compact_dash_returns_singleton(self) -> None:
        assert GeoZone.py_parse_str("DE-LU", return_value="dataclass") is self._de_lu

    def test_compact_space_returns_singleton(self) -> None:
        assert GeoZone.py_parse_str("DE LU", return_value="dataclass") is self._de_lu

    def test_free_text_returns_singleton(self) -> None:
        result = GeoZone.py_parse_str("Norway NO2 wind power", return_value="dataclass")
        assert result is self._no2

    def test_unknown_returns_none(self) -> None:
        assert GeoZone.py_parse_str("TOTALLY_UNKNOWN_XYZ", return_value="dataclass") is None

    def test_empty_returns_none(self) -> None:
        assert GeoZone.py_parse_str("", return_value="dataclass") is None

    def test_whitespace_returns_none(self) -> None:
        assert GeoZone.py_parse_str("   ", return_value="dataclass") is None

    def test_none_input_returns_none(self) -> None:
        assert GeoZone.py_parse_str(None, return_value="dataclass") is None

    def test_result_is_geozone_instance(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="dataclass")
        assert isinstance(result, GeoZone)

    def test_result_fields_match(self) -> None:
        result = GeoZone.py_parse_str("FR", return_value="dataclass")
        assert result.key == "FR"
        assert result.country_iso == "FR"
        assert result.eic == "10YFR-RTE------C"
        assert result.gtype == GeoZoneType.COUNTRY
        assert result.wkb == self._fr.wkb

    def test_consistent_with_py_parse_bin_dataclass(self) -> None:
        """py_parse_str dataclass → py_parse_bin dataclass produce the same singleton."""
        zone_via_str = GeoZone.py_parse_str("France", return_value="dataclass")
        wkb = GeoZone.py_parse_str("France")
        zone_via_bin = GeoZone.py_parse_bin(wkb, return_value="dataclass")
        assert zone_via_str is zone_via_bin

    def test_is_frozen_singleton(self) -> None:
        result = GeoZone.py_parse_str("ZRH", return_value="dataclass")
        with pytest.raises((AttributeError, TypeError)):
            result.key = "MUTATED"  # type: ignore[misc]

