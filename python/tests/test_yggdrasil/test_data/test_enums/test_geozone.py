import pytest

from yggdrasil.data.enums.geozone import GeoZone, GeoZoneType, load_geozones


@pytest.fixture(scope="module", autouse=True)
def geozones_loaded() -> None:
    GeoZone.clear_cache()
    load_geozones()


@pytest.fixture(autouse=True)
def clear_internal_lru_state_between_tests() -> None:
    # Keep the registry loaded, but make each test independent from cached resolver internals.
    GeoZone._resolver_cache.cache_clear()
    GeoZone._build_bidding_zone_regex_cache.cache_clear()
    GeoZone._build_bin_lookup_cache.cache_clear()


@pytest.mark.parametrize(
    ("value", "expected_attr", "expected_type"),
    [
        ("Europe", "EUROPE", GeoZoneType.CONTINENT),
        ("FR", "FRANCE", GeoZoneType.COUNTRY),
        ("France", "FRANCE", GeoZoneType.COUNTRY),
        ("BZN|FR", "FRANCE", GeoZoneType.COUNTRY),
        ("UK", "UNITED_KINGDOM", GeoZoneType.COUNTRY),
        ("UK photovoltaic power generation forecast Meteologica 30min", "UNITED_KINGDOM", GeoZoneType.COUNTRY),
        ("Zurich", "ZURICH", GeoZoneType.CITY),
        ("SE1", "SE1", GeoZoneType.ZONE),
        ("Swiss Alps", "SWISS_ALPS", GeoZoneType.ZONE),
    ],
)
def test_parse_str_resolves_common_inputs(value: str, expected_attr: str, expected_type: int) -> None:
    zone = GeoZone.parse_str(value)

    assert zone is getattr(GeoZone, expected_attr)
    assert zone.gtype == expected_type


@pytest.mark.parametrize(
    ("value", "expected_attr"),
    [
        ("united kingdom", "UNITED_KINGDOM"),
        ("UNITED-KINGDOM", "UNITED_KINGDOM"),
        ("bzn/fr", "FRANCE"),
        ("pjm dom", "PJM_DOM"),
        ("swiss-alps", "SWISS_ALPS"),
    ],
)
def test_parse_str_normalizes_separators_and_case(value: str, expected_attr: str) -> None:
    assert GeoZone.parse_str(value) is getattr(GeoZone, expected_attr)


@pytest.mark.parametrize(
    ("value", "expected_attr"),
    [
        ("Europe France", "EUROPE"),
        ("France Zurich", "FRANCE"),
        ("Zurich SE1", "ZURICH"),
    ],
)
def test_parse_str_prefers_broader_geotype_when_multiple_tokens_match(value: str, expected_attr: str) -> None:
    assert GeoZone.parse_str(value) is getattr(GeoZone, expected_attr)


def test_parse_str_uses_token_windows_for_free_text() -> None:
    assert GeoZone.parse_str("Sweden SE1 wind power") is GeoZone.SE1
    assert GeoZone.parse_str("flow around pjm dom peak hours") is GeoZone.PJM_DOM


@pytest.mark.parametrize(
    "value",
    [
        "47.3769, 8.5417",
        "47.3769 8.5417",
        "47.3769|8.5417",
        "47.3769;8.5417",
    ],
)
def test_parse_str_resolves_coordinate_strings(value: str) -> None:
    assert GeoZone.parse_str(value) is GeoZone.ZURICH


@pytest.mark.parametrize(
    ("value", "expected_attr"),
    [
        ({"key": "FR"}, "FRANCE"),
        ({"name": "Zurich"}, "ZURICH"),
        ({"eic": "10YFR-RTE------C"}, "FRANCE"),
        ({"lat": 47.3769, "lon": 8.5417}, "ZURICH"),
        ((47.3769, 8.5417), "ZURICH"),
    ],
)
def test_parse_accepts_multiple_input_shapes(value, expected_attr: str) -> None:
    assert GeoZone.parse(value) is getattr(GeoZone, expected_attr)


def test_parse_accepts_binary_wkb() -> None:
    assert GeoZone.parse(GeoZone.ZURICH.wkb) is GeoZone.ZURICH


@pytest.mark.parametrize(
    ("value", "return_value", "expected"),
    [
        ("France", "country_iso", "FR"),
        ("Zurich", "city_iso", "ZRH"),
    ],
)
def test_py_parse_str_scalar_fields(value: str, return_value: str, expected: str) -> None:
    assert GeoZone.py_parse_str(value, return_value=return_value) == expected


def test_py_parse_str_struct_and_dataclass() -> None:
    struct_value = GeoZone.py_parse_str("SE1", return_value="struct")
    dataclass_value = GeoZone.py_parse_str("SE1", return_value="dataclass")

    assert struct_value["key"] == "SE1"
    assert struct_value["country_iso"] == "SE"
    assert dataclass_value is GeoZone.SE1


@pytest.mark.parametrize(
    "return_value",
    ["wkb", "country_iso", "city_iso", "point", "struct", "dataclass"],
)
def test_py_parse_str_returns_nullish_values_for_missing_input(return_value: str) -> None:
    value = GeoZone.py_parse_str(None, return_value=return_value)

    if return_value == "point":
        assert value == {"lat": None, "lon": None}
    elif return_value == "struct":
        assert value["key"] is None
        assert value["wkb"] is None
    else:
        assert value is None


def test_py_parse_bin_variants() -> None:
    assert GeoZone.py_parse_bin(GeoZone.ZURICH.wkb, return_value="dataclass") is GeoZone.ZURICH
    assert GeoZone.py_parse_bin(GeoZone.ZURICH.wkb, return_value="country_iso") == "CH"
    assert GeoZone.py_parse_bin(GeoZone.ZURICH.wkb, return_value="point") == {
        "lat": pytest.approx(47.3769),
        "lon": pytest.approx(8.5417),
    }


@pytest.mark.parametrize(
    "value",
    ["", "   ", "definitely_not_a_zone", None],
)
def test_parse_str_returns_none_for_unknown_values(value) -> None:
    if value is None:
        assert GeoZone.parse(value) is None
    else:
        assert GeoZone.parse_str(value) is None


def test_load_geozones_sets_convenience_aliases() -> None:
    assert GeoZone.UK is GeoZone.UNITED_KINGDOM
    assert GeoZone.USA is GeoZone.UNITED_STATES
    assert GeoZone.US is GeoZone.UNITED_STATES


def test_clear_cache_empties_registries() -> None:
    GeoZone.clear_cache()

    assert GeoZone.CACHE_ALL == []
    assert GeoZone.CACHE_BY_KEY == {}
    assert GeoZone.CACHE_BY_NAME == {}
    assert GeoZone.CACHE_BY_GEOM == {}

    load_geozones()
    assert GeoZone.parse_str("France") is GeoZone.FRANCE
