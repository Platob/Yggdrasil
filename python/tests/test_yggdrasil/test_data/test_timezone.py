"""Unit tests for :class:`~yggdrasil.data.enums.timezone.Timezone`.

Coverage
────────
- Construction (direct, parse, parse_str)
- IANA name resolution, alias lookup, UTC offset parsing
- Pre-built constants (UTC, CET, EASTERN, …)
- Comparison / hashing / dunder methods
- utc_offset, utc_offset_hours
- is_utc, is_fixed_offset, is_dst, dst_offset
- abbreviation, distance_to
- localize, convert, midnight
- now, today (smoke)
- ZoneInfo interop (to_zoneinfo, key, tzinfo)
- Arrow integration (from_arrow_type, arrow_timestamp_type)
- Polars integration (from_polars_type, polars_normalize)
- all_iana enumeration
- Edge cases and error handling
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pyarrow as pa
import polars as pl
import pytest

from yggdrasil.enums.timezone import Timezone


# ===========================================================================
# Construction & parsing
# ===========================================================================

class TestConstruction:

    def test_direct_iana(self):
        tz = Timezone("Europe/Paris")
        assert tz.iana == "Europe/Paris"

    def test_str(self):
        assert str(Timezone("UTC")) == "UTC"

    def test_repr(self):
        assert repr(Timezone("Asia/Tokyo")) == "Timezone('Asia/Tokyo')"

    def test_frozen(self):
        tz = Timezone("UTC")
        with pytest.raises(AttributeError):
            tz.iana = "Europe/Paris"  # type: ignore[misc]


class TestParse:

    def test_parse_timezone_returns_same(self):
        tz = Timezone("UTC")
        assert Timezone.from_(tz) is tz

    def test_parse_none_returns_utc(self):
        assert Timezone.from_(None) == Timezone("UTC")

    def test_parse_zoneinfo(self):
        zi = ZoneInfo("Europe/Paris")
        tz = Timezone.from_(zi)
        assert tz.iana == "Europe/Paris"

    def test_parse_string(self):
        assert Timezone.from_("Europe/London").iana == "Europe/London"

    def test_parse_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="int"):
            Timezone.from_(42)


class TestParseStr:

    def test_exact_iana(self):
        assert Timezone.from_("America/New_York").iana == "America/New_York"

    def test_alias_cest(self):
        # CEST is not in IANA, resolved via alias
        assert Timezone.from_("CEST").iana == "Europe/Paris"

    def test_alias_et(self):
        # ET is not in IANA, resolved via alias
        assert Timezone.from_("ET").iana == "America/New_York"

    def test_alias_pt(self):
        # PT is not in IANA, resolved via alias
        assert Timezone.from_("PT").iana == "America/Los_Angeles"

    def test_alias_gmt(self):
        # GMT is a valid IANA name → returned as-is
        assert Timezone.from_("GMT").iana == "GMT"

    def test_alias_z(self):
        # Z is not in IANA, resolved via alias → UTC
        assert Timezone.from_("Z").iana == "UTC"

    def test_alias_jst(self):
        # JST may be a valid IANA name; if so, returned as-is
        result = Timezone.from_("JST")
        assert result.iana in ("JST", "Asia/Tokyo")

    def test_alias_case_insensitive(self):
        # "cest" is not IANA → alias lookup (case-insensitive)
        assert Timezone.from_("cest").iana == "Europe/Paris"

    def test_alias_edt(self):
        # EDT is not a valid IANA name → alias → America/New_York
        assert Timezone.from_("EDT").iana == "America/New_York"

    def test_alias_hkt(self):
        assert Timezone.from_("HKT").iana == "Asia/Hong_Kong"

    def test_utc_offset_plus(self):
        tz = Timezone.from_("+01:00")
        assert tz.iana == "Etc/GMT-1"

    def test_utc_offset_minus(self):
        tz = Timezone.from_("-05:00")
        assert tz.iana == "Etc/GMT+5"

    def test_utc_prefix_offset(self):
        tz = Timezone.from_("UTC+03:00")
        assert tz.iana == "Etc/GMT-3"

    def test_utc_prefix_minus(self):
        tz = Timezone.from_("UTC-08:00")
        assert tz.iana == "Etc/GMT+8"

    def test_offset_zero_returns_utc(self):
        assert Timezone.from_("+00:00").iana == "UTC"
        assert Timezone.from_("-00:00").iana == "UTC"

    def test_non_hour_aligned_raises(self):
        with pytest.raises(ValueError, match="non-hour-aligned"):
            Timezone.from_("+05:30")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Timezone.from_("")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown timezone"):
            Timezone.from_("Narnia/Wardrobe")

    def test_whitespace_stripped(self):
        assert Timezone.from_("  UTC  ").iana == "UTC"

    def test_non_string_raises(self):
        with pytest.raises(TypeError, match="Cannot derive Timezone"):
            Timezone.from_(42)  # type: ignore[arg-type]


class TestNaive:
    """``Timezone.NAIVE`` is the tz-less sentinel that replaces ``None``."""

    def test_naive_constant_exists(self):
        assert isinstance(Timezone.NAIVE, Timezone)

    def test_naive_is_falsy(self):
        # ``__bool__`` is False for NAIVE so callers can ``if self.tz:``
        # to mean "is timezone-aware".
        assert not Timezone.NAIVE
        assert bool(Timezone.NAIVE) is False
        assert bool(Timezone.UTC) is True

    def test_naive_str_is_empty(self):
        assert str(Timezone.NAIVE) == ""

    def test_naive_repr(self):
        assert repr(Timezone.NAIVE) == "Timezone.NAIVE"

    def test_naive_is_naive(self):
        assert Timezone.NAIVE.is_naive() is True
        assert Timezone.UTC.is_naive() is False
        assert Timezone.CET.is_naive() is False

    def test_naive_is_not_utc(self):
        assert Timezone.NAIVE.is_utc() is False

    def test_naive_to_zoneinfo_raises(self):
        with pytest.raises(ValueError, match="NAIVE"):
            Timezone.NAIVE.to_zoneinfo()

    def test_naive_utc_offset_raises(self):
        with pytest.raises(ValueError, match="NAIVE"):
            Timezone.NAIVE.utc_offset()

    def test_naive_localize_raises(self):
        with pytest.raises(ValueError, match="NAIVE"):
            Timezone.NAIVE.localize(dt.datetime(2024, 1, 1))

    def test_naive_convert_raises(self):
        aware = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        with pytest.raises(ValueError, match="NAIVE"):
            Timezone.NAIVE.convert(aware)

    def test_naive_dst_offset_zero(self):
        assert Timezone.NAIVE.dst_offset() == dt.timedelta(0)

    def test_naive_abbreviation_empty(self):
        assert Timezone.NAIVE.abbreviation() == ""

    def test_arrow_timestamp_type_drops_tz(self):
        t = Timezone.NAIVE.arrow_timestamp_type("us")
        assert t.tz is None or t.tz == ""


class TestFromExtended:
    """``Timezone.from_`` accepts richer Python shapes than parse used to."""

    def test_from_datetime_aware(self):
        aware = dt.datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Paris"))
        assert Timezone.from_(aware).iana == "Europe/Paris"

    def test_from_tzinfo(self):
        assert Timezone.from_(dt.timezone.utc).iana == "UTC"

    def test_from_tzinfo_fixed_offset(self):
        offset = dt.timezone(dt.timedelta(hours=3))
        assert Timezone.from_(offset).iana == "Etc/GMT-3"

    def test_from_object_with_tz_attribute(self):
        class FakeArrowTimestamp:
            tz = "Europe/Paris"

        assert Timezone.from_(FakeArrowTimestamp()).iana == "Europe/Paris"

    def test_from_object_with_time_zone_attribute(self):
        class FakePolarsDatetime:
            time_zone = "Asia/Tokyo"

        assert Timezone.from_(FakePolarsDatetime()).iana == "Asia/Tokyo"

    def test_from_default_returned_on_unknown(self):
        assert Timezone.from_("Narnia/Wardrobe", default=None) is None
        assert Timezone.from_(42, default=Timezone.UTC) is Timezone.UTC


# ===========================================================================
# Pre-built constants
# ===========================================================================

class TestConstants:

    def test_utc(self):
        assert Timezone.UTC.iana == "UTC"

    def test_cet(self):
        assert Timezone.CET.iana == "Europe/Paris"

    def test_wet(self):
        assert Timezone.WET.iana == "Europe/Lisbon"

    def test_eet(self):
        assert Timezone.EET.iana == "Europe/Helsinki"

    def test_eastern(self):
        assert Timezone.EASTERN.iana == "America/New_York"

    def test_central(self):
        assert Timezone.CENTRAL.iana == "America/Chicago"

    def test_mountain(self):
        assert Timezone.MOUNTAIN.iana == "America/Denver"

    def test_pacific(self):
        assert Timezone.PACIFIC.iana == "America/Los_Angeles"

    def test_jst(self):
        assert Timezone.JST.iana == "Asia/Tokyo"

    def test_sgt(self):
        assert Timezone.SGT.iana == "Asia/Singapore"


# ===========================================================================
# Comparison / hashing
# ===========================================================================

class TestComparison:

    def test_eq_same(self):
        assert Timezone("UTC") == Timezone("UTC")

    def test_eq_different(self):
        assert Timezone("UTC") != Timezone("Europe/Paris")

    def test_eq_string(self):
        assert Timezone("UTC") == "UTC"

    def test_eq_string_reverse(self):
        assert "UTC" == Timezone("UTC")

    def test_neq_string(self):
        assert Timezone("UTC") != "Europe/Paris"

    def test_eq_unsupported_type(self):
        assert Timezone("UTC") != 42

    def test_hash_consistency(self):
        a = Timezone("Europe/Paris")
        b = Timezone("Europe/Paris")
        assert hash(a) == hash(b)

    def test_hash_different(self):
        # Not strictly required, but useful to check
        assert hash(Timezone("UTC")) != hash(Timezone("Europe/Paris"))

    def test_usable_as_dict_key(self):
        d = {Timezone("UTC"): "utc", Timezone("Europe/Paris"): "paris"}
        assert d[Timezone("UTC")] == "utc"

    def test_usable_in_set(self):
        s = {Timezone("UTC"), Timezone("UTC"), Timezone("Europe/Paris")}
        assert len(s) == 2


# ===========================================================================
# ZoneInfo interop
# ===========================================================================

class TestZoneInfoInterop:

    def test_to_zoneinfo(self):
        tz = Timezone("Europe/Paris")
        zi = tz.to_zoneinfo()
        assert isinstance(zi, ZoneInfo)
        assert zi.key == "Europe/Paris"

    def test_to_zoneinfo_cached(self):
        tz = Timezone("Europe/Paris")
        assert tz.to_zoneinfo() is tz.to_zoneinfo()

    def test_key_property(self):
        assert Timezone("Asia/Tokyo").key == "Asia/Tokyo"

    def test_tzinfo_property(self):
        tz = Timezone("UTC")
        assert isinstance(tz.tzinfo, dt.tzinfo)


# ===========================================================================
# Offset / introspection
# ===========================================================================

class TestOffset:

    def test_utc_offset_utc(self):
        tz = Timezone("UTC")
        at = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        assert tz.utc_offset(at) == dt.timedelta(0)

    def test_utc_offset_paris_winter(self):
        """CET = UTC+1 in winter."""
        tz = Timezone("Europe/Paris")
        at = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        assert tz.utc_offset(at) == dt.timedelta(hours=1)

    def test_utc_offset_paris_summer(self):
        """CEST = UTC+2 in summer."""
        tz = Timezone("Europe/Paris")
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert tz.utc_offset(at) == dt.timedelta(hours=2)

    def test_utc_offset_naive_treated_as_utc(self):
        """Naive datetime is assumed UTC for offset calculation."""
        tz = Timezone("Europe/Paris")
        at = dt.datetime(2024, 7, 1, 12, 0)  # naive
        offset = tz.utc_offset(at)
        assert offset == dt.timedelta(hours=2)

    def test_utc_offset_default_now(self):
        """Calling with no argument should not raise."""
        offset = Timezone.UTC.utc_offset()
        assert offset == dt.timedelta(0)

    def test_utc_offset_hours_kolkata(self):
        at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert Timezone("Asia/Kolkata").utc_offset_hours(at) == 5.5

    def test_utc_offset_hours_new_york_winter(self):
        at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert Timezone.EASTERN.utc_offset_hours(at) == -5.0

    def test_utc_offset_hours_new_york_summer(self):
        at = dt.datetime(2024, 7, 1, tzinfo=dt.timezone.utc)
        assert Timezone.EASTERN.utc_offset_hours(at) == -4.0


class TestUtcSecondsOffset:
    """``utc_seconds_offset`` returns the fixed offset in seconds.

    DST zones and :attr:`Timezone.NAIVE` return ``None``. Etc/GMT
    follows IANA's reversed sign convention — ``Etc/GMT-3`` means
    UTC+3, so the property reports ``+10800``.
    """

    def test_utc_returns_zero(self):
        assert Timezone.UTC.utc_seconds_offset == 0
        assert Timezone("Etc/UTC").utc_seconds_offset == 0
        assert Timezone("GMT").utc_seconds_offset == 0
        assert Timezone("Etc/GMT+0").utc_seconds_offset == 0

    def test_etc_gmt_negative_means_east_of_utc(self):
        assert Timezone("Etc/GMT-1").utc_seconds_offset == 3600
        assert Timezone("Etc/GMT-3").utc_seconds_offset == 10800
        assert Timezone("Etc/GMT-12").utc_seconds_offset == 12 * 3600

    def test_etc_gmt_positive_means_west_of_utc(self):
        assert Timezone("Etc/GMT+5").utc_seconds_offset == -5 * 3600
        assert Timezone("Etc/GMT+8").utc_seconds_offset == -8 * 3600

    def test_dst_zones_return_none(self):
        # ``utc_offset(at)`` is the call to use for DST-aware values.
        assert Timezone("Europe/Paris").utc_seconds_offset is None
        assert Timezone.EASTERN.utc_seconds_offset is None
        assert Timezone("Asia/Tokyo").utc_seconds_offset is None

    def test_naive_returns_none(self):
        assert Timezone.NAIVE.utc_seconds_offset is None

    def test_offset_strings_round_trip(self):
        assert Timezone.from_("+01:00").utc_seconds_offset == 3600
        assert Timezone.from_("-05:00").utc_seconds_offset == -5 * 3600
        assert Timezone.from_("+00:00").utc_seconds_offset == 0


class TestIntrospection:

    def test_is_utc_true(self):
        assert Timezone("UTC").is_utc() is True
        assert Timezone("Etc/UTC").is_utc() is True
        assert Timezone("GMT").is_utc() is True

    def test_is_utc_false(self):
        assert Timezone("Europe/Paris").is_utc() is False

    def test_is_fixed_offset_etc(self):
        assert Timezone("Etc/GMT-5").is_fixed_offset() is True

    def test_is_fixed_offset_utc(self):
        assert Timezone("UTC").is_fixed_offset() is True

    def test_is_fixed_offset_false(self):
        assert Timezone("Europe/Paris").is_fixed_offset() is False

    def test_is_dst_paris_summer(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.is_dst(at) is True

    def test_is_dst_paris_winter(self):
        at = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.is_dst(at) is False

    def test_is_dst_utc_never(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.UTC.is_dst(at) is False

    def test_dst_offset_summer(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.dst_offset(at) == dt.timedelta(hours=1)

    def test_dst_offset_winter(self):
        at = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.dst_offset(at) == dt.timedelta(0)

    def test_abbreviation_paris_summer(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.abbreviation(at) == "CEST"

    def test_abbreviation_paris_winter(self):
        at = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.abbreviation(at) == "CET"

    def test_abbreviation_utc(self):
        at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert Timezone.UTC.abbreviation(at) == "UTC"


class TestDistanceTo:

    def test_utc_to_cet_summer(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.UTC.distance_to(Timezone.CET, at) == dt.timedelta(hours=2)

    def test_utc_to_cet_winter(self):
        at = dt.datetime(2024, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.UTC.distance_to(Timezone.CET, at) == dt.timedelta(hours=1)

    def test_cet_to_eastern_summer(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        d = Timezone.CET.distance_to(Timezone.EASTERN, at)
        # CEST=+2, EDT=-4 → distance = -6h
        assert d == dt.timedelta(hours=-6)

    def test_same_tz_zero(self):
        at = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        assert Timezone.CET.distance_to(Timezone.CET, at) == dt.timedelta(0)


# ===========================================================================
# Clock
# ===========================================================================

class TestClock:

    def test_now_returns_aware(self):
        result = Timezone.UTC.now()
        assert result.tzinfo is not None

    def test_now_utc_timezone_matches(self):
        result = Timezone.UTC.now()
        # Should be within a few seconds of utcnow
        diff = abs((result - dt.datetime.now(dt.timezone.utc)).total_seconds())
        assert diff < 2

    def test_today_returns_date(self):
        result = Timezone.UTC.today()
        assert isinstance(result, dt.date)

    def test_now_paris_is_aware(self):
        result = Timezone.CET.now()
        assert result.tzinfo is not None


# ===========================================================================
# Conversion helpers
# ===========================================================================

class TestLocalize:

    def test_localize_naive(self):
        naive = dt.datetime(2024, 6, 1, 12, 0)
        result = Timezone.CET.localize(naive)
        assert result.tzinfo is not None
        assert result.year == 2024

    def test_localize_aware_raises(self):
        aware = dt.datetime(2024, 6, 1, 12, 0, tzinfo=dt.timezone.utc)
        with pytest.raises(ValueError, match="Cannot localize"):
            Timezone.CET.localize(aware)


class TestConvert:

    def test_convert_utc_to_paris(self):
        utc_dt = dt.datetime(2024, 7, 1, 12, 0, tzinfo=dt.timezone.utc)
        result = Timezone.CET.convert(utc_dt)
        assert result.hour == 14  # UTC+2 in summer

    def test_convert_paris_to_new_york(self):
        paris_dt = Timezone.CET.localize(dt.datetime(2024, 7, 1, 14, 0))
        result = Timezone.EASTERN.convert(paris_dt)
        # CEST→EDT: 14:00 Paris = 12:00 UTC = 08:00 EDT
        assert result.hour == 8

    def test_convert_naive_raises(self):
        naive = dt.datetime(2024, 6, 1, 12, 0)
        with pytest.raises(ValueError, match="Cannot convert a naive"):
            Timezone.CET.convert(naive)


class TestMidnight:

    def test_midnight_specific_date(self):
        result = Timezone.CET.midnight(dt.date(2024, 3, 15))
        assert result.hour == 0
        assert result.minute == 0
        assert result.day == 15
        assert result.tzinfo is not None

    def test_midnight_default_today(self):
        result = Timezone.UTC.midnight()
        assert result.hour == 0
        assert result.date() == Timezone.UTC.today()


# ===========================================================================
# Arrow integration
# ===========================================================================

class TestArrowIntegration:

    def test_from_arrow_type_timestamp(self):
        t = pa.timestamp("us", tz="Europe/Paris")
        tz = Timezone.from_arrow_type(t)
        assert tz is not None
        assert tz.iana == "Europe/Paris"

    def test_from_arrow_type_no_tz(self):
        t = pa.timestamp("us")
        assert Timezone.from_arrow_type(t) is None

    def test_from_arrow_type_non_timestamp(self):
        assert Timezone.from_arrow_type(pa.int64()) is None

    def test_arrow_timestamp_type(self):
        t = Timezone.CET.arrow_timestamp_type("ns")
        assert pa.types.is_timestamp(t)
        assert t.tz == "Europe/Paris"
        assert t.unit == "ns"

    def test_arrow_timestamp_type_default_unit(self):
        t = Timezone.UTC.arrow_timestamp_type()
        assert t.unit == "us"
        assert t.tz == "UTC"


# ===========================================================================
# Polars integration
# ===========================================================================

class TestPolarsIntegration:

    def test_from_polars_type_with_tz(self):
        tz = Timezone.from_polars_type(pl.Datetime("us", "Europe/Paris"))
        assert tz is not None
        assert tz.iana == "Europe/Paris"

    def test_from_polars_type_without_tz(self):
        assert Timezone.from_polars_type(pl.Datetime("us")) is None

    def test_from_polars_type_non_datetime(self):
        assert Timezone.from_polars_type(pl.Int64()) is None

    def test_polars_normalize_series(self):
        s = pl.Series("tz", ["CEST", "EDT", "UTC", "Bogus", None])
        result = Timezone.polars_normalize(s, lazy=True)
        values = result.to_list()
        assert values[0] == "Europe/Paris"       # CEST → alias
        assert values[1] == "America/New_York"   # EDT → alias
        assert values[2] == "UTC"
        assert values[3] is None
        assert values[4] is None

    def test_polars_normalize_series_map_elements(self):
        s = pl.Series("tz", ["PT", "HKT"])
        result = Timezone.polars_normalize(s)
        assert result.to_list() == ["America/Los_Angeles", "Asia/Hong_Kong"]

    def test_polars_normalize_expr(self):
        df = pl.DataFrame({"tz": ["CEST", "UTC"]})
        expr = Timezone.polars_normalize(pl.col("tz"))
        result = df.select(expr).to_series()
        assert result.to_list() == ["Europe/Paris", "UTC"]

    def test_polars_normalize_unsupported_return_value(self):
        s = pl.Series("tz", ["UTC"])
        with pytest.raises(ValueError, match="Unsupported return_value"):
            Timezone.polars_normalize(s, return_value="offset")  # type: ignore[arg-type]

    def test_polars_normalize_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Expected polars"):
            Timezone.polars_normalize("CET")  # type: ignore[arg-type]


# ===========================================================================
# Enumeration
# ===========================================================================

class TestEnumeration:

    def test_all_iana_returns_frozenset(self):
        result = Timezone.all_iana()
        assert isinstance(result, frozenset)

    def test_all_iana_contains_utc(self):
        assert "UTC" in Timezone.all_iana()

    def test_all_iana_contains_europe_paris(self):
        assert "Europe/Paris" in Timezone.all_iana()

    def test_all_iana_large(self):
        assert len(Timezone.all_iana()) > 400


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_etc_gmt_minus_sign_convention(self):
        """IANA Etc/GMT sign is *inverted*: Etc/GMT-1 means UTC+1."""
        tz = Timezone("Etc/GMT-1")
        at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert tz.utc_offset(at) == dt.timedelta(hours=1)

    def test_parse_offset_without_colon(self):
        tz = Timezone.from_("+0300")
        assert tz.iana == "Etc/GMT-3"

    def test_parse_utc_without_offset(self):
        assert Timezone.from_("UTC").iana == "UTC"

    def test_multiple_alias_forms_converge(self):
        """ET, EDT should resolve to America/New_York via alias."""
        expected = "America/New_York"
        for alias in ("ET", "EDT", "et", "edt"):
            assert Timezone.from_(alias).iana == expected, f"Failed for {alias}"

    def test_constant_identity(self):
        """Constants are Timezone instances."""
        assert isinstance(Timezone.UTC, Timezone)
        assert isinstance(Timezone.CET, Timezone)

    def test_constant_eq_parse(self):
        assert Timezone.UTC == Timezone.from_("UTC")
        assert Timezone.CET == Timezone.from_("Europe/Paris")

