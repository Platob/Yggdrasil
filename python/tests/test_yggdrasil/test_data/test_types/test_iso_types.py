"""Tests for ISO data types (country, currency, continent, subdivision, city).

Covers the full cast matrix: Python object, Arrow array, Polars Series/Expr,
Pandas Series.  ``safe=True`` raises on eager paths, never on lazy ones.
"""
from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.iso import (
    ISOCityType,
    ISOContinentType,
    ISOCountryType,
    ISOCurrencyType,
    ISOSubdivisionType,
    ISOType,
    TimezoneType,
)
from yggdrasil.polars.tests import PolarsTestCase


class _Opts:
    """Minimal CastOptions stand-in used by the cast_* internals."""

    def __init__(self, safe: bool = False):
        self.safe = safe
        self.target_field = None
        self.source_field = None

    def check_source(self, obj):
        return self

    def copy(self, **kw):
        new = _Opts(safe=kw.get("safe", self.safe))
        return new


# ---------------------------------------------------------------------------
# ISOCountryType
# ---------------------------------------------------------------------------


class TestISOCountryTypePyObj(unittest.TestCase):
    def test_alpha2_default(self):
        t = ISOCountryType()
        self.assertEqual(t.alpha, 2)

    def test_alpha_invalid(self):
        with self.assertRaises(ValueError):
            ISOCountryType(alpha=4)

    def test_parse_name(self):
        t = ISOCountryType()
        self.assertEqual(t.convert_pyobj("France", nullable=True), "FR")

    def test_parse_alpha3_input(self):
        t = ISOCountryType()
        self.assertEqual(t.convert_pyobj("FRA", nullable=True), "FR")

    def test_parse_alpha2_input(self):
        t = ISOCountryType()
        self.assertEqual(t.convert_pyobj("fr", nullable=True), "FR")

    def test_parse_numeric(self):
        t = ISOCountryType()
        self.assertEqual(t.convert_pyobj("250", nullable=True), "FR")

    def test_alpha3_output(self):
        t = ISOCountryType(alpha=3)
        self.assertEqual(t.convert_pyobj("France", nullable=True), "FRA")
        self.assertEqual(t.convert_pyobj("250", nullable=True), "FRA")
        self.assertEqual(t.convert_pyobj("US", nullable=True), "USA")

    def test_alias_uk(self):
        self.assertEqual(ISOCountryType().convert_pyobj("UK", nullable=True), "GB")

    def test_alias_usa(self):
        self.assertEqual(ISOCountryType().convert_pyobj("USA", nullable=True), "US")

    def test_unknown_safe_false_returns_none(self):
        self.assertIsNone(ISOCountryType().convert_pyobj("zzzzz", nullable=True))

    def test_unknown_safe_true_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            ISOCountryType().convert_pyobj("zzzzz", nullable=True, safe=True)

    def test_none_nullable(self):
        self.assertIsNone(ISOCountryType().convert_pyobj(None, nullable=True))

    def test_none_not_nullable_raises(self):
        with self.assertRaises(ValueError):
            ISOCountryType().convert_pyobj(None, nullable=False)

    def test_repr_str(self):
        t = ISOCountryType(alpha=3)
        self.assertEqual(repr(t), "ISOCountryType(alpha=3)")
        self.assertEqual(str(t), "iso_country(3)")


class TestISOCountryTypeArrow(ArrowTestCase):
    def test_cast_string_alpha2(self):
        t = ISOCountryType()
        arr = pa.array(["France", "UK", "USA", "xyz", None, "250"])
        out = t._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertEqual(out, ["FR", "GB", "US", None, None, "FR"])

    def test_cast_alpha3(self):
        t = ISOCountryType(alpha=3)
        arr = pa.array(["France", "UK", "250"])
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["FRA", "GBR", "FRA"],
        )

    def test_cast_safe_true_raises(self):
        t = ISOCountryType()
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            t._cast_arrow_array(pa.array(["nosuch"]), _Opts(safe=True))

    def test_cast_safe_false_nulls(self):
        t = ISOCountryType()
        out = t._cast_arrow_array(pa.array(["nosuch"]), _Opts(safe=False))
        self.assertEqual(out.to_pylist(), [None])


class TestISOCountryTypePolars(PolarsTestCase):
    def test_eager_series(self):
        pl = self.pl
        s = pl.Series("c", ["France", "UK", "USA", None])
        out = ISOCountryType()._cast_polars_series(s, _Opts())
        self.assertEqual(out.to_list(), ["FR", "GB", "US", None])

    def test_lazy_expr(self):
        pl = self.pl
        df = pl.DataFrame({"c": ["France", "xyz", None]})
        t = ISOCountryType(alpha=3)
        out = df.select(t._cast_polars_expr(pl.col("c"), _Opts()).alias("c"))
        self.assertEqual(out["c"].to_list(), ["FRA", None, None])

    def test_lazy_never_raises_with_safe_true(self):
        """safe=True on lazy paths silently yields nulls (values not observable)."""
        pl = self.pl
        df = pl.DataFrame({"c": ["nosuch"]})
        t = ISOCountryType()
        out = df.select(t._cast_polars_expr(pl.col("c"), _Opts(safe=True)).alias("c"))
        self.assertEqual(out["c"].to_list(), [None])


class TestISOCountryTypeDict(unittest.TestCase):
    def test_to_dict(self):
        d = ISOCountryType(alpha=3).to_dict()
        self.assertEqual(d["name"], "ISOCountryType")
        self.assertEqual(d["iso"], "iso_country")
        self.assertEqual(d["alpha"], 3)

    def test_from_dict(self):
        d = {"name": "ISOCountryType", "alpha": 3}
        t = ISOCountryType.from_dict(d)
        self.assertEqual(t.alpha, 3)

    def test_handles_dict(self):
        self.assertTrue(ISOCountryType.handles_dict({"name": "ISOCountryType"}))
        self.assertTrue(ISOCountryType.handles_dict({"iso": "iso_country"}))


# ---------------------------------------------------------------------------
# ISOCurrencyType
# ---------------------------------------------------------------------------


class TestISOCurrencyTypePyObj(unittest.TestCase):
    def test_alpha3_code(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("USD", nullable=True), "USD")

    def test_lowercase(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("usd", nullable=True), "USD")

    def test_symbol_dollar(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("$", nullable=True), "USD")

    def test_symbol_euro(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("\u20AC", nullable=True), "EUR")

    def test_symbol_pound(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("\u00A3", nullable=True), "GBP")

    def test_symbol_yen(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("\u00A5", nullable=True), "JPY")

    def test_name(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("dollar", nullable=True), "USD")

    def test_numeric(self):
        self.assertEqual(ISOCurrencyType().convert_pyobj("978", nullable=True), "EUR")

    def test_unknown_safe_false(self):
        self.assertIsNone(ISOCurrencyType().convert_pyobj("nope", nullable=True))

    def test_unknown_safe_true_raises(self):
        with self.assertRaises(ValueError):
            ISOCurrencyType().convert_pyobj("nope", nullable=True, safe=True)


class TestISOCurrencyTypeArrow(ArrowTestCase):
    def test_cast_mixed(self):
        c = ISOCurrencyType()
        arr = pa.array(["USD", "\u20AC", "$", "dollar", "978", "garbage", None])
        self.assertEqual(
            c._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["USD", "EUR", "USD", "USD", "EUR", None, None],
        )


class TestISOCurrencyTypePolars(PolarsTestCase):
    def test_lazy_expr(self):
        pl = self.pl
        df = pl.DataFrame({"c": ["USD", "\u20AC", "dollar", "nope", None]})
        c = ISOCurrencyType()
        out = df.select(c._cast_polars_expr(pl.col("c"), _Opts()).alias("c"))
        self.assertEqual(out["c"].to_list(), ["USD", "EUR", "USD", None, None])


# ---------------------------------------------------------------------------
# ISOContinentType
# ---------------------------------------------------------------------------


class TestISOContinentType(unittest.TestCase):
    def test_alpha2(self):
        self.assertEqual(ISOContinentType().convert_pyobj("EU", nullable=True), "EU")

    def test_name(self):
        self.assertEqual(ISOContinentType().convert_pyobj("Europe", nullable=True), "EU")

    def test_multi_word(self):
        self.assertEqual(
            ISOContinentType().convert_pyobj("North America", nullable=True), "NA"
        )

    def test_squashed(self):
        self.assertEqual(
            ISOContinentType().convert_pyobj("SOUTHAMERICA", nullable=True), "SA"
        )

    def test_unknown(self):
        self.assertIsNone(ISOContinentType().convert_pyobj("moon", nullable=True))


class TestISOContinentTypeArrow(ArrowTestCase):
    def test_cast(self):
        t = ISOContinentType()
        arr = pa.array(["Europe", "NA", "North America", "SOUTHAMERICA", "xx", None])
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["EU", "NA", "NA", "SA", None, None],
        )


# ---------------------------------------------------------------------------
# ISOSubdivisionType
# ---------------------------------------------------------------------------


class TestISOSubdivisionType(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("US-CA", nullable=True), "US-CA"
        )

    def test_alpha3_prefix(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("USA-CA", nullable=True), "US-CA"
        )

    def test_numeric_subdivision(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("FR-75", nullable=True), "FR-75"
        )

    def test_country_name_with_code(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("France 75", nullable=True), "FR-75"
        )

    def test_country_name_with_subdivision_name(self):
        # Berlin -> "BER" (first 3 alnum of the remainder).
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("Germany - Berlin", nullable=True),
            "DE-BER",
        )

    def test_multi_word_country(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("UNITED STATES CA", nullable=True),
            "US-CA",
        )

    def test_country_only_trailing_dash(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("CH-", nullable=True), "CH"
        )

    def test_country_only_bare(self):
        self.assertEqual(
            ISOSubdivisionType().convert_pyobj("GB", nullable=True), "GB"
        )

    def test_unknown_country(self):
        self.assertIsNone(
            ISOSubdivisionType().convert_pyobj("ZZ-CA", nullable=True)
        )

    def test_bad_format(self):
        self.assertIsNone(
            ISOSubdivisionType().convert_pyobj("!!!", nullable=True)
        )


class TestISOSubdivisionTypeArrow(ArrowTestCase):
    def test_cast(self):
        t = ISOSubdivisionType()
        arr = pa.array(["US-CA", "FR-75", "GB-ENG", "xx-yy", None])
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["US-CA", "FR-75", "GB-ENG", None, None],
        )

    def test_cast_flexible(self):
        t = ISOSubdivisionType()
        arr = pa.array(
            [
                "France 75",
                "Germany - Berlin",
                "UNITED STATES CA",
                "CH-",
                "USA-CA",
                None,
            ]
        )
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["FR-75", "DE-BER", "US-CA", "CH", "US-CA", None],
        )


# ---------------------------------------------------------------------------
# ISOCityType
# ---------------------------------------------------------------------------


class TestISOCityType(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(ISOCityType().convert_pyobj("FR-PAR", nullable=True), "FR-PAR")

    def test_space_separator(self):
        self.assertEqual(ISOCityType().convert_pyobj("FR PAR", nullable=True), "FR-PAR")

    def test_no_separator(self):
        self.assertEqual(ISOCityType().convert_pyobj("FRPAR", nullable=True), "FR-PAR")

    def test_alpha3_prefix(self):
        self.assertEqual(ISOCityType().convert_pyobj("USA-NYC", nullable=True), "US-NYC")

    def test_country_name_city_name(self):
        self.assertEqual(
            ISOCityType().convert_pyobj("France Paris", nullable=True), "FR-PAR"
        )

    def test_trailing_dash_country_only(self):
        self.assertEqual(ISOCityType().convert_pyobj("FR-", nullable=True), "FR")

    def test_short_city_truncated(self):
        # "PA" is only 2 chars but we accept partial input; result is "FR-PA".
        self.assertEqual(ISOCityType().convert_pyobj("FR-PA", nullable=True), "FR-PA")

    def test_unknown_country(self):
        self.assertIsNone(ISOCityType().convert_pyobj("ZZ-YYY", nullable=True))


class TestISOSubdivisionTypePolars(PolarsTestCase):
    def test_lazy_expr_flexible(self):
        pl = self.pl
        df = pl.DataFrame(
            {"c": ["France 75", "Germany - Berlin", "CH-", "xxx", None]}
        )
        out = df.select(
            ISOSubdivisionType()._cast_polars_expr(pl.col("c"), _Opts()).alias("c")
        )
        self.assertEqual(out["c"].to_list(), ["FR-75", "DE-BER", "CH", None, None])


class TestISOCityTypeArrow(ArrowTestCase):
    def test_cast(self):
        t = ISOCityType()
        arr = pa.array(["FR-PAR", "FRPAR", "xx-yyy", None])
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["FR-PAR", "FR-PAR", None, None],
        )

    def test_cast_flexible(self):
        t = ISOCityType()
        arr = pa.array(["France Paris", "Germany - Berlin", "USA-NYC", "CH-", None])
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            ["FR-PAR", "DE-BER", "US-NYC", "CH", None],
        )


# ---------------------------------------------------------------------------
# TimezoneType
# ---------------------------------------------------------------------------


class TestTimezoneTypePyObj(unittest.TestCase):
    def test_canonical_passthrough(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("Europe/Paris", nullable=True),
            "Europe/Paris",
        )

    def test_case_insensitive(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("europe/paris", nullable=True),
            "Europe/Paris",
        )
        self.assertEqual(
            TimezoneType().convert_pyobj("EUROPE/PARIS", nullable=True),
            "Europe/Paris",
        )

    def test_utc_alias(self):
        self.assertEqual(TimezoneType().convert_pyobj("UTC", nullable=True), "Etc/UTC")
        self.assertEqual(TimezoneType().convert_pyobj("utc", nullable=True), "Etc/UTC")
        self.assertEqual(TimezoneType().convert_pyobj("Zulu", nullable=True), "Etc/UTC")

    def test_legacy_abbrev_rewritten(self):
        # CET/MET/WET etc. are not treated as canonical — they map onto a
        # representative Area/Location zone.
        self.assertEqual(TimezoneType().convert_pyobj("CET", nullable=True), "Europe/Paris")
        self.assertEqual(TimezoneType().convert_pyobj("MET", nullable=True), "Europe/Paris")
        self.assertEqual(TimezoneType().convert_pyobj("EET", nullable=True), "Europe/Athens")
        self.assertEqual(TimezoneType().convert_pyobj("WET", nullable=True), "Europe/Lisbon")
        self.assertEqual(TimezoneType().convert_pyobj("MST", nullable=True), "America/Phoenix")
        self.assertEqual(TimezoneType().convert_pyobj("HST", nullable=True), "Pacific/Honolulu")
        self.assertEqual(
            TimezoneType().convert_pyobj("PST8PDT", nullable=True),
            "America/Los_Angeles",
        )

    def test_backward_link_aliases(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("US/Eastern", nullable=True),
            "America/New_York",
        )
        self.assertEqual(
            TimezoneType().convert_pyobj("Asia/Calcutta", nullable=True),
            "Asia/Kolkata",
        )
        self.assertEqual(
            TimezoneType().convert_pyobj("GB", nullable=True),
            "Europe/London",
        )

    def test_whitespace_around_separator(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("Europe / Paris", nullable=True),
            "Europe/Paris",
        )

    def test_space_inside_location(self):
        # Real IANA names use underscores where there would be spaces.
        self.assertEqual(
            TimezoneType().convert_pyobj("America/New York", nullable=True),
            "America/New_York",
        )

    def test_windows_backslash(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("Europe\\Paris", nullable=True),
            "Europe/Paris",
        )

    def test_etc_signed_offset(self):
        self.assertEqual(
            TimezoneType().convert_pyobj("Etc/GMT+5", nullable=True),
            "Etc/GMT+5",
        )

    def test_unknown_safe_false(self):
        self.assertIsNone(TimezoneType().convert_pyobj("Mars/Olympus", nullable=True))

    def test_unknown_safe_true_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            TimezoneType().convert_pyobj("Mars/Olympus", nullable=True, safe=True)

    def test_none_nullable(self):
        self.assertIsNone(TimezoneType().convert_pyobj(None, nullable=True))

    def test_none_not_nullable_raises(self):
        with self.assertRaises(ValueError):
            TimezoneType().convert_pyobj(None, nullable=False)

    def test_repr_str(self):
        t = TimezoneType()
        self.assertEqual(repr(t), "TimezoneType()")
        self.assertEqual(str(t), "timezone")


class TestTimezoneTypeOffsets(unittest.TestCase):
    """UTC-offset strings are preserved in ±HH:MM canonical form."""

    def test_iso_offset_positive(self):
        self.assertEqual(TimezoneType().convert_pyobj("+05:00", nullable=True), "+05:00")

    def test_iso_offset_negative(self):
        self.assertEqual(TimezoneType().convert_pyobj("-08:00", nullable=True), "-08:00")

    def test_short_offset_padded(self):
        self.assertEqual(TimezoneType().convert_pyobj("+5", nullable=True), "+05:00")
        self.assertEqual(TimezoneType().convert_pyobj("-8", nullable=True), "-08:00")

    def test_compact_offset(self):
        self.assertEqual(TimezoneType().convert_pyobj("+0530", nullable=True), "+05:30")
        self.assertEqual(TimezoneType().convert_pyobj("-0800", nullable=True), "-08:00")

    def test_utc_gmt_prefix(self):
        self.assertEqual(TimezoneType().convert_pyobj("UTC+5", nullable=True), "+05:00")
        self.assertEqual(TimezoneType().convert_pyobj("GMT-0530", nullable=True), "-05:30")
        self.assertEqual(TimezoneType().convert_pyobj("UTC+14", nullable=True), "+14:00")

    def test_bare_z(self):
        self.assertEqual(TimezoneType().convert_pyobj("Z", nullable=True), "+00:00")

    def test_out_of_range_rejected(self):
        self.assertIsNone(TimezoneType().convert_pyobj("+25:00", nullable=True))
        self.assertIsNone(TimezoneType().convert_pyobj("UTC+25", nullable=True))
        self.assertIsNone(TimezoneType().convert_pyobj("+05:99", nullable=True))


class TestTimezoneTypeCrossType(unittest.TestCase):
    """tzinfo / ZoneInfo / timedelta inputs convert into canonical strings."""

    def test_zoneinfo_input(self):
        from zoneinfo import ZoneInfo

        self.assertEqual(
            TimezoneType().convert_pyobj(ZoneInfo("Europe/Paris"), nullable=True),
            "Europe/Paris",
        )

    def test_datetime_timezone_utc_input(self):
        import datetime as dt

        self.assertEqual(
            TimezoneType().convert_pyobj(dt.timezone.utc, nullable=True),
            "+00:00",
        )

    def test_datetime_timezone_offset_input(self):
        import datetime as dt

        self.assertEqual(
            TimezoneType().convert_pyobj(
                dt.timezone(dt.timedelta(hours=-5)), nullable=True
            ),
            "-05:00",
        )

    def test_timedelta_input(self):
        import datetime as dt

        self.assertEqual(
            TimezoneType().convert_pyobj(
                dt.timedelta(hours=5, minutes=30), nullable=True
            ),
            "+05:30",
        )
        self.assertEqual(
            TimezoneType().convert_pyobj(dt.timedelta(hours=-8), nullable=True),
            "-08:00",
        )

    def test_to_tzinfo_iana(self):
        from zoneinfo import ZoneInfo

        self.assertEqual(TimezoneType.to_tzinfo("Europe/Paris"), ZoneInfo("Europe/Paris"))

    def test_to_tzinfo_offset(self):
        import datetime as dt

        self.assertEqual(
            TimezoneType.to_tzinfo("+05:30"),
            dt.timezone(dt.timedelta(hours=5, minutes=30)),
        )

    def test_to_timedelta(self):
        import datetime as dt

        self.assertEqual(
            TimezoneType.to_timedelta("-08:00"), dt.timedelta(hours=-8)
        )

    def test_registry_tzinfo_to_str(self):
        import datetime as dt

        from yggdrasil.data.cast import convert

        self.assertEqual(
            convert(dt.timezone(dt.timedelta(hours=-5)), str), "-05:00"
        )

    def test_registry_zoneinfo_to_str(self):
        from zoneinfo import ZoneInfo

        from yggdrasil.data.cast import convert

        self.assertEqual(convert(ZoneInfo("Europe/Paris"), str), "Europe/Paris")

    def test_registry_timedelta_to_str(self):
        import datetime as dt

        from yggdrasil.data.cast import convert

        self.assertEqual(convert(dt.timedelta(hours=5, minutes=30), str), "+05:30")

    def test_registry_str_to_zoneinfo(self):
        from zoneinfo import ZoneInfo

        from yggdrasil.data.cast import convert

        self.assertEqual(convert("Europe/Paris", ZoneInfo), ZoneInfo("Europe/Paris"))


class TestTimezoneTypeArrow(ArrowTestCase):
    def test_cast_mixed(self):
        t = TimezoneType()
        arr = pa.array(
            [
                "UTC",
                "Europe/Paris",
                "europe / paris",
                "CET",
                "US/Eastern",
                "Asia/Calcutta",
                "America/New York",
                None,
                "Mars/Olympus",
            ]
        )
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            [
                "Etc/UTC",
                "Europe/Paris",
                "Europe/Paris",
                "Europe/Paris",
                "America/New_York",
                "Asia/Kolkata",
                "America/New_York",
                None,
                None,
            ],
        )

    def test_cast_offsets(self):
        t = TimezoneType()
        arr = pa.array(
            [
                "+05:00",
                "-08:00",
                "UTC+5",
                "+0530",
                "Z",
                "UTC+25",
                "+25:00",
                "GMT-0530",
            ]
        )
        self.assertEqual(
            t._cast_arrow_array(arr, _Opts()).to_pylist(),
            [
                "+05:00",
                "-08:00",
                "+05:00",
                "+05:30",
                "+00:00",
                None,
                None,
                "-05:30",
            ],
        )

    def test_cast_safe_true_raises(self):
        t = TimezoneType()
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            t._cast_arrow_array(pa.array(["Mars/Olympus"]), _Opts(safe=True))

    def test_cast_safe_false_nulls(self):
        t = TimezoneType()
        out = t._cast_arrow_array(pa.array(["Mars/Olympus"]), _Opts(safe=False))
        self.assertEqual(out.to_pylist(), [None])


class TestTimezoneTypePolars(PolarsTestCase):
    def test_eager_series(self):
        pl = self.pl
        s = pl.Series("tz", ["UTC", "Europe/Paris", "CET", None, "nope"])
        out = TimezoneType()._cast_polars_series(s, _Opts())
        self.assertEqual(
            out.to_list(), ["Etc/UTC", "Europe/Paris", "Europe/Paris", None, None]
        )

    def test_eager_offsets(self):
        pl = self.pl
        s = pl.Series("tz", ["+05:00", "UTC+5", "Z", "UTC+25", "GMT-0530"])
        out = TimezoneType()._cast_polars_series(s, _Opts())
        self.assertEqual(
            out.to_list(), ["+05:00", "+05:00", "+00:00", None, "-05:30"]
        )

    def test_lazy_expr(self):
        pl = self.pl
        df = pl.DataFrame(
            {"tz": ["UTC", "Europe/Paris", "CET", "US/Eastern", None, "nope"]}
        )
        out = df.select(
            TimezoneType()._cast_polars_expr(pl.col("tz"), _Opts()).alias("tz")
        )
        self.assertEqual(
            out["tz"].to_list(),
            ["Etc/UTC", "Europe/Paris", "Europe/Paris", "America/New_York", None, None],
        )

    def test_lazy_offsets(self):
        pl = self.pl
        df = pl.DataFrame({"tz": ["+05:00", "UTC-8", "Z", "+25:00"]})
        out = df.select(
            TimezoneType()._cast_polars_expr(pl.col("tz"), _Opts()).alias("tz")
        )
        self.assertEqual(
            out["tz"].to_list(), ["+05:00", "-08:00", "+00:00", None]
        )


class TestTimezoneTypeDict(unittest.TestCase):
    def test_to_dict(self):
        d = TimezoneType().to_dict()
        self.assertEqual(d["name"], "TimezoneType")
        self.assertEqual(d["iso"], "timezone")

    def test_handles_dict(self):
        self.assertTrue(TimezoneType.handles_dict({"name": "TimezoneType"}))
        self.assertTrue(TimezoneType.handles_dict({"iso": "timezone"}))

    def test_from_dict_round_trip(self):
        t1 = TimezoneType()
        t2 = TimezoneType.from_dict(t1.to_dict())
        self.assertEqual(t1, t2)


# ---------------------------------------------------------------------------
# ISOType base — framework integration invariants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-type (outgoing) casts from ISO sources.
# ---------------------------------------------------------------------------


class TestISOCrossTypeArrow(ArrowTestCase):
    """ISO code arrays should cast cleanly to numeric / string / duration
    targets through the ``_outgoing_cast_arrow_array`` hook on the source type.
    """

    def _field(self, name, dtype):
        from yggdrasil.data.data_field import Field
        return Field(name=name, dtype=dtype)

    def test_country_alpha2_to_int(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("n", IntegerType(byte_size=4, signed=True))
        arr = pa.array(["FR", "US", "DE", None, "JP", "XX"])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [250, 840, 276, None, 392, None])
        self.assertEqual(out.type, pa.int32())

    def test_country_alpha3_to_int(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("c", ISOCountryType(alpha=3))
        tgt = self._field("n", IntegerType(byte_size=2, signed=True))
        arr = pa.array(["FRA", "USA", "DEU", None])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [250, 840, 276, None])
        self.assertEqual(out.type, pa.int16())

    def test_country_alpha_crossover(self):
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("c3", ISOCountryType(alpha=3))
        arr = pa.array(["FR", "US", None, "DE"])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), ["FRA", "USA", None, "DEU"])

    def test_country_to_float(self):
        from yggdrasil.data.types import FloatingPointType
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("f", FloatingPointType(byte_size=8))
        arr = pa.array(["FR", "US", None])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [250.0, 840.0, None])

    def test_currency_to_int(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("ccy", ISOCurrencyType())
        tgt = self._field("n", IntegerType(byte_size=4, signed=True))
        arr = pa.array(["USD", "EUR", "JPY", None, "GBP", "NOTACCY"])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [840, 978, 392, None, 826, None])

    def test_timezone_to_duration_seconds(self):
        from yggdrasil.data.types import DurationType
        src = self._field("tz", TimezoneType())
        tgt = self._field("d", DurationType(byte_size=8, unit="s"))
        arr = pa.array(["UTC", "+05:30", "-08:00", None])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        import datetime as dt
        self.assertEqual(
            out.to_pylist(),
            [
                dt.timedelta(0),
                dt.timedelta(hours=5, minutes=30),
                dt.timedelta(hours=-8),
                None,
            ],
        )

    def test_timezone_to_int_seconds(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("tz", TimezoneType())
        tgt = self._field("n", IntegerType(byte_size=8, signed=True))
        arr = pa.array(["UTC", "+05:30", "-08:00", None])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [0, 19800, -28800, None])

    def test_outgoing_cast_preserves_chunks(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("n", IntegerType(byte_size=4, signed=True))
        arr = pa.chunked_array([["FR", "US"], [None, "DE"]])
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertIsInstance(out, pa.ChunkedArray)
        self.assertEqual(out.to_pylist(), [250, 840, None, 276])

    def test_outgoing_cast_empty(self):
        from yggdrasil.data.types import IntegerType
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("n", IntegerType(byte_size=4, signed=True))
        arr = pa.array([], type=pa.string())
        out = tgt.dtype.cast_arrow_array(arr, source_field=src, target_field=tgt)
        self.assertEqual(out.to_pylist(), [])
        self.assertEqual(out.type, pa.int32())


class TestISOCrossTypePolars(PolarsTestCase):
    def _field(self, name, dtype):
        from yggdrasil.data.data_field import Field
        return Field(name=name, dtype=dtype)

    def test_country_to_int(self):
        from yggdrasil.data.types import IntegerType
        pl = self.pl
        src = self._field("c", ISOCountryType(alpha=2))
        tgt = self._field("c", IntegerType(byte_size=4, signed=True))
        s = pl.Series("c", ["FR", "US", "DE", None])
        out = tgt.dtype.cast_polars_series(s, source_field=src, target_field=tgt)
        self.assertEqual(out.to_list(), [250, 840, 276, None])

    def test_currency_to_int(self):
        from yggdrasil.data.types import IntegerType
        pl = self.pl
        src = self._field("ccy", ISOCurrencyType())
        tgt = self._field("ccy", IntegerType(byte_size=4, signed=True))
        s = pl.Series("ccy", ["USD", "EUR", "JPY"])
        out = tgt.dtype.cast_polars_series(s, source_field=src, target_field=tgt)
        self.assertEqual(out.to_list(), [840, 978, 392])


class TestISOTypeFrameworks(ArrowTestCase):
    def test_to_arrow(self):
        for t in (
            ISOCountryType(),
            ISOCurrencyType(),
            ISOContinentType(),
            ISOSubdivisionType(),
            ISOCityType(),
            TimezoneType(),
        ):
            self.assertEqual(t.to_arrow(), pa.string())

    def test_to_databricks_ddl(self):
        for t in (ISOCountryType(), ISOCurrencyType(), TimezoneType()):
            self.assertEqual(t.to_databricks_ddl(), "STRING")

    def test_base_not_in_registry_standalone(self):
        # Basic smoke check: the base ISOType is abstract — exercising it
        # directly should fail.
        with self.assertRaises(NotImplementedError):
            ISOType()._resolve_token("X")


if __name__ == "__main__":
    unittest.main()
