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
# ISOType base — framework integration invariants
# ---------------------------------------------------------------------------


class TestISOTypeFrameworks(ArrowTestCase):
    def test_to_arrow(self):
        for t in (
            ISOCountryType(),
            ISOCurrencyType(),
            ISOContinentType(),
            ISOSubdivisionType(),
            ISOCityType(),
        ):
            self.assertEqual(t.to_arrow(), pa.string())

    def test_to_databricks_ddl(self):
        for t in (ISOCountryType(), ISOCurrencyType()):
            self.assertEqual(t.to_databricks_ddl(), "STRING")

    def test_base_not_in_registry_standalone(self):
        # Basic smoke check: the base ISOType is abstract — exercising it
        # directly should fail.
        with self.assertRaises(NotImplementedError):
            ISOType()._resolve_token("X")


if __name__ == "__main__":
    unittest.main()
