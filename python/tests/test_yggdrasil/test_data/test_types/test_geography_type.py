"""Tests for GeographyType — first-class data type for geographic zone data.

Covers:
- type_id = GEOGRAPHY
- SRID parameter (default, explicit, ANY, normalization)
- Databricks DDL: GEOGRAPHY(srid) syntax
- Arrow conversion (to_arrow → string, handles_arrow_type → False)
- Polars fallback to string
- Dict round-trip with srid
- Merge behavior
- Cast: Arrow array normalization with safe=True and safe=False
- Cast: handles various input formats (ISO codes, names, aliases, coords)
- Cast: non-string inputs get stringified first
- Cast: nulls pass through
- Bulk parse utilities: parse_geography_arrow, parse_geography_polars
- Output field selection: code, name, country_iso, region_iso, ccy, gtype
- Python object conversion
- Default values
- Repr / str
- DataType.from_str("geography"), "geography(4326)", "geography(ANY)"
"""

from __future__ import annotations

import unittest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.extensions.geography import (
    DEFAULT_SRID,
    GeographyType,
    parse_geography_arrow,
    parse_geography_polars,
)
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.polars.tests import PolarsTestCase

# ---------------------------------------------------------------------------
# type_id
# ---------------------------------------------------------------------------


class TestTypeId(unittest.TestCase):
    def test_type_id_is_geography(self):
        self.assertEqual(GeographyType().type_id, DataTypeId.GEOGRAPHY)

    def test_children_fields_empty(self):
        self.assertEqual(GeographyType().children_fields, [])


# ---------------------------------------------------------------------------
# SRID parameter
# ---------------------------------------------------------------------------


class TestSRID(unittest.TestCase):
    def test_default_srid(self):
        self.assertEqual(GeographyType().srid, DEFAULT_SRID)
        self.assertEqual(GeographyType().srid, 4326)

    def test_explicit_srid_int(self):
        self.assertEqual(GeographyType(srid=3857).srid, 3857)

    def test_srid_any(self):
        self.assertEqual(GeographyType(srid="ANY").srid, "ANY")

    def test_srid_any_case_insensitive(self):
        self.assertEqual(GeographyType(srid="any").srid, "ANY")
        self.assertEqual(GeographyType(srid="Any").srid, "ANY")

    def test_srid_string_numeric(self):
        self.assertEqual(GeographyType(srid="4326").srid, 4326)

    def test_srid_none_defaults(self):
        self.assertEqual(GeographyType(srid=None).srid, DEFAULT_SRID)

    def test_invalid_srid_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid SRID"):
            GeographyType(srid="BOGUS")

    def test_equality_same_srid(self):
        self.assertEqual(GeographyType(srid=4326), GeographyType(srid=4326))
        self.assertEqual(GeographyType(srid="ANY"), GeographyType(srid="ANY"))

    def test_equality_different_srid(self):
        self.assertNotEqual(GeographyType(srid=4326), GeographyType(srid=3857))
        self.assertNotEqual(GeographyType(srid=4326), GeographyType(srid="ANY"))


# ---------------------------------------------------------------------------
# Databricks DDL
# ---------------------------------------------------------------------------


class TestDatabricksDDL(unittest.TestCase):
    def test_ddl_default(self):
        self.assertEqual(GeographyType().to_databricks_ddl(), "GEOGRAPHY(4326)")

    def test_ddl_explicit_srid(self):
        self.assertEqual(
            GeographyType(srid=3857).to_databricks_ddl(), "GEOGRAPHY(3857)"
        )

    def test_ddl_any(self):
        self.assertEqual(
            GeographyType(srid="ANY").to_databricks_ddl(), "GEOGRAPHY(ANY)"
        )

    def test_ddl_output_does_not_affect_ddl(self):
        self.assertEqual(
            GeographyType(output="name").to_databricks_ddl(), "GEOGRAPHY(4326)"
        )


# ---------------------------------------------------------------------------
# Arrow conversion — plain string, no extension wrapping
# ---------------------------------------------------------------------------


class TestArrowConversion(ArrowTestCase):
    def test_to_arrow_is_string(self):
        pa = self.pa
        self.assertEqual(GeographyType().to_arrow(), pa.string())

    def test_handles_arrow_type_always_false(self):
        pa = self.pa
        self.assertFalse(GeographyType.handles_arrow_type(pa.string()))
        self.assertFalse(GeographyType.handles_arrow_type(pa.int64()))

    def test_from_arrow_type_raises(self):
        pa = self.pa
        with self.assertRaisesRegex(TypeError, "Cannot infer GeographyType"):
            GeographyType.from_arrow_type(pa.string())


# ---------------------------------------------------------------------------
# Polars fallback
# ---------------------------------------------------------------------------


class TestPolarsConversion(PolarsTestCase):
    def test_to_polars_is_string(self):
        self.assertEqual(GeographyType().to_polars(), self.pl.String)

    def test_handles_polars_type_always_false(self):
        self.assertFalse(GeographyType.handles_polars_type(self.pl.String))


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


class TestDictRoundTrip(unittest.TestCase):
    def test_to_dict_default(self):
        d = GeographyType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.GEOGRAPHY))
        self.assertEqual(d["name"], "GEOGRAPHY")
        self.assertNotIn("srid", d)
        self.assertNotIn("output", d)

    def test_to_dict_custom_srid(self):
        d = GeographyType(srid=3857).to_dict()
        self.assertEqual(d["srid"], "3857")

    def test_to_dict_srid_any(self):
        d = GeographyType(srid="ANY").to_dict()
        self.assertEqual(d["srid"], "ANY")

    def test_handles_dict_by_id(self):
        self.assertTrue(GeographyType.handles_dict({"id": int(DataTypeId.GEOGRAPHY)}))

    def test_handles_dict_by_name(self):
        self.assertTrue(GeographyType.handles_dict({"name": "GEOGRAPHY"}))
        self.assertTrue(GeographyType.handles_dict({"name": "geography"}))

    def test_handles_dict_false(self):
        self.assertFalse(GeographyType.handles_dict({"id": int(DataTypeId.STRING)}))

    def test_from_dict(self):
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "3857", "output": "name"}
        result = GeographyType.from_dict(d)
        self.assertEqual(result.srid, 3857)
        self.assertEqual(result.output, "name")

    def test_dict_round_trip(self):
        original = GeographyType(srid=3857, output="country_iso")
        restored = GeographyType.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_datatype_from_dict_dispatch(self):
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "ANY"}
        result = DataType.from_dict(d)
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, "ANY")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge(unittest.TestCase):
    def test_same_type_keeps_self(self):
        a = GeographyType()
        self.assertIs(a.merge_with(GeographyType()), a)

    def test_merge_with_null_returns_self(self):
        from yggdrasil.data.types.primitive import NullType

        a = GeographyType()
        self.assertIs(a.merge_with(NullType()), a)


# ---------------------------------------------------------------------------
# Casting — Arrow
# ---------------------------------------------------------------------------


class TestCastArrow(ArrowTestCase):
    def test_cast_resolves_iso_codes(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR", "DE", "CH-ZH"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.to_pylist()
        self.assertEqual(codes[0], "FR IDF")
        self.assertEqual(codes[1], "DE BE")
        self.assertEqual(codes[2], "CH ZH")

    def test_cast_resolves_names(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["France", "zuerich"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertEqual(result.to_pylist()[0], "FR IDF")
        self.assertEqual(result.to_pylist()[1], "CH ZH")

    def test_cast_safe_false_nulls_unresolvable(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR", "TOTALLY_BOGUS", "DE"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.to_pylist()
        self.assertIsNotNone(codes[0])
        self.assertIsNone(codes[1])
        self.assertIsNotNone(codes[2])

    def test_cast_safe_true_raises_on_unresolvable(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR", "TOTALLY_BOGUS"], type=pa.string())

        class _Opts:
            safe = True

        with self.assertRaisesRegex(ValueError, "Cannot resolve geography value"):
            geo._cast_arrow_array(arr, _Opts())

    def test_cast_preserves_nulls(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR", None, "DE"], type=pa.string())

        class _Opts:
            safe = False

        codes = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertIsNotNone(codes[0])
        self.assertIsNone(codes[1])
        self.assertIsNotNone(codes[2])

    def test_cast_with_output_name(self):
        pa = self.pa
        geo = GeographyType(output="name")
        arr = pa.array(["FR", "DE"], type=pa.string())

        class _Opts:
            safe = False

        names = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertEqual(names[0], "France")
        self.assertEqual(names[1], "Germany")

    def test_cast_with_output_ccy(self):
        pa = self.pa
        geo = GeographyType(output="ccy")
        arr = pa.array(["FR", "CH-ZH"], type=pa.string())

        class _Opts:
            safe = False

        ccys = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertEqual(ccys[0], "EUR")
        self.assertEqual(ccys[1], "CHF")

    def test_cast_from_int_array_safe_false(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array([42, 99], type=pa.int64())

        class _Opts:
            safe = False

        codes = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertIsNone(codes[0])
        self.assertIsNone(codes[1])

    def test_cast_empty_array(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array([], type=pa.string())

        class _Opts:
            safe = False

        self.assertEqual(len(geo._cast_arrow_array(arr, _Opts())), 0)

    def test_cast_coordinate_string(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["47.3769, 8.5417"], type=pa.string())

        class _Opts:
            safe = False

        self.assertEqual(geo._cast_arrow_array(arr, _Opts()).to_pylist()[0], "CH ZH")

    def test_result_is_plain_string_not_extension(self):
        """Cast result should be a plain string array, not an extension array."""
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertEqual(result.type, pa.string())
        self.assertNotIsInstance(result.type, pa.ExtensionType)


# ---------------------------------------------------------------------------
# Bulk parse utilities
# ---------------------------------------------------------------------------


class TestParseGeographyArrow(ArrowTestCase):
    def test_basic_parse(self):
        pa = self.pa
        arr = pa.array(["FR", "DE", None, "xyzzy_404"], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        codes = result.to_pylist()
        self.assertIsNotNone(codes[0])
        self.assertIsNotNone(codes[1])
        self.assertIsNone(codes[2])
        self.assertIsNone(codes[3])

    def test_output_name(self):
        pa = self.pa
        arr = pa.array(["FR", "CH-ZH"], type=pa.string())
        result = parse_geography_arrow(arr, output="name")
        self.assertEqual(result.to_pylist(), ["France", "Zurich"])

    def test_chunked_array(self):
        pa = self.pa
        arr = pa.chunked_array([["FR", "DE"], ["CH-ZH"]], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        self.assertIsInstance(result, pa.ChunkedArray)
        self.assertEqual(len(result.to_pylist()), 3)

    def test_safe_true_raises(self):
        pa = self.pa
        arr = pa.array(["FR", "BOGUS"], type=pa.string())
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            parse_geography_arrow(arr, safe=True)


class TestParseGeographyPolars(PolarsTestCase):
    def test_basic_parse(self):
        pl = self.pl
        s = pl.Series("zone", ["FR", "DE", None, "xyzzy_404"])
        result = parse_geography_polars(s, safe=False)
        self.assertEqual(result.name, "zone")
        self.assertEqual(result.dtype, pl.String)
        values = result.to_list()
        self.assertIsNotNone(values[0])
        self.assertIsNotNone(values[1])
        self.assertIsNone(values[2])
        self.assertIsNone(values[3])

    def test_output_name(self):
        pl = self.pl
        s = pl.Series("zone", ["FR", "CH-ZH"])
        result = parse_geography_polars(s, output="name")
        self.assertEqual(result.to_list(), ["France", "Zurich"])


# ---------------------------------------------------------------------------
# Python object conversion
# ---------------------------------------------------------------------------


class TestConvertPyobj(unittest.TestCase):
    def test_convert_valid(self):
        self.assertIsNotNone(GeographyType().convert_pyobj("France", nullable=True))

    def test_convert_none_nullable(self):
        self.assertIsNone(GeographyType().convert_pyobj(None, nullable=True))

    def test_convert_none_not_nullable_raises(self):
        with self.assertRaisesRegex(ValueError, "non-nullable"):
            GeographyType().convert_pyobj(None, nullable=False)

    def test_convert_unknown_safe_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            GeographyType().convert_pyobj(
                "TOTALLY_BOGUS_ZONE", nullable=True, safe=True
            )

    def test_convert_unknown_unsafe_returns_none(self):
        self.assertIsNone(
            GeographyType().convert_pyobj(
                "TOTALLY_BOGUS_ZONE", nullable=True, safe=False
            )
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults(unittest.TestCase):
    def test_default_pyobj_nullable(self):
        self.assertIsNone(GeographyType().default_pyobj(nullable=True))

    def test_default_pyobj_not_nullable(self):
        self.assertEqual(GeographyType().default_pyobj(nullable=False), "WORLD")

    def test_default_arrow_scalar_nullable(self):
        self.assertIsNone(GeographyType().default_arrow_scalar(nullable=True).as_py())

    def test_default_arrow_scalar_not_nullable(self):
        self.assertEqual(
            GeographyType().default_arrow_scalar(nullable=False).as_py(), "WORLD"
        )


# ---------------------------------------------------------------------------
# Repr / str
# ---------------------------------------------------------------------------


class TestRepr(unittest.TestCase):
    def test_repr_default(self):
        self.assertEqual(repr(GeographyType()), "GeographyType()")

    def test_repr_custom_srid(self):
        self.assertEqual(repr(GeographyType(srid=3857)), "GeographyType(srid=3857)")

    def test_repr_srid_any(self):
        self.assertEqual(repr(GeographyType(srid="ANY")), "GeographyType(srid='ANY')")

    def test_repr_custom_output(self):
        self.assertEqual(
            repr(GeographyType(output="name")), "GeographyType(output='name')"
        )

    def test_str_default(self):
        self.assertEqual(str(GeographyType()), "geography")

    def test_str_custom_srid(self):
        self.assertEqual(str(GeographyType(srid=3857)), "geography(3857)")

    def test_str_srid_any(self):
        self.assertEqual(str(GeographyType(srid="ANY")), "geography(ANY)")


# ---------------------------------------------------------------------------
# DataType.from_str
# ---------------------------------------------------------------------------


class TestFromStr(unittest.TestCase):
    def test_from_str_geography(self):
        result = DataType.from_str("geography")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 4326)

    def test_from_str_geography_srid(self):
        result = DataType.from_str("geography(4326)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 4326)

    def test_from_str_geography_srid_3857(self):
        result = DataType.from_str("geography(3857)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)

    def test_from_str_geography_any(self):
        result = DataType.from_str("geography(ANY)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, "ANY")

    def test_from_str_geo(self):
        self.assertIsInstance(DataType.from_str("geo"), GeographyType)

    def test_from_str_geozone(self):
        self.assertIsInstance(DataType.from_str("geozone"), GeographyType)

    def test_from_str_geolocation(self):
        self.assertIsInstance(DataType.from_str("geolocation"), GeographyType)

    def test_from_str_GEOGRAPHY_uppercase(self):
        result = DataType.from_str("GEOGRAPHY(4326)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 4326)


# ---------------------------------------------------------------------------
# DataType.from_any / from_dict
# ---------------------------------------------------------------------------


class TestFromAny(unittest.TestCase):
    def test_from_any_string(self):
        self.assertIsInstance(DataType.from_any("geography"), GeographyType)

    def test_from_any_string_with_srid(self):
        result = DataType.from_any("geography(3857)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)

    def test_from_dict_dispatch(self):
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "3857", "output": "name"}
        result = DataType.from_dict(d)
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)
        self.assertEqual(result.output, "name")
