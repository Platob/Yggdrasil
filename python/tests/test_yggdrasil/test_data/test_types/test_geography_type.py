"""Tests for GeographyType — first-class data type for geographic coordinates.

Inner type: struct<lat: float64 not null, lon: float64 not null>.

Covers:
- type_id = GEOGRAPHY, children_fields = [lat, lon]
- SRID parameter (default, explicit, ANY, OGC:CRS84, normalization)
- model parameter (None, SPHERICAL)
- Databricks DDL: GEOGRAPHY(srid) / GEOGRAPHY(srid, model)
- Arrow: to_arrow -> struct<lat, lon>
- Polars: to_polars -> Struct
- Dict round-trip
- Merge
- Cast: string arrays -> struct<lat, lon> via GeoZone catalog
- Cast: safe=True raises, safe=False nulls
- Cast: struct passthrough
- Bulk parse: parse_geography_arrow, parse_geography_polars
- Python object conversion -> {lat, lon} dicts
- Default values
- Repr / str
- DataType.from_str("geography"), "geography(4326)", etc.
"""

from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.extensions.geography import (
    DEFAULT_SRID,
    GEOGRAPHY_ARROW_TYPE,
    GeographyType,
    parse_geography_arrow,
    parse_geography_polars,
)
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.polars.tests import PolarsTestCase

# ---------------------------------------------------------------------------
# type_id / children_fields
# ---------------------------------------------------------------------------


class TestTypeId(unittest.TestCase):
    def test_type_id_is_geography(self):
        self.assertEqual(GeographyType().type_id, DataTypeId.GEOGRAPHY)

    def test_children_fields_lat_lon(self):
        fields = GeographyType().children_fields
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0].name, "lat")
        self.assertEqual(fields[1].name, "lon")


# ---------------------------------------------------------------------------
# SRID + model
# ---------------------------------------------------------------------------


class TestSRID(unittest.TestCase):
    def test_default_srid(self):
        self.assertEqual(GeographyType().srid, 4326)

    def test_explicit_srid_int(self):
        self.assertEqual(GeographyType(srid=3857).srid, 3857)

    def test_srid_any(self):
        self.assertEqual(GeographyType(srid="ANY").srid, "ANY")

    def test_srid_any_case_insensitive(self):
        self.assertEqual(GeographyType(srid="any").srid, "ANY")

    def test_srid_string_numeric(self):
        self.assertEqual(GeographyType(srid="4326").srid, 4326)

    def test_srid_none_defaults(self):
        self.assertEqual(GeographyType(srid=None).srid, DEFAULT_SRID)

    def test_srid_named_crs(self):
        self.assertEqual(GeographyType(srid="OGC:CRS84").srid, "OGC:CRS84")

    def test_srid_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid SRID"):
            GeographyType(srid="")

    def test_equality(self):
        self.assertEqual(GeographyType(srid=4326), GeographyType(srid=4326))
        self.assertNotEqual(GeographyType(srid=4326), GeographyType(srid=3857))

    def test_model_default_none(self):
        self.assertIsNone(GeographyType().model)

    def test_model_spherical(self):
        geo = GeographyType(srid="OGC:CRS84", model="SPHERICAL")
        self.assertEqual(geo.model, "SPHERICAL")

    def test_model_normalized_uppercase(self):
        self.assertEqual(GeographyType(model="spherical").model, "SPHERICAL")


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

    def test_ddl_ogc_crs84_spherical(self):
        geo = GeographyType(srid="OGC:CRS84", model="SPHERICAL")
        self.assertEqual(geo.to_databricks_ddl(), "GEOGRAPHY(OGC:CRS84, SPHERICAL)")

    def test_ddl_with_model_only(self):
        geo = GeographyType(model="SPHERICAL")
        self.assertEqual(geo.to_databricks_ddl(), "GEOGRAPHY(4326, SPHERICAL)")


# ---------------------------------------------------------------------------
# Arrow — struct<lat: float64, lon: float64>
# ---------------------------------------------------------------------------


class TestArrowConversion(ArrowTestCase):
    def test_to_arrow_is_struct(self):
        arrow_type = GeographyType().to_arrow()
        self.assertTrue(pa.types.is_struct(arrow_type))
        self.assertEqual(arrow_type.num_fields, 2)
        self.assertEqual(arrow_type.field("lat").type, pa.float64())
        self.assertEqual(arrow_type.field("lon").type, pa.float64())

    def test_to_arrow_matches_constant(self):
        self.assertEqual(GeographyType().to_arrow(), GEOGRAPHY_ARROW_TYPE)

    def test_handles_arrow_type_false(self):
        self.assertFalse(GeographyType.handles_arrow_type(GEOGRAPHY_ARROW_TYPE))
        self.assertFalse(GeographyType.handles_arrow_type(pa.string()))

    def test_from_arrow_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Cannot infer GeographyType"):
            GeographyType.from_arrow_type(GEOGRAPHY_ARROW_TYPE)


# ---------------------------------------------------------------------------
# Polars — Struct
# ---------------------------------------------------------------------------


class TestPolarsConversion(PolarsTestCase):
    def test_to_polars_is_struct(self):
        pl = self.pl
        polars_type = GeographyType().to_polars()
        self.assertIsInstance(polars_type, pl.Struct)


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


class TestDictRoundTrip(unittest.TestCase):
    def test_to_dict_default(self):
        d = GeographyType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.GEOGRAPHY))
        self.assertEqual(d["name"], "GEOGRAPHY")
        self.assertNotIn("srid", d)
        self.assertNotIn("model", d)

    def test_to_dict_custom_srid(self):
        self.assertEqual(GeographyType(srid=3857).to_dict()["srid"], "3857")

    def test_handles_dict_by_id(self):
        self.assertTrue(GeographyType.handles_dict({"id": int(DataTypeId.GEOGRAPHY)}))

    def test_handles_dict_by_name(self):
        self.assertTrue(GeographyType.handles_dict({"name": "GEOGRAPHY"}))

    def test_from_dict(self):
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "3857", "model": "SPHERICAL"}
        result = GeographyType.from_dict(d)
        self.assertEqual(result.srid, 3857)
        self.assertEqual(result.model, "SPHERICAL")

    def test_dict_round_trip(self):
        original = GeographyType(srid=3857, model="SPHERICAL")
        self.assertEqual(GeographyType.from_dict(original.to_dict()), original)

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

    def test_merge_with_null(self):
        from yggdrasil.data.types.primitive import NullType

        a = GeographyType()
        self.assertIs(a.merge_with(NullType()), a)


# ---------------------------------------------------------------------------
# Cast — string arrays to struct<lat, lon>
# ---------------------------------------------------------------------------


class TestCastArrow(ArrowTestCase):
    def test_cast_resolves_to_lat_lon(self):
        geo = GeographyType()
        arr = pa.array(["FR", "CH-ZH"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertTrue(pa.types.is_struct(result.type))
        rows = result.to_pylist()
        # France
        self.assertAlmostEqual(rows[0]["lat"], 46.2276, places=2)
        self.assertAlmostEqual(rows[0]["lon"], 2.2137, places=2)
        # Zurich
        self.assertAlmostEqual(rows[1]["lat"], 47.3769, places=2)
        self.assertAlmostEqual(rows[1]["lon"], 8.5417, places=2)

    def test_cast_safe_false_nulls_unresolvable(self):
        geo = GeographyType()
        arr = pa.array(["FR", "TOTALLY_BOGUS", "DE"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        rows = result.to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertIsNotNone(rows[2])

    def test_cast_safe_true_raises(self):
        geo = GeographyType()
        arr = pa.array(["FR", "TOTALLY_BOGUS"], type=pa.string())

        class _Opts:
            safe = True

        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            geo._cast_arrow_array(arr, _Opts())

    def test_cast_preserves_nulls(self):
        geo = GeographyType()
        arr = pa.array(["FR", None, "DE"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertIsNotNone(rows[2])

    def test_cast_struct_passthrough(self):
        """If input is already struct<lat, lon>, pass through unchanged."""
        geo = GeographyType()
        struct_arr = pa.array(
            [{"lat": 1.0, "lon": 2.0}],
            type=GEOGRAPHY_ARROW_TYPE,
        )

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(struct_arr, _Opts())
        self.assertIs(result, struct_arr)

    def test_cast_coordinate_string(self):
        geo = GeographyType()
        arr = pa.array(["47.3769, 8.5417"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], 47.3769, places=2)
        self.assertAlmostEqual(rows[0]["lon"], 8.5417, places=2)

    def test_cast_empty_array(self):
        geo = GeographyType()
        arr = pa.array([], type=pa.string())

        class _Opts:
            safe = False

        self.assertEqual(len(geo._cast_arrow_array(arr, _Opts())), 0)

    def test_result_is_struct_type(self):
        geo = GeographyType()
        arr = pa.array(["FR"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertTrue(pa.types.is_struct(result.type))
        self.assertEqual(result.type.num_fields, 2)


# ---------------------------------------------------------------------------
# Bulk parse
# ---------------------------------------------------------------------------


class TestParseGeographyArrow(ArrowTestCase):
    def test_basic_parse(self):
        arr = pa.array(["FR", "DE", None, "xyzzy_404"], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        rows = result.to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNotNone(rows[1])
        self.assertIsNone(rows[2])
        self.assertIsNone(rows[3])

    def test_result_has_lat_lon(self):
        arr = pa.array(["FR"], type=pa.string())
        result = parse_geography_arrow(arr)
        self.assertTrue(pa.types.is_struct(result.type))
        row = result.to_pylist()[0]
        self.assertIn("lat", row)
        self.assertIn("lon", row)

    def test_chunked_array(self):
        arr = pa.chunked_array([["FR", "DE"], ["CH-ZH"]], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        self.assertIsInstance(result, pa.ChunkedArray)
        self.assertEqual(len(result.to_pylist()), 3)

    def test_safe_true_raises(self):
        arr = pa.array(["FR", "BOGUS"], type=pa.string())
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            parse_geography_arrow(arr, safe=True)


class TestParseGeographyPolars(PolarsTestCase):
    def test_basic_parse(self):
        pl = self.pl
        s = pl.Series("zone", ["FR", "DE", None, "xyzzy_404"])
        result = parse_geography_polars(s, safe=False)
        self.assertEqual(result.name, "zone")
        self.assertIsInstance(result.dtype, pl.Struct)
        values = result.to_list()
        self.assertIsNotNone(values[0])
        self.assertIsNotNone(values[1])
        self.assertIsNone(values[2])
        self.assertIsNone(values[3])


# ---------------------------------------------------------------------------
# Python object conversion
# ---------------------------------------------------------------------------


class TestConvertPyobj(unittest.TestCase):
    def test_convert_string_returns_dict(self):
        result = GeographyType().convert_pyobj("France", nullable=True)
        self.assertIsInstance(result, dict)
        self.assertIn("lat", result)
        self.assertIn("lon", result)

    def test_convert_dict_passthrough(self):
        result = GeographyType().convert_pyobj({"lat": 1.0, "lon": 2.0}, nullable=True)
        self.assertEqual(result, {"lat": 1.0, "lon": 2.0})

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
        result = GeographyType().default_pyobj(nullable=False)
        self.assertEqual(result, {"lat": 0.0, "lon": 0.0})

    def test_default_arrow_scalar_nullable(self):
        self.assertIsNone(GeographyType().default_arrow_scalar(nullable=True).as_py())

    def test_default_arrow_scalar_not_nullable(self):
        s = GeographyType().default_arrow_scalar(nullable=False)
        self.assertEqual(s.as_py(), {"lat": 0.0, "lon": 0.0})


# ---------------------------------------------------------------------------
# Repr / str
# ---------------------------------------------------------------------------


class TestRepr(unittest.TestCase):
    def test_repr_default(self):
        self.assertEqual(repr(GeographyType()), "GeographyType()")

    def test_repr_custom_srid(self):
        self.assertEqual(repr(GeographyType(srid=3857)), "GeographyType(srid=3857)")

    def test_repr_with_model(self):
        self.assertEqual(
            repr(GeographyType(srid="OGC:CRS84", model="SPHERICAL")),
            "GeographyType(srid='OGC:CRS84', model='SPHERICAL')",
        )

    def test_str_default(self):
        self.assertEqual(str(GeographyType()), "geography")

    def test_str_custom_srid(self):
        self.assertEqual(str(GeographyType(srid=3857)), "geography(3857)")

    def test_str_with_model(self):
        self.assertEqual(
            str(GeographyType(srid="OGC:CRS84", model="SPHERICAL")),
            "geography(OGC:CRS84, SPHERICAL)",
        )


# ---------------------------------------------------------------------------
# DataType.from_str
# ---------------------------------------------------------------------------


class TestFromStr(unittest.TestCase):
    def test_from_str_geography(self):
        result = DataType.from_str("geography")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 4326)

    def test_from_str_geography_srid(self):
        result = DataType.from_str("geography(3857)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)

    def test_from_str_geography_any(self):
        result = DataType.from_str("geography(ANY)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, "ANY")

    def test_from_str_ogc_crs84_spherical(self):
        result = DataType.from_str("geography(OGC:CRS84, SPHERICAL)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, "OGC:CRS84")
        self.assertEqual(result.model, "SPHERICAL")

    def test_from_str_geo(self):
        self.assertIsInstance(DataType.from_str("geo"), GeographyType)

    def test_from_str_GEOGRAPHY_uppercase(self):
        result = DataType.from_str("GEOGRAPHY(4326)")
        self.assertIsInstance(result, GeographyType)


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
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "3857", "model": "SPHERICAL"}
        result = DataType.from_dict(d)
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)
        self.assertEqual(result.model, "SPHERICAL")
