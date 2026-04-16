"""Tests for GeographyType — first-class data type for lat/lon coordinates.

Inner type: struct<lat: float64, lon: float64>.
No GeoZone dependency — pure coordinate parsing and casting.
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
    def test_type_id(self):
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

    def test_explicit_srid(self):
        self.assertEqual(GeographyType(srid=3857).srid, 3857)

    def test_srid_any(self):
        self.assertEqual(GeographyType(srid="ANY").srid, "ANY")

    def test_srid_named_crs(self):
        self.assertEqual(GeographyType(srid="OGC:CRS84").srid, "OGC:CRS84")

    def test_srid_string_numeric(self):
        self.assertEqual(GeographyType(srid="4326").srid, 4326)

    def test_srid_none_defaults(self):
        self.assertEqual(GeographyType(srid=None).srid, DEFAULT_SRID)

    def test_srid_empty_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid SRID"):
            GeographyType(srid="")

    def test_model_default_none(self):
        self.assertIsNone(GeographyType().model)

    def test_model_spherical(self):
        self.assertEqual(GeographyType(model="SPHERICAL").model, "SPHERICAL")

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
        self.assertEqual(
            GeographyType(model="SPHERICAL").to_databricks_ddl(),
            "GEOGRAPHY(4326, SPHERICAL)",
        )


# ---------------------------------------------------------------------------
# Arrow — struct<lat, lon>
# ---------------------------------------------------------------------------


class TestArrowConversion(ArrowTestCase):
    def test_to_arrow_is_struct(self):
        t = GeographyType().to_arrow()
        self.assertTrue(pa.types.is_struct(t))
        self.assertEqual(t.field("lat").type, pa.float64())
        self.assertEqual(t.field("lon").type, pa.float64())

    def test_to_arrow_matches_constant(self):
        self.assertEqual(GeographyType().to_arrow(), GEOGRAPHY_ARROW_TYPE)

    def test_handles_arrow_type_false(self):
        self.assertFalse(GeographyType.handles_arrow_type(GEOGRAPHY_ARROW_TYPE))


# ---------------------------------------------------------------------------
# Polars — Struct
# ---------------------------------------------------------------------------


class TestPolarsConversion(PolarsTestCase):
    def test_to_polars_is_struct(self):
        self.assertIsInstance(GeographyType().to_polars(), self.pl.Struct)


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


class TestDict(unittest.TestCase):
    def test_to_dict_default(self):
        d = GeographyType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.GEOGRAPHY))
        self.assertEqual(d["name"], "GEOGRAPHY")
        self.assertNotIn("srid", d)

    def test_to_dict_custom_srid(self):
        self.assertEqual(GeographyType(srid=3857).to_dict()["srid"], "3857")

    def test_handles_dict_by_id(self):
        self.assertTrue(GeographyType.handles_dict({"id": int(DataTypeId.GEOGRAPHY)}))

    def test_handles_dict_by_name(self):
        self.assertTrue(GeographyType.handles_dict({"name": "GEOGRAPHY"}))

    def test_dict_round_trip(self):
        original = GeographyType(srid=3857, model="SPHERICAL")
        self.assertEqual(GeographyType.from_dict(original.to_dict()), original)

    def test_datatype_from_dict_dispatch(self):
        d = {"id": int(DataTypeId.GEOGRAPHY)}
        self.assertIsInstance(DataType.from_dict(d), GeographyType)


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
# Cast — string arrays → struct<lat, lon>
# ---------------------------------------------------------------------------


class TestCastStrings(ArrowTestCase):
    def test_comma_separated(self):
        geo = GeographyType()
        arr = pa.array(["48.8566, 2.3522", "47.3769, 8.5417"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], 48.8566, places=3)
        self.assertAlmostEqual(rows[0]["lon"], 2.3522, places=3)
        self.assertAlmostEqual(rows[1]["lat"], 47.3769, places=3)

    def test_space_separated(self):
        geo = GeographyType()
        arr = pa.array(["48.8566 2.3522"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], 48.8566, places=3)

    def test_pipe_separated(self):
        geo = GeographyType()
        arr = pa.array(["48.8566|2.3522"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], 48.8566, places=3)

    def test_semicolon_separated(self):
        geo = GeographyType()
        arr = pa.array(["48.8566;2.3522"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], 48.8566, places=3)

    def test_negative_coords(self):
        geo = GeographyType()
        arr = pa.array(["-33.8688, 151.2093"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertAlmostEqual(rows[0]["lat"], -33.8688, places=3)
        self.assertAlmostEqual(rows[0]["lon"], 151.2093, places=3)

    def test_safe_false_nulls_bad_strings(self):
        geo = GeographyType()
        arr = pa.array(["48.8, 2.3", "not coords", None], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertIsNone(rows[2])

    def test_safe_true_raises_on_bad_string(self):
        geo = GeographyType()
        arr = pa.array(["not coords"], type=pa.string())

        class _Opts:
            safe = True

        with self.assertRaisesRegex(ValueError, "Cannot parse coordinate"):
            geo._cast_arrow_array(arr, _Opts())

    def test_empty_array(self):
        geo = GeographyType()
        arr = pa.array([], type=pa.string())

        class _Opts:
            safe = False

        self.assertEqual(len(geo._cast_arrow_array(arr, _Opts())), 0)

    def test_out_of_range_coords_become_null(self):
        geo = GeographyType()
        # lat > 90 is invalid
        arr = pa.array(["999.0, 999.0"], type=pa.string())

        class _Opts:
            safe = False

        rows = geo._cast_arrow_array(arr, _Opts()).to_pylist()
        self.assertIsNone(rows[0])


# ---------------------------------------------------------------------------
# Cast — struct passthrough and rename
# ---------------------------------------------------------------------------


class TestCastStruct(ArrowTestCase):
    def test_struct_passthrough(self):
        geo = GeographyType()
        arr = pa.array(
            [{"lat": 48.8, "lon": 2.3}],
            type=GEOGRAPHY_ARROW_TYPE,
        )

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertIs(result, arr)

    def test_struct_rename_latitude_longitude(self):
        geo = GeographyType()
        arr = pa.array(
            [{"latitude": 48.8, "longitude": 2.3}],
            type=pa.struct(
                [
                    pa.field("latitude", pa.float64()),
                    pa.field("longitude", pa.float64()),
                ]
            ),
        )

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        row = result.to_pylist()[0]
        self.assertIn("lat", row)
        self.assertIn("lon", row)
        self.assertAlmostEqual(row["lat"], 48.8, places=1)

    def test_struct_rename_lng(self):
        geo = GeographyType()
        arr = pa.array(
            [{"lat": 48.8, "lng": 2.3}],
            type=pa.struct(
                [
                    pa.field("lat", pa.float64()),
                    pa.field("lng", pa.float64()),
                ]
            ),
        )

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        row = result.to_pylist()[0]
        self.assertEqual(row["lat"], 48.8)
        self.assertEqual(row["lon"], 2.3)


# ---------------------------------------------------------------------------
# Bulk parse utilities
# ---------------------------------------------------------------------------


class TestParseArrow(ArrowTestCase):
    def test_basic_parse(self):
        arr = pa.array(["48.8, 2.3", None, "bad"], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        rows = result.to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertIsNone(rows[2])

    def test_chunked_array(self):
        arr = pa.chunked_array(
            [["48.8, 2.3"], ["47.3, 8.5"]],
            type=pa.string(),
        )
        result = parse_geography_arrow(arr, safe=False)
        self.assertIsInstance(result, pa.ChunkedArray)
        self.assertEqual(len(result.to_pylist()), 2)


class TestParsePolars(PolarsTestCase):
    def test_basic_parse(self):
        pl = self.pl
        s = pl.Series("coords", ["48.8, 2.3", None, "bad"])
        result = parse_geography_polars(s, safe=False)
        self.assertEqual(result.name, "coords")
        self.assertIsInstance(result.dtype, pl.Struct)
        values = result.to_list()
        self.assertIsNotNone(values[0])
        self.assertIsNone(values[1])
        self.assertIsNone(values[2])


# ---------------------------------------------------------------------------
# Python object conversion
# ---------------------------------------------------------------------------


class TestConvertPyobj(unittest.TestCase):
    def test_dict_with_lat_lon(self):
        result = GeographyType().convert_pyobj({"lat": 48.8, "lon": 2.3}, nullable=True)
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_dict_with_latitude_longitude(self):
        result = GeographyType().convert_pyobj(
            {"latitude": 48.8, "longitude": 2.3},
            nullable=True,
        )
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_dict_with_lng(self):
        result = GeographyType().convert_pyobj({"lat": 48.8, "lng": 2.3}, nullable=True)
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_tuple(self):
        result = GeographyType().convert_pyobj((48.8, 2.3), nullable=True)
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_list(self):
        result = GeographyType().convert_pyobj([48.8, 2.3], nullable=True)
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_string(self):
        result = GeographyType().convert_pyobj("48.8, 2.3", nullable=True)
        self.assertAlmostEqual(result["lat"], 48.8, places=1)
        self.assertAlmostEqual(result["lon"], 2.3, places=1)

    def test_object_with_attrs(self):
        class Point:
            lat = 48.8
            lon = 2.3

        result = GeographyType().convert_pyobj(Point(), nullable=True)
        self.assertEqual(result, {"lat": 48.8, "lon": 2.3})

    def test_none_nullable(self):
        self.assertIsNone(GeographyType().convert_pyobj(None, nullable=True))

    def test_none_not_nullable_raises(self):
        with self.assertRaisesRegex(ValueError, "non-nullable"):
            GeographyType().convert_pyobj(None, nullable=False)

    def test_bad_value_safe_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            GeographyType().convert_pyobj("garbage", nullable=True, safe=True)

    def test_bad_value_unsafe_returns_none(self):
        self.assertIsNone(
            GeographyType().convert_pyobj("garbage", nullable=True, safe=False)
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults(unittest.TestCase):
    def test_default_pyobj_nullable(self):
        self.assertIsNone(GeographyType().default_pyobj(nullable=True))

    def test_default_pyobj_not_nullable(self):
        self.assertEqual(
            GeographyType().default_pyobj(nullable=False), {"lat": 0.0, "lon": 0.0}
        )

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
        self.assertIsInstance(DataType.from_str("geography"), GeographyType)

    def test_from_str_geography_srid(self):
        result = DataType.from_str("geography(3857)")
        self.assertEqual(result.srid, 3857)

    def test_from_str_geography_any(self):
        self.assertEqual(DataType.from_str("geography(ANY)").srid, "ANY")

    def test_from_str_ogc_crs84_spherical(self):
        result = DataType.from_str("geography(OGC:CRS84, SPHERICAL)")
        self.assertEqual(result.srid, "OGC:CRS84")
        self.assertEqual(result.model, "SPHERICAL")

    def test_from_str_geo(self):
        self.assertIsInstance(DataType.from_str("geo"), GeographyType)

    def test_from_str_GEOGRAPHY_uppercase(self):
        self.assertIsInstance(DataType.from_str("GEOGRAPHY(4326)"), GeographyType)


# ---------------------------------------------------------------------------
# DataType.from_any
# ---------------------------------------------------------------------------


class TestFromAny(unittest.TestCase):
    def test_from_any_string(self):
        self.assertIsInstance(DataType.from_any("geography"), GeographyType)

    def test_from_dict_dispatch(self):
        d = {"id": int(DataTypeId.GEOGRAPHY), "srid": "3857"}
        result = DataType.from_dict(d)
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)
