"""Tests for GeographyType — extension type for geographic zone data.

Covers:
- Extension registration and type_id
- SRID parameter (default, explicit, ANY, normalization)
- Databricks DDL: GEOGRAPHY(srid) syntax
- Arrow conversion (to_arrow, from_arrow_type, handles_arrow_type)
- Arrow IPC round-trip
- Polars fallback to storage type (string)
- Dict round-trip with srid
- Serialization metadata round-trip with srid
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

import json
import unittest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.extensions.base import (
    ExtensionType,
    _EXTENSION_REGISTRY,
    get_extension_type,
)
from yggdrasil.data.types.extensions.geography import (
    DEFAULT_SRID,
    GeographyType,
    parse_geography_arrow,
    parse_geography_polars,
)
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.polars.tests import PolarsTestCase

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_auto_registered(self):
        self.assertIn("yggdrasil.geography", _EXTENSION_REGISTRY)

    def test_get_extension_type(self):
        cls = get_extension_type("yggdrasil.geography")
        self.assertIs(cls, GeographyType)

    def test_type_id(self):
        self.assertEqual(GeographyType().type_id, DataTypeId.EXTENSION)


# ---------------------------------------------------------------------------
# SRID parameter
# ---------------------------------------------------------------------------


class TestSRID(unittest.TestCase):
    def test_default_srid(self):
        self.assertEqual(GeographyType().srid, DEFAULT_SRID)
        self.assertEqual(GeographyType().srid, 4326)

    def test_explicit_srid_int(self):
        geo = GeographyType(srid=3857)
        self.assertEqual(geo.srid, 3857)

    def test_srid_any(self):
        geo = GeographyType(srid="ANY")
        self.assertEqual(geo.srid, "ANY")

    def test_srid_any_case_insensitive(self):
        self.assertEqual(GeographyType(srid="any").srid, "ANY")
        self.assertEqual(GeographyType(srid="Any").srid, "ANY")

    def test_srid_string_numeric(self):
        geo = GeographyType(srid="4326")
        self.assertEqual(geo.srid, 4326)

    def test_srid_none_defaults(self):
        geo = GeographyType(srid=None)
        self.assertEqual(geo.srid, DEFAULT_SRID)

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

    def test_ddl_with_output_still_geography(self):
        self.assertEqual(
            GeographyType(output="name").to_databricks_ddl(), "GEOGRAPHY(4326)"
        )


# ---------------------------------------------------------------------------
# Serialization metadata
# ---------------------------------------------------------------------------


class TestSerialization(unittest.TestCase):
    def test_serialize_default(self):
        # Default srid=4326, output="code" — empty metadata.
        self.assertEqual(GeographyType().serialize_metadata(), b"")

    def test_serialize_custom_srid(self):
        raw = GeographyType(srid=3857).serialize_metadata()
        payload = json.loads(raw)
        self.assertEqual(payload["srid"], "3857")

    def test_serialize_srid_any(self):
        raw = GeographyType(srid="ANY").serialize_metadata()
        payload = json.loads(raw)
        self.assertEqual(payload["srid"], "ANY")

    def test_serialize_custom_output(self):
        raw = GeographyType(output="name").serialize_metadata()
        payload = json.loads(raw)
        self.assertEqual(payload, {"output": "name"})

    def test_deserialize_empty(self):
        restored = GeographyType.deserialize_metadata(b"")
        self.assertEqual(restored, GeographyType())

    def test_deserialize_with_srid(self):
        raw = json.dumps({"srid": "3857"}).encode()
        restored = GeographyType.deserialize_metadata(raw)
        self.assertEqual(restored.srid, 3857)

    def test_deserialize_srid_any(self):
        raw = json.dumps({"srid": "ANY"}).encode()
        restored = GeographyType.deserialize_metadata(raw)
        self.assertEqual(restored.srid, "ANY")

    def test_round_trip_srid(self):
        original = GeographyType(srid=3857, output="country_iso")
        raw = original.serialize_metadata()
        restored = GeographyType.deserialize_metadata(raw)
        self.assertEqual(restored, original)

    def test_round_trip_any(self):
        original = GeographyType(srid="ANY")
        raw = original.serialize_metadata()
        restored = GeographyType.deserialize_metadata(raw)
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Arrow conversion
# ---------------------------------------------------------------------------


class TestArrowConversion(ArrowTestCase):
    def test_to_arrow_is_extension(self):
        pa = self.pa
        arrow_type = GeographyType().to_arrow()
        self.assertIsInstance(arrow_type, pa.ExtensionType)
        self.assertEqual(arrow_type.extension_name, "yggdrasil.geography")

    def test_storage_type_is_string(self):
        pa = self.pa
        arrow_type = GeographyType().to_arrow()
        self.assertEqual(arrow_type.storage_type, pa.string())

    def test_handles_arrow_type_true(self):
        arrow_type = GeographyType().to_arrow()
        self.assertTrue(GeographyType.handles_arrow_type(arrow_type))
        self.assertTrue(ExtensionType.handles_arrow_type(arrow_type))

    def test_handles_arrow_type_false_for_plain_string(self):
        pa = self.pa
        self.assertFalse(GeographyType.handles_arrow_type(pa.string()))

    def test_from_arrow_type_dispatch(self):
        arrow_type = GeographyType(srid=3857, output="name").to_arrow()
        restored = ExtensionType.from_arrow_type(arrow_type)
        self.assertIsInstance(restored, GeographyType)
        self.assertEqual(restored.srid, 3857)
        self.assertEqual(restored.output, "name")

    def test_arrow_round_trip_default(self):
        original = GeographyType()
        restored = ExtensionType.from_arrow_type(original.to_arrow())
        self.assertEqual(restored, original)

    def test_arrow_round_trip_with_srid_any(self):
        original = GeographyType(srid="ANY", output="region_iso")
        restored = ExtensionType.from_arrow_type(original.to_arrow())
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Arrow IPC round-trip
# ---------------------------------------------------------------------------


class TestArrowIPC(ArrowTestCase):
    def test_ipc_roundtrip(self):
        pa = self.pa
        geo = GeographyType(srid=3857, output="name")
        arrow_type = geo.to_arrow()
        storage = pa.array(["France", "Germany"], type=pa.string())
        arr = pa.ExtensionArray.from_storage(arrow_type, storage)
        table = pa.table({"zone": arr})

        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        reader = pa.ipc.open_stream(sink.getvalue())
        result = reader.read_all()

        col_type = result.column("zone").type
        self.assertIsInstance(col_type, pa.ExtensionType)
        self.assertEqual(col_type.extension_name, "yggdrasil.geography")
        raw = col_type.__arrow_ext_serialize__()
        payload = json.loads(raw)
        self.assertEqual(payload["srid"], "3857")
        self.assertEqual(payload["output"], "name")


# ---------------------------------------------------------------------------
# Polars fallback
# ---------------------------------------------------------------------------


class TestPolarsConversion(PolarsTestCase):
    def test_to_polars_falls_back_to_string(self):
        polars_type = GeographyType().to_polars()
        self.assertEqual(polars_type, self.pl.String)


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


class TestDictRoundTrip(unittest.TestCase):
    def test_to_dict_default(self):
        d = GeographyType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.EXTENSION))
        self.assertEqual(d["extension_name"], "yggdrasil.geography")
        self.assertNotIn("srid", d)
        self.assertNotIn("output", d)

    def test_to_dict_custom_srid(self):
        d = GeographyType(srid=3857).to_dict()
        self.assertEqual(d["srid"], "3857")

    def test_to_dict_srid_any(self):
        d = GeographyType(srid="ANY").to_dict()
        self.assertEqual(d["srid"], "ANY")

    def test_to_dict_custom_output(self):
        d = GeographyType(output="name").to_dict()
        self.assertEqual(d["output"], "name")

    def test_from_dict_dispatch(self):
        d = GeographyType(srid="ANY", output="ccy").to_dict()
        restored = ExtensionType.from_dict(d)
        self.assertIsInstance(restored, GeographyType)
        self.assertEqual(restored.srid, "ANY")
        self.assertEqual(restored.output, "ccy")

    def test_full_round_trip(self):
        original = GeographyType(srid=3857, output="country_iso")
        restored = ExtensionType.from_dict(original.to_dict())
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge(unittest.TestCase):
    def test_same_type_keeps_self(self):
        a = GeographyType()
        b = GeographyType()
        self.assertIs(a.merge_with(b), a)

    def test_merge_with_null_returns_self(self):
        from yggdrasil.data.types.primitive import NullType

        a = GeographyType()
        result = a.merge_with(NullType())
        self.assertIs(result, a)


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
        self.assertIsInstance(result.type, pa.ExtensionType)
        codes = result.storage.to_pylist()
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
        codes = result.storage.to_pylist()
        self.assertEqual(codes[0], "FR IDF")
        self.assertEqual(codes[1], "CH ZH")

    def test_cast_resolves_aliases(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FRA", "DEU", "EUROPE"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.storage.to_pylist()
        self.assertIsNotNone(codes[0])
        self.assertIsNotNone(codes[1])
        self.assertEqual(codes[2], "EU")

    def test_cast_safe_false_nulls_unresolvable(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["FR", "TOTALLY_BOGUS", "DE"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.storage.to_pylist()
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

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.storage.to_pylist()
        self.assertIsNotNone(codes[0])
        self.assertIsNone(codes[1])
        self.assertIsNotNone(codes[2])

    def test_cast_with_output_name(self):
        pa = self.pa
        geo = GeographyType(output="name")
        arr = pa.array(["FR", "DE"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        names = result.storage.to_pylist()
        self.assertEqual(names[0], "France")
        self.assertEqual(names[1], "Germany")

    def test_cast_with_output_country_iso(self):
        pa = self.pa
        geo = GeographyType(output="country_iso")
        arr = pa.array(["France", "zuerich"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        isos = result.storage.to_pylist()
        self.assertEqual(isos[0], "FR")
        self.assertEqual(isos[1], "CH")

    def test_cast_with_output_ccy(self):
        pa = self.pa
        geo = GeographyType(output="ccy")
        arr = pa.array(["FR", "CH-ZH"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        ccys = result.storage.to_pylist()
        self.assertEqual(ccys[0], "EUR")
        self.assertEqual(ccys[1], "CHF")

    def test_cast_from_int_array(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array([42, 99], type=pa.int64())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.storage.to_pylist()
        self.assertIsNone(codes[0])
        self.assertIsNone(codes[1])

    def test_cast_empty_array(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array([], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        self.assertEqual(len(result), 0)

    def test_cast_coordinate_string(self):
        pa = self.pa
        geo = GeographyType()
        arr = pa.array(["47.3769, 8.5417"], type=pa.string())

        class _Opts:
            safe = False

        result = geo._cast_arrow_array(arr, _Opts())
        codes = result.storage.to_pylist()
        self.assertEqual(codes[0], "CH ZH")


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
        names = result.to_pylist()
        self.assertEqual(names[0], "France")
        self.assertEqual(names[1], "Zurich")

    def test_chunked_array(self):
        pa = self.pa
        arr = pa.chunked_array([["FR", "DE"], ["CH-ZH"]], type=pa.string())
        result = parse_geography_arrow(arr, safe=False)
        self.assertIsInstance(result, pa.ChunkedArray)
        codes = result.to_pylist()
        self.assertEqual(len(codes), 3)
        self.assertIsNotNone(codes[0])

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
        geo = GeographyType()
        result = geo.convert_pyobj("France", nullable=True)
        self.assertIsNotNone(result)

    def test_convert_none_nullable(self):
        geo = GeographyType()
        self.assertIsNone(geo.convert_pyobj(None, nullable=True))

    def test_convert_none_not_nullable_raises(self):
        geo = GeographyType()
        with self.assertRaisesRegex(ValueError, "non-nullable"):
            geo.convert_pyobj(None, nullable=False)

    def test_convert_unknown_safe_raises(self):
        geo = GeographyType()
        with self.assertRaisesRegex(ValueError, "Cannot resolve"):
            geo.convert_pyobj("TOTALLY_BOGUS_ZONE", nullable=True, safe=True)

    def test_convert_unknown_unsafe_returns_none(self):
        geo = GeographyType()
        result = geo.convert_pyobj("TOTALLY_BOGUS_ZONE", nullable=True, safe=False)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults(unittest.TestCase):
    def test_default_pyobj_nullable(self):
        self.assertIsNone(GeographyType().default_pyobj(nullable=True))

    def test_default_pyobj_not_nullable(self):
        self.assertEqual(GeographyType().default_pyobj(nullable=False), "WORLD")

    def test_default_arrow_scalar_nullable(self):
        s = GeographyType().default_arrow_scalar(nullable=True)
        self.assertIsNone(s.as_py())

    def test_default_arrow_scalar_not_nullable(self):
        s = GeographyType().default_arrow_scalar(nullable=False)
        self.assertEqual(s.as_py(), "WORLD")


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

    def test_repr_both(self):
        self.assertEqual(
            repr(GeographyType(srid=3857, output="name")),
            "GeographyType(srid=3857, output='name')",
        )

    def test_str_default(self):
        self.assertEqual(str(GeographyType()), "geography")

    def test_str_custom_srid(self):
        self.assertEqual(str(GeographyType(srid=3857)), "geography(3857)")

    def test_str_srid_any(self):
        self.assertEqual(str(GeographyType(srid="ANY")), "geography(ANY)")

    def test_str_custom_output(self):
        self.assertEqual(str(GeographyType(output="ccy")), "geography(4326)[ccy]")


# ---------------------------------------------------------------------------
# DataType.from_str integration
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
        result = DataType.from_str("geo")
        self.assertIsInstance(result, GeographyType)

    def test_from_str_geozone(self):
        result = DataType.from_str("geozone")
        self.assertIsInstance(result, GeographyType)

    def test_from_str_geolocation(self):
        result = DataType.from_str("geolocation")
        self.assertIsInstance(result, GeographyType)

    def test_from_str_GEOGRAPHY_uppercase(self):
        result = DataType.from_str("GEOGRAPHY(4326)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 4326)


# ---------------------------------------------------------------------------
# DataType.from_any / from_dict integration
# ---------------------------------------------------------------------------


class TestFromAny(unittest.TestCase):
    def test_from_any_string(self):
        result = DataType.from_any("geography")
        self.assertIsInstance(result, GeographyType)

    def test_from_any_string_with_srid(self):
        result = DataType.from_any("geography(3857)")
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)

    def test_from_dict_dispatch(self):
        d = {
            "id": int(DataTypeId.EXTENSION),
            "extension_name": "yggdrasil.geography",
            "srid": "3857",
            "output": "name",
        }
        result = DataType.from_dict(d)
        self.assertIsInstance(result, GeographyType)
        self.assertEqual(result.srid, 3857)
        self.assertEqual(result.output, "name")
