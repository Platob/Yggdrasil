"""Tests for ObjectType -- the opaque/variant Python object type.

Covers:
- type_id / children_fields / repr / str
- Arrow conversion (to_arrow, handles_arrow_type, from_arrow_type error)
- Polars conversion (to_polars, from_polars_type, handles_polars_type)
- Spark conversion (to_spark, handles_spark_type, from_spark_type error)
- Databricks DDL
- Dict round-trip (to_dict, from_dict, handles_dict)
- Merge behavior
- Default values (nullable and non-nullable)
- Cast bypass — ObjectType returns input unchanged
- DataType.from_parsed integration for OBJECT
- DataType.from_pytype integration for `object`
- DataType.from_str integration for "object"
- DataType.from_dict integration
"""

from __future__ import annotations

import unittest
from typing import Any

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.primitive import NullType, ObjectType, StringType
from yggdrasil.polars.tests import PolarsTestCase

# ---------------------------------------------------------------------------
# Basic properties (pure logic, no engine dependency)
# ---------------------------------------------------------------------------


class TestObjectTypeBasics(unittest.TestCase):

    def test_type_id(self):
        self.assertEqual(ObjectType().type_id, DataTypeId.OBJECT)

    def test_children_fields_empty(self):
        self.assertEqual(ObjectType().children_fields, [])

    def test_repr(self):
        self.assertEqual(repr(ObjectType()), "ObjectType()")

    def test_str(self):
        self.assertEqual(str(ObjectType()), "object")

    def test_frozen_dataclass(self):
        obj = ObjectType()
        with self.assertRaises(AttributeError):
            obj.foo = "bar"

    def test_equality(self):
        self.assertEqual(ObjectType(), ObjectType())

    def test_hash(self):
        self.assertEqual(hash(ObjectType()), hash(ObjectType()))


# ---------------------------------------------------------------------------
# Arrow conversion
# ---------------------------------------------------------------------------


class TestObjectTypeArrow(ArrowTestCase):

    def test_to_arrow(self):
        pa = self.pa
        arrow_type = ObjectType().to_arrow()
        self.assertEqual(arrow_type, pa.large_binary())

    def test_handles_arrow_type_always_false(self):
        pa = self.pa
        # ObjectType can't be inferred from any Arrow type.
        self.assertFalse(ObjectType.handles_arrow_type(pa.large_binary()))
        self.assertFalse(ObjectType.handles_arrow_type(pa.string()))
        self.assertFalse(ObjectType.handles_arrow_type(pa.int64()))
        self.assertFalse(ObjectType.handles_arrow_type(pa.null()))

    def test_from_arrow_type_raises(self):
        pa = self.pa
        with self.assertRaisesRegex(TypeError, "Cannot infer ObjectType"):
            ObjectType.from_arrow_type(pa.large_binary())

    def test_default_arrow_scalar_nullable(self):
        s = ObjectType().default_arrow_scalar(nullable=True)
        self.assertIsNone(s.as_py())
        self.assertEqual(s.type, pa.large_binary())

    def test_default_arrow_scalar_not_nullable_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "nullable=False"):
            ObjectType().default_arrow_scalar(nullable=False)


# ---------------------------------------------------------------------------
# Polars conversion
# ---------------------------------------------------------------------------


class TestObjectTypePolars(PolarsTestCase):

    def test_to_polars(self):
        pl = self.pl
        polars_type = ObjectType().to_polars()
        self.assertEqual(polars_type, pl.Object)

    def test_handles_polars_type_true(self):
        pl = self.pl
        self.assertTrue(ObjectType.handles_polars_type(pl.Object))

    def test_handles_polars_type_false(self):
        pl = self.pl
        self.assertFalse(ObjectType.handles_polars_type(pl.String))
        self.assertFalse(ObjectType.handles_polars_type(pl.Int64))

    def test_from_polars_type_object(self):
        pl = self.pl
        result = ObjectType.from_polars_type(pl.Object)
        self.assertIsInstance(result, ObjectType)

    def test_from_polars_type_wrong_type_raises(self):
        pl = self.pl
        with self.assertRaisesRegex(TypeError, "Expected Polars Object dtype"):
            ObjectType.from_polars_type(pl.String)

    def test_datatype_from_polars_type_dispatch(self):
        pl = self.pl
        result = DataType.from_polars_type(pl.Object)
        self.assertIsInstance(result, ObjectType)


# ---------------------------------------------------------------------------
# Spark conversion (pure logic — no Spark dependency needed)
# ---------------------------------------------------------------------------


class TestObjectTypeSpark(unittest.TestCase):

    def test_handles_spark_type_always_false(self):
        # Spark has no native object type.
        self.assertFalse(ObjectType.handles_spark_type(object()))

    def test_from_spark_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Cannot infer ObjectType"):
            ObjectType.from_spark_type(object())


# ---------------------------------------------------------------------------
# Databricks DDL (pure logic)
# ---------------------------------------------------------------------------


class TestObjectTypeDDL(unittest.TestCase):

    def test_to_databricks_ddl(self):
        self.assertEqual(ObjectType().to_databricks_ddl(), "BINARY")


# ---------------------------------------------------------------------------
# Dict round-trip (pure logic)
# ---------------------------------------------------------------------------


class TestObjectTypeDict(unittest.TestCase):

    def test_to_dict(self):
        d = ObjectType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.OBJECT))
        self.assertEqual(d["name"], "OBJECT")

    def test_handles_dict_by_id(self):
        self.assertTrue(ObjectType.handles_dict({"id": int(DataTypeId.OBJECT)}))

    def test_handles_dict_by_name(self):
        self.assertTrue(ObjectType.handles_dict({"name": "OBJECT"}))
        self.assertTrue(ObjectType.handles_dict({"name": "object"}))

    def test_handles_dict_false(self):
        self.assertFalse(ObjectType.handles_dict({"id": int(DataTypeId.STRING)}))
        self.assertFalse(ObjectType.handles_dict({"name": "STRING"}))

    def test_from_dict(self):
        d = {"id": int(DataTypeId.OBJECT), "name": "OBJECT"}
        result = ObjectType.from_dict(d)
        self.assertIsInstance(result, ObjectType)

    def test_dict_round_trip(self):
        original = ObjectType()
        restored = ObjectType.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_datatype_from_dict_dispatch(self):
        d = {"id": int(DataTypeId.OBJECT), "name": "OBJECT"}
        result = DataType.from_dict(d)
        self.assertIsInstance(result, ObjectType)


# ---------------------------------------------------------------------------
# Merge (pure logic)
# ---------------------------------------------------------------------------


class TestObjectTypeMerge(unittest.TestCase):

    def test_merge_same_type(self):
        a = ObjectType()
        b = ObjectType()
        result = a.merge_with(b)
        self.assertIs(result, a)

    def test_merge_with_null_returns_self(self):
        a = ObjectType()
        result = a.merge_with(NullType())
        self.assertIs(result, a)

    def test_merge_null_with_object_returns_object(self):
        result = NullType().merge_with(ObjectType())
        self.assertIsInstance(result, ObjectType)

    def test_merge_with_different_type_keeps_self(self):
        a = ObjectType()
        b = StringType()
        result = a.merge_with(b)
        # Different type_id, so _merge_with_different_id applies.
        # OBJECT=0 < STRING=11, so with no upcast/downcast, returns self.
        self.assertIs(result, a)


# ---------------------------------------------------------------------------
# Cast bypass — ObjectType returns input unchanged
# ---------------------------------------------------------------------------


class TestObjectTypeCastBypassArrow(ArrowTestCase):

    def test_cast_arrow_array_returns_input_unchanged(self):
        pa = self.pa
        obj = ObjectType()
        arr = pa.array([1, 2, 3], type=pa.int32())

        class _Opts:
            safe = True

        result = obj._cast_arrow_array(arr, _Opts())
        # Should be the exact same object — no copy, no cast.
        self.assertIs(result, arr)

    def test_cast_chunked_array_returns_input_unchanged(self):
        pa = self.pa
        obj = ObjectType()
        arr = pa.chunked_array([[1, 2], [3, 4]], type=pa.int64())

        class _Opts:
            safe = True

        result = obj._cast_chunked_array(arr, _Opts())
        self.assertIs(result, arr)

    def test_cast_string_array_returns_input_unchanged(self):
        pa = self.pa
        obj = ObjectType()
        arr = pa.array(["hello", "world"], type=pa.string())

        class _Opts:
            safe = True

        result = obj._cast_arrow_array(arr, _Opts())
        self.assertIs(result, arr)

    def test_cast_binary_array_returns_input_unchanged(self):
        pa = self.pa
        obj = ObjectType()
        arr = pa.array([b"\x00", b"\x01"], type=pa.binary())

        class _Opts:
            safe = True

        result = obj._cast_arrow_array(arr, _Opts())
        self.assertIs(result, arr)


class TestObjectTypeCastBypassPolars(PolarsTestCase):

    def test_cast_polars_series_returns_input_unchanged(self):
        pl = self.pl
        obj = ObjectType()
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)

        class _Opts:
            safe = True
            target_field = None

        result = obj._cast_polars_series(s, _Opts())
        self.assertIs(result, s)


# ---------------------------------------------------------------------------
# Defaults (pure logic)
# ---------------------------------------------------------------------------


class TestObjectTypeDefaults(unittest.TestCase):

    def test_default_pyobj_nullable(self):
        self.assertIsNone(ObjectType().default_pyobj(nullable=True))

    def test_default_pyobj_not_nullable_raises(self):
        with self.assertRaisesRegex(NotImplementedError, "not supported"):
            ObjectType().default_pyobj(nullable=False)


# ---------------------------------------------------------------------------
# DataType.from_parsed integration
# ---------------------------------------------------------------------------


class TestFromParsed(unittest.TestCase):

    def test_from_str_object(self):
        result = DataType.from_str("object")
        self.assertIsInstance(result, ObjectType)

    def test_from_str_any(self):
        # Parser maps "any" to OBJECT type_id too.
        result = DataType.from_str("any")
        self.assertIsInstance(result, ObjectType)

    def test_from_str_variant(self):
        # Parser maps "variant" to OBJECT type_id.
        result = DataType.from_str("variant")
        self.assertIsInstance(result, ObjectType)

    def test_from_str_json_still_string(self):
        # JSON type_id should remain StringType.
        result = DataType.from_str("json")
        self.assertIsInstance(result, StringType)


# ---------------------------------------------------------------------------
# DataType.from_pytype integration
# ---------------------------------------------------------------------------


class TestFromPytype(unittest.TestCase):

    def test_from_pytype_object(self):
        result = DataType.from_pytype(object)
        self.assertIsInstance(result, ObjectType)

    def test_from_pytype_any_still_string(self):
        # `Any` type hint still maps to StringType.
        result = DataType.from_pytype(Any)
        self.assertIsInstance(result, StringType)


# ---------------------------------------------------------------------------
# DataType.from_any integration
# ---------------------------------------------------------------------------


class TestFromAny(unittest.TestCase):

    def test_from_any_string_object(self):
        result = DataType.from_any("object")
        self.assertIsInstance(result, ObjectType)

    def test_from_any_dict_object(self):
        result = DataType.from_any({"id": int(DataTypeId.OBJECT)})
        self.assertIsInstance(result, ObjectType)

    def test_from_any_type_id_int(self):
        result = DataType.from_any(int(DataTypeId.OBJECT))
        self.assertIsInstance(result, ObjectType)
