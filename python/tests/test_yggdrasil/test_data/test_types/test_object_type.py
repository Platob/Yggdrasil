"""``ObjectType`` — the variant / opaque-Python-object catch-all.

ObjectType has a deliberately weird contract: it claims to be a real
type but does not own a value shape. That means:

* No engine can *infer* it from a native dtype (``handles_*`` is
  False / always-False except polars Object).
* Cast operations are identity — every ``_cast_*`` override returns
  its input untouched. The base-class fast-path covers the public
  entry points; the subclass overrides are defensive against direct
  internal calls (e.g. tabular casts inside a struct walk).
* Defaults only exist for ``nullable=True``; non-nullable raises
  because there is no "zero variant".

The file is organized by surface so a regression points at the right
piece of behavior (basic identity, arrow side, polars side, dict
serde, merge, defaults, cast bypass).
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
# Identity / dunder
# ---------------------------------------------------------------------------


class TestObjectTypeIdentity(unittest.TestCase):

    def test_type_id(self) -> None:
        self.assertEqual(ObjectType().type_id, DataTypeId.OBJECT)

    def test_children_fields_is_empty(self) -> None:
        self.assertEqual(ObjectType().children, [])

    def test_repr(self) -> None:
        self.assertEqual(repr(ObjectType()), "ObjectType()")

    def test_str(self) -> None:
        self.assertEqual(str(ObjectType()), "object")

    def test_is_frozen(self) -> None:
        obj = ObjectType()
        with self.assertRaises(AttributeError):
            obj.foo = "bar"  # type: ignore[attr-defined]

    def test_equal_instances_are_equal(self) -> None:
        self.assertEqual(ObjectType(), ObjectType())

    def test_equal_instances_share_hash(self) -> None:
        self.assertEqual(hash(ObjectType()), hash(ObjectType()))


# ---------------------------------------------------------------------------
# Arrow side
# ---------------------------------------------------------------------------


class TestObjectTypeArrow(ArrowTestCase):

    def test_to_arrow_uses_large_binary(self) -> None:
        self.assertEqual(ObjectType().to_arrow(), self.pa.large_binary())

    def test_handles_arrow_type_is_always_false(self) -> None:
        pa = self.pa
        for dtype in (pa.large_binary(), pa.string(), pa.int64(), pa.null()):
            with self.subTest(dtype=dtype):
                self.assertFalse(ObjectType.handles_arrow_type(dtype))

    def test_from_arrow_type_raises_with_helpful_message(self) -> None:
        with self.assertRaisesRegex(TypeError, "Cannot infer ObjectType"):
            ObjectType.from_arrow_type(self.pa.large_binary())

    def test_default_arrow_scalar_nullable_returns_typed_null(self) -> None:
        scalar = ObjectType().default_arrow_scalar(nullable=True)

        self.assertEqual(scalar.type, pa.large_binary())
        self.assertIsNone(scalar.as_py())

    def test_default_arrow_scalar_non_nullable_raises(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "nullable=False"):
            ObjectType().default_arrow_scalar(nullable=False)


# ---------------------------------------------------------------------------
# Polars side
# ---------------------------------------------------------------------------


class TestObjectTypePolars(PolarsTestCase):

    def test_to_polars_is_pl_object(self) -> None:
        self.assertEqual(ObjectType().to_polars(), self.pl.Object)

    def test_handles_polars_type_true_for_object(self) -> None:
        self.assertTrue(ObjectType.handles_polars_type(self.pl.Object))

    def test_handles_polars_type_false_for_others(self) -> None:
        for dtype in (self.pl.String, self.pl.Int64):
            with self.subTest(dtype=dtype):
                self.assertFalse(ObjectType.handles_polars_type(dtype))

    def test_from_polars_type_object_round_trips(self) -> None:
        self.assertIsInstance(
            ObjectType.from_polars_type(self.pl.Object), ObjectType
        )

    def test_from_polars_type_rejects_non_object(self) -> None:
        with self.assertRaisesRegex(TypeError, "Expected Polars Object dtype"):
            ObjectType.from_polars_type(self.pl.String)

    def test_datatype_dispatch_resolves_to_object_type(self) -> None:
        self.assertIsInstance(
            DataType.from_polars_type(self.pl.Object), ObjectType
        )


# ---------------------------------------------------------------------------
# Spark side (no SparkSession needed — pure type-system checks)
# ---------------------------------------------------------------------------


class TestObjectTypeSpark(unittest.TestCase):

    def test_handles_spark_type_is_always_false(self) -> None:
        # Spark has no native object type.
        self.assertFalse(ObjectType.handles_spark_type(object()))

    def test_from_spark_type_raises_with_helpful_message(self) -> None:
        with self.assertRaisesRegex(TypeError, "Cannot infer ObjectType"):
            ObjectType.from_spark_type(object())


# ---------------------------------------------------------------------------
# Databricks DDL
# ---------------------------------------------------------------------------


class TestObjectTypeDDL(unittest.TestCase):

    def test_to_spark_name_is_binary(self) -> None:
        self.assertEqual(ObjectType().to_spark_name(), "BINARY")


# ---------------------------------------------------------------------------
# Dict serialization
# ---------------------------------------------------------------------------


class TestObjectTypeDict(unittest.TestCase):

    def test_to_dict_carries_id_and_name(self) -> None:
        d = ObjectType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.OBJECT))
        self.assertEqual(d["name"], "OBJECT")

    def test_handles_dict_by_id(self) -> None:
        self.assertTrue(ObjectType.handles_dict({"id": int(DataTypeId.OBJECT)}))

    def test_handles_dict_by_name_case_insensitive(self) -> None:
        self.assertTrue(ObjectType.handles_dict({"name": "OBJECT"}))
        self.assertTrue(ObjectType.handles_dict({"name": "object"}))

    def test_handles_dict_false_for_other_types(self) -> None:
        self.assertFalse(ObjectType.handles_dict({"id": int(DataTypeId.STRING)}))
        self.assertFalse(ObjectType.handles_dict({"name": "STRING"}))

    def test_from_dict_round_trip(self) -> None:
        d = {"id": int(DataTypeId.OBJECT), "name": "OBJECT"}
        self.assertIsInstance(ObjectType.from_dict(d), ObjectType)

    def test_to_dict_from_dict_round_trip(self) -> None:
        original = ObjectType()
        self.assertEqual(ObjectType.from_dict(original.to_dict()), original)

    def test_datatype_from_dict_dispatches_to_object_type(self) -> None:
        d = {"id": int(DataTypeId.OBJECT), "name": "OBJECT"}
        self.assertIsInstance(DataType.from_dict(d), ObjectType)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestObjectTypeMerge(unittest.TestCase):

    def test_object_with_object_returns_self(self) -> None:
        a = ObjectType()
        self.assertIs(a.merge_with(ObjectType()), a)

    def test_null_with_object_resolves_to_object(self) -> None:
        self.assertIsInstance(NullType().merge_with(ObjectType()), ObjectType)

    def test_object_with_string_picks_string_under_default_widening(self) -> None:
        # OBJECT=0 vs STRING=11; merge prefers the larger type_id when
        # neither downcast nor upcast is set (`_merge_with_different_id`
        # default). See base.DataType._merge_with_different_id.
        a = ObjectType()
        result = a.merge_with(StringType())
        self.assertIsInstance(result, StringType)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestObjectTypeDefaults(unittest.TestCase):

    def test_nullable_default_is_none(self) -> None:
        self.assertIsNone(ObjectType().default_pyobj(nullable=True))

    def test_non_nullable_default_raises(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "not supported"):
            ObjectType().default_pyobj(nullable=False)


# ---------------------------------------------------------------------------
# Cast bypass — every cast operation is identity for ObjectType
# ---------------------------------------------------------------------------


class _OpaqueOpts:
    """Minimal stand-in for ``CastOptions`` — only ``safe`` is read."""

    safe = True
    target_field = None


class TestObjectTypeCastBypassArrow(ArrowTestCase):

    def test_arrow_array_passthrough_int(self) -> None:
        arr = self.pa.array([1, 2, 3], type=self.pa.int32())
        self.assertIs(ObjectType()._cast_arrow_array(arr, _OpaqueOpts()), arr)

    def test_arrow_array_passthrough_string(self) -> None:
        arr = self.pa.array(["hello", "world"], type=self.pa.string())
        self.assertIs(ObjectType()._cast_arrow_array(arr, _OpaqueOpts()), arr)

    def test_arrow_array_passthrough_binary(self) -> None:
        arr = self.pa.array([b"\x00", b"\x01"], type=self.pa.binary())
        self.assertIs(ObjectType()._cast_arrow_array(arr, _OpaqueOpts()), arr)

    def test_chunked_array_passthrough(self) -> None:
        arr = self.pa.chunked_array([[1, 2], [3, 4]], type=self.pa.int64())
        self.assertIs(ObjectType()._cast_chunked_array(arr, _OpaqueOpts()), arr)


class TestObjectTypeCastBypassPolars(PolarsTestCase):

    def test_polars_series_passthrough(self) -> None:
        s = self.pl.Series("x", [1, 2, 3], dtype=self.pl.Int64)
        self.assertIs(ObjectType()._cast_polars_series(s, _OpaqueOpts()), s)


# ---------------------------------------------------------------------------
# Cross-entry-point integration — DataType.from_str / from_pytype / from_any
# ---------------------------------------------------------------------------


class TestObjectTypeIntegration(unittest.TestCase):

    def test_from_str_object(self) -> None:
        self.assertIsInstance(DataType.from_str("object"), ObjectType)

    def test_from_str_any_alias(self) -> None:
        self.assertIsInstance(DataType.from_str("any"), ObjectType)

    def test_from_str_variant_alias(self) -> None:
        self.assertIsInstance(DataType.from_str("variant"), ObjectType)

    def test_from_str_json_resolves_to_bjson(self) -> None:
        # Bare ``json`` defaults to the binary-shaped variant; the
        # text form is reachable via ``sjson`` / ``json_string``.
        from yggdrasil.data.types import BJsonType, SJsonType

        self.assertIsInstance(DataType.from_str("json"), BJsonType)
        self.assertIsInstance(DataType.from_str("sjson"), SJsonType)
        self.assertIsInstance(DataType.from_str("jsonb"), BJsonType)

    def test_from_pytype_object_class(self) -> None:
        self.assertIsInstance(DataType.from_pytype(object), ObjectType)

    def test_from_pytype_Any_is_string_not_object(self) -> None:
        self.assertIsInstance(DataType.from_pytype(Any), StringType)

    def test_from_any_string(self) -> None:
        self.assertIsInstance(DataType.from_any("object"), ObjectType)

    def test_from_any_dict(self) -> None:
        self.assertIsInstance(
            DataType.from_any({"id": int(DataTypeId.OBJECT)}), ObjectType
        )

    def test_from_any_type_id_int(self) -> None:
        self.assertIsInstance(
            DataType.from_any(int(DataTypeId.OBJECT)), ObjectType
        )
