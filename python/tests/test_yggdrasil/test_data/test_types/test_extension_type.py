"""Tests for ExtensionType -- the user-defined extension type system.

Covers:
- Registry auto-registration on subclass definition
- Registry lookup (success + failure)
- PyArrow bridge: to_arrow / from_arrow_type round-trip
- Arrow IPC serialize / deserialize
- Polars / Spark / Databricks DDL fallback to storage type
- Dict round-trip (to_dict / from_dict)
- Merge behavior
- Error messages on misuse
- Extension types with parameters
- Repr
"""
from __future__ import annotations

import json
import unittest
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, ClassVar

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types.extensions.base import (
    ExtensionType,
    _EXTENSION_REGISTRY,
    get_extension_registry,
    get_extension_type,
)
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.polars.tests import PolarsTestCase


# ---------------------------------------------------------------------------
# Fixture: concrete extension types for testing
# ---------------------------------------------------------------------------
_TEST_PREFIX = "yggdrasil.test."


@dataclass(frozen=True)
class _UuidType(ExtensionType):
    """Simple extension type with no extra fields -- UUID as fixed 16-byte binary."""
    extension_name: ClassVar[str] = f"{_TEST_PREFIX}uuid"
    storage_type: ClassVar[pa.DataType] = pa.binary(16)

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        return _uuid.UUID(int=0).bytes


@dataclass(frozen=True)
class _QuantityType(ExtensionType):
    """Extension type with a parameter -- physical quantity with a unit label."""
    extension_name: ClassVar[str] = f"{_TEST_PREFIX}quantity"
    storage_type: ClassVar[pa.DataType] = pa.float64()
    unit: str = "m"


# ---------------------------------------------------------------------------
# Registry tests (pure logic, no engine dependency)
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):


    def test_auto_registration(self):
        self.assertIn(_UuidType.extension_name, _EXTENSION_REGISTRY)
        self.assertIn(_QuantityType.extension_name, _EXTENSION_REGISTRY)

    def test_get_extension_type_success(self):
        cls = get_extension_type(_UuidType.extension_name)
        self.assertIs(cls, _UuidType)

    def test_get_extension_type_failure(self):
        with self.assertRaisesRegex(KeyError, "No extension type registered"):
            get_extension_type("totally.bogus.name")

    def test_get_extension_registry_is_snapshot(self):
        snapshot = get_extension_registry()
        self.assertIn(_UuidType.extension_name, snapshot)
        snapshot.clear()
        self.assertIn(_UuidType.extension_name, _EXTENSION_REGISTRY)

    def test_collision_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "Extension name collision"):
            import pyarrow as pa

            @dataclass(frozen=True)
            class _Duplicate(ExtensionType):
                extension_name: ClassVar[str] = _UuidType.extension_name
                storage_type: ClassVar[Any] = pa.binary(16)


# ---------------------------------------------------------------------------
# type_id (pure logic)
# ---------------------------------------------------------------------------

class TestTypeId(unittest.TestCase):


    def test_type_id(self):
        self.assertEqual(_UuidType().type_id, DataTypeId.EXTENSION)
        self.assertEqual(_QuantityType(unit="kg").type_id, DataTypeId.EXTENSION)


# ---------------------------------------------------------------------------
# Serialization / deserialization (pure logic)
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):


    def test_serialize_no_fields(self):
        self.assertEqual(_UuidType().serialize_metadata(), b"")

    def test_serialize_with_fields(self):
        raw = _QuantityType(unit="kg").serialize_metadata()
        payload = json.loads(raw)
        self.assertEqual(payload, {"unit": "kg"})

    def test_deserialize_no_fields(self):
        restored = _UuidType.deserialize_metadata(b"")
        self.assertIsInstance(restored, _UuidType)

    def test_deserialize_with_fields(self):
        raw = json.dumps({"unit": "kg"}).encode()
        restored = _QuantityType.deserialize_metadata(raw)
        self.assertIsInstance(restored, _QuantityType)
        self.assertEqual(restored.unit, "kg")

    def test_round_trip(self):
        original = _QuantityType(unit="s")
        raw = original.serialize_metadata()
        restored = _QuantityType.deserialize_metadata(raw)
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Arrow conversion
# ---------------------------------------------------------------------------

class TestArrowConversion(ArrowTestCase):


    def test_to_arrow_returns_extension_type(self):
        pa = self.pa
        arrow_type = _UuidType().to_arrow()
        self.assertIsInstance(arrow_type, pa.ExtensionType)
        self.assertEqual(arrow_type.extension_name, _UuidType.extension_name)

    def test_to_arrow_preserves_storage_type(self):
        pa = self.pa
        arrow_type = _UuidType().to_arrow()
        self.assertEqual(arrow_type.storage_type, pa.binary(16))

    def test_to_arrow_with_params(self):
        pa = self.pa
        arrow_type = _QuantityType(unit="kg").to_arrow()
        self.assertIsInstance(arrow_type, pa.ExtensionType)
        self.assertEqual(arrow_type.storage_type, pa.float64())
        raw = arrow_type.__arrow_ext_serialize__()
        payload = json.loads(raw)
        self.assertEqual(payload["unit"], "kg")

    def test_handles_arrow_type_true(self):
        arrow_type = _UuidType().to_arrow()
        self.assertTrue(ExtensionType.handles_arrow_type(arrow_type))
        self.assertTrue(_UuidType.handles_arrow_type(arrow_type))

    def test_handles_arrow_type_false_for_standard(self):
        pa = self.pa
        self.assertFalse(ExtensionType.handles_arrow_type(pa.int64()))
        self.assertFalse(_UuidType.handles_arrow_type(pa.int64()))

    def test_handles_arrow_type_wrong_subclass(self):
        arrow_type = _UuidType().to_arrow()
        self.assertFalse(_QuantityType.handles_arrow_type(arrow_type))

    def test_from_arrow_type_base_class_dispatch(self):
        arrow_type = _QuantityType(unit="kg").to_arrow()
        restored = ExtensionType.from_arrow_type(arrow_type)
        self.assertIsInstance(restored, _QuantityType)
        self.assertEqual(restored.unit, "kg")

    def test_from_arrow_type_concrete_subclass(self):
        arrow_type = _UuidType().to_arrow()
        restored = _UuidType.from_arrow_type(arrow_type)
        self.assertIsInstance(restored, _UuidType)

    def test_from_arrow_type_wrong_subclass_raises(self):
        arrow_type = _UuidType().to_arrow()
        with self.assertRaisesRegex(TypeError, "does not match"):
            _QuantityType.from_arrow_type(arrow_type)

    def test_from_arrow_type_not_extension_raises(self):
        pa = self.pa
        with self.assertRaisesRegex(TypeError, "Expected a PyArrow ExtensionType"):
            ExtensionType.from_arrow_type(pa.int64())

    def test_arrow_round_trip_no_params(self):
        original = _UuidType()
        restored = ExtensionType.from_arrow_type(original.to_arrow())
        self.assertIs(type(restored), _UuidType)

    def test_arrow_round_trip_with_params(self):
        original = _QuantityType(unit="m/s")
        restored = ExtensionType.from_arrow_type(original.to_arrow())
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Arrow array operations
# ---------------------------------------------------------------------------

class TestArrowArrays(ArrowTestCase):


    def test_extension_array_from_storage(self):
        pa = self.pa
        arrow_type = _UuidType().to_arrow()
        storage = pa.array([b"\x00" * 16, b"\x01" * 16], type=pa.binary(16))
        ext_array = pa.ExtensionArray.from_storage(arrow_type, storage)
        self.assertEqual(len(ext_array), 2)
        self.assertEqual(ext_array.type.extension_name, _UuidType.extension_name)

    def test_extension_array_storage_roundtrip(self):
        pa = self.pa
        arrow_type = _UuidType().to_arrow()
        storage = pa.array([b"\xAB" * 16], type=pa.binary(16))
        ext_array = pa.ExtensionArray.from_storage(arrow_type, storage)
        self.assertTrue(ext_array.storage.equals(storage))


# ---------------------------------------------------------------------------
# Arrow IPC round-trip
# ---------------------------------------------------------------------------

class TestArrowIPC(ArrowTestCase):


    def test_ipc_roundtrip_no_params(self):
        pa = self.pa
        arrow_type = _UuidType().to_arrow()
        storage = pa.array([b"\x00" * 16], type=pa.binary(16))
        arr = pa.ExtensionArray.from_storage(arrow_type, storage)
        table = pa.table({"col": arr})

        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        reader = pa.ipc.open_stream(sink.getvalue())
        result = reader.read_all()

        col_type = result.column("col").type
        self.assertIsInstance(col_type, pa.ExtensionType)
        self.assertEqual(col_type.extension_name, _UuidType.extension_name)

    def test_ipc_roundtrip_with_params(self):
        pa = self.pa
        arrow_type = _QuantityType(unit="kg").to_arrow()
        storage = pa.array([1.0, 2.0, 3.0], type=pa.float64())
        arr = pa.ExtensionArray.from_storage(arrow_type, storage)
        table = pa.table({"mass": arr})

        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        reader = pa.ipc.open_stream(sink.getvalue())
        result = reader.read_all()

        col_type = result.column("mass").type
        self.assertIsInstance(col_type, pa.ExtensionType)
        self.assertEqual(col_type.extension_name, _QuantityType.extension_name)
        raw = col_type.__arrow_ext_serialize__()
        self.assertEqual(json.loads(raw)["unit"], "kg")


# ---------------------------------------------------------------------------
# Polars fallback
# ---------------------------------------------------------------------------

class TestPolarsConversion(PolarsTestCase):


    def test_to_polars_falls_back_to_storage(self):
        polars_type = _QuantityType(unit="m").to_polars()
        self.assertEqual(polars_type, self.pl.Float64)

    def test_handles_polars_type_is_false(self):
        self.assertFalse(_UuidType.handles_polars_type(object()))


# ---------------------------------------------------------------------------
# Databricks DDL fallback (pure logic)
# ---------------------------------------------------------------------------

class TestExtDatabricksDDL(unittest.TestCase):


    def test_ddl_falls_back_to_storage(self):
        self.assertEqual(_QuantityType(unit="m").to_databricks_ddl(), "DOUBLE")

    def test_ddl_binary_storage(self):
        ddl = _UuidType().to_databricks_ddl()
        self.assertEqual(ddl, "BINARY")


# ---------------------------------------------------------------------------
# Dict round-trip (pure logic)
# ---------------------------------------------------------------------------

class TestDictRoundTrip(unittest.TestCase):


    def test_to_dict_no_params(self):
        d = _UuidType().to_dict()
        self.assertEqual(d["id"], int(DataTypeId.EXTENSION))
        self.assertEqual(d["extension_name"], _UuidType.extension_name)

    def test_to_dict_with_params(self):
        d = _QuantityType(unit="kg").to_dict()
        self.assertEqual(d["unit"], "kg")
        self.assertEqual(d["extension_name"], _QuantityType.extension_name)

    def test_handles_dict_base(self):
        d = {"id": int(DataTypeId.EXTENSION), "extension_name": _UuidType.extension_name}
        self.assertTrue(ExtensionType.handles_dict(d))

    def test_handles_dict_concrete(self):
        d = {"id": int(DataTypeId.EXTENSION), "extension_name": _UuidType.extension_name}
        self.assertTrue(_UuidType.handles_dict(d))
        self.assertFalse(_QuantityType.handles_dict(d))

    def test_from_dict_base_dispatch(self):
        d = {"id": int(DataTypeId.EXTENSION), "extension_name": _QuantityType.extension_name, "unit": "s"}
        restored = ExtensionType.from_dict(d)
        self.assertIsInstance(restored, _QuantityType)
        self.assertEqual(restored.unit, "s")

    def test_from_dict_concrete(self):
        d = {"id": int(DataTypeId.EXTENSION), "extension_name": _UuidType.extension_name}
        restored = _UuidType.from_dict(d)
        self.assertIsInstance(restored, _UuidType)

    def test_from_dict_unknown_name_raises(self):
        d = {"id": int(DataTypeId.EXTENSION), "extension_name": "nope.nope.nope"}
        with self.assertRaisesRegex(ValueError, "No extension type registered"):
            ExtensionType.from_dict(d)

    def test_from_dict_missing_name_raises(self):
        d = {"id": int(DataTypeId.EXTENSION)}
        with self.assertRaisesRegex(ValueError, "without 'extension_name'"):
            ExtensionType.from_dict(d)

    def test_full_round_trip(self):
        original = _QuantityType(unit="N")
        restored = ExtensionType.from_dict(original.to_dict())
        self.assertEqual(restored, original)


# ---------------------------------------------------------------------------
# Merge (pure logic)
# ---------------------------------------------------------------------------

class TestMerge(unittest.TestCase):


    def test_same_type_keeps_self(self):
        a = _UuidType()
        b = _UuidType()
        self.assertIs(a.merge_with(b), a)

    def test_different_extension_keeps_self(self):
        a = _UuidType()
        b = _QuantityType(unit="m")
        result = a.merge_with(b)
        self.assertIs(result, a)

    def test_merge_with_null_returns_self(self):
        from yggdrasil.data.types.primitive import NullType
        a = _UuidType()
        result = a.merge_with(NullType())
        self.assertIs(result, a)


# ---------------------------------------------------------------------------
# Repr (pure logic)
# ---------------------------------------------------------------------------

class TestRepr(unittest.TestCase):


    def test_repr_no_params(self):
        r = repr(_UuidType())
        self.assertEqual(r, "_UuidType()")

    def test_repr_with_params(self):
        r = repr(_QuantityType(unit="kg"))
        self.assertEqual(r, "_QuantityType(unit='kg')")


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------

class TestCasting(ArrowTestCase):


    def test_cast_arrow_array(self):
        pa = self.pa
        qty = _QuantityType(unit="m")
        storage = pa.array([1.0, 2.0, 3.0], type=pa.float64())

        class _Opts:
            safe = True

        result = qty._cast_arrow_array(storage, _Opts())
        self.assertIsInstance(result.type, pa.ExtensionType)
        self.assertEqual(result.type.extension_name, _QuantityType.extension_name)
        self.assertTrue(result.storage.equals(storage))

    def test_cast_unwraps_extension_source(self):
        pa = self.pa
        qty = _QuantityType(unit="m")
        arrow_type = qty.to_arrow()
        storage = pa.array([10.0], type=pa.float64())
        ext_arr = pa.ExtensionArray.from_storage(arrow_type, storage)

        class _Opts:
            safe = True

        result = qty._cast_arrow_array(ext_arr, _Opts())
        self.assertIsInstance(result.type, pa.ExtensionType)
        self.assertTrue(result.storage.equals(storage))


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults(ArrowTestCase):


    def test_default_pyobj_nullable(self):
        self.assertIsNone(_UuidType().default_pyobj(nullable=True))

    def test_default_pyobj_not_nullable_custom(self):
        result = _UuidType().default_pyobj(nullable=False)
        self.assertEqual(result, _uuid.UUID(int=0).bytes)

    def test_default_pyobj_not_nullable_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "default_pyobj"):
            _QuantityType(unit="m").default_pyobj(nullable=False)

    def test_default_arrow_scalar_nullable(self):
        s = _UuidType().default_arrow_scalar(nullable=True)
        self.assertIsNone(s.as_py())

    def test_children_fields_empty(self):
        self.assertEqual(_UuidType().children_fields, [])


# ---------------------------------------------------------------------------
# DataType.from_arrow_type dispatch integration
# ---------------------------------------------------------------------------

class TestDataTypeDispatch(ArrowTestCase):


    def test_from_arrow_type_dispatches(self):
        from yggdrasil.data.types.base import DataType
        arrow_type = _QuantityType(unit="kg").to_arrow()
        dt = DataType.from_arrow_type(arrow_type)
        self.assertIsInstance(dt, _QuantityType)
        self.assertEqual(dt.unit, "kg")

    def test_from_arrow_type_unknown_extension(self):
        pa = self.pa
        from yggdrasil.data.types.base import DataType

        class _Alien(pa.ExtensionType):
            def __init__(self):
                pa.ExtensionType.__init__(self, pa.int32(), "test.alien.unknown")
            def __arrow_ext_serialize__(self):
                return b""
            @classmethod
            def __arrow_ext_deserialize__(cls, storage_type, serialized):
                return cls()

        try:
            pa.register_extension_type(_Alien())
        except pa.ArrowKeyError:
            pass

        alien_type = _Alien()
        with self.assertRaises(TypeError):
            DataType.from_arrow_type(alien_type)

        with self.assertRaisesRegex(TypeError, "No Yggdrasil extension type registered"):
            ExtensionType.from_arrow_type(alien_type)

        try:
            pa.unregister_extension_type("test.alien.unknown")
        except Exception:
            pass
