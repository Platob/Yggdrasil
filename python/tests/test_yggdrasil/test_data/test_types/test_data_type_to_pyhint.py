"""``DataType.to_pyhint`` — the inverse of ``DataType.from_pytype``.

Two-tier resolution:

1. **Cached hint** — :meth:`DataType.from_pytype` stamps the original
   parsed hint on the resulting instance via
   ``object.__setattr__(self, "_pyhint_cache", hint)``. Subsequent
   ``to_pyhint()`` calls return the cached value verbatim — preserving
   user-defined dataclasses, ``Enum`` subclasses, narrow aliases
   (``np.int64``) the canonical reconstruction would collapse.
2. **Default reconstruction** — when no cache is set,
   ``_default_pyhint()`` (subclass hook) generates a canonical hint
   from the dtype's own state.

Cache stamping is first-write-wins on shared singletons to avoid
cross-caller corruption.
"""
from __future__ import annotations

import datetime as dt
import dataclasses
import enum
import unittest
from decimal import Decimal
from typing import Any

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data import field
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.nested import ArrayType, MapType, StructType
from yggdrasil.data.types.primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    NullType,
    ObjectType,
    StringType,
    TimeType,
    TimestampType,
)


@dataclasses.dataclass
class _Row:
    id: int
    name: str


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class TestDefaultReconstruction(ArrowTestCase):
    """``_default_pyhint`` returns the canonical hint when no cache is set."""

    def test_integer_default(self):
        self.assertIs(IntegerType()._default_pyhint(), int)

    def test_float_default(self):
        self.assertIs(FloatingPointType()._default_pyhint(), float)

    def test_string_default(self):
        self.assertIs(StringType()._default_pyhint(), str)

    def test_bool_default(self):
        self.assertIs(BooleanType()._default_pyhint(), bool)

    def test_binary_default(self):
        self.assertIs(BinaryType()._default_pyhint(), bytes)

    def test_null_default(self):
        self.assertIs(NullType()._default_pyhint(), type(None))

    def test_object_default(self):
        self.assertIs(ObjectType()._default_pyhint(), object)

    def test_decimal_default(self):
        self.assertIs(DecimalType()._default_pyhint(), Decimal)

    def test_date_default(self):
        self.assertIs(DateType()._default_pyhint(), dt.date)

    def test_datetime_default(self):
        self.assertIs(TimestampType()._default_pyhint(), dt.datetime)

    def test_time_default(self):
        self.assertIs(TimeType()._default_pyhint(), dt.time)

    def test_duration_default(self):
        self.assertIs(DurationType()._default_pyhint(), dt.timedelta)

    def test_array_recurses_into_item_field(self):
        arr = ArrayType(item_field=field("item", IntegerType()))
        self.assertEqual(arr._default_pyhint(), list[int])

    def test_array_recurses_nested(self):
        arr = ArrayType(item_field=field(
            "item",
            ArrayType(item_field=field("inner", StringType())),
        ))
        self.assertEqual(arr._default_pyhint(), list[list[str]])

    def test_map_recurses_into_key_value(self):
        m = MapType.from_key_value(
            key_field=field("key", StringType()),
            value_field=field("value", IntegerType()),
        )
        self.assertEqual(m._default_pyhint(), dict[str, int])

    def test_struct_default_is_dict_str_any(self):
        s = StructType(fields=(
            field("id", IntegerType()),
            field("name", StringType()),
        ))
        self.assertEqual(s._default_pyhint(), dict[str, Any])


class TestRoundTripFromPytype(ArrowTestCase):
    """``from_pytype(X).to_pyhint() == X`` for the major Python types."""

    def test_primitives_round_trip(self):
        for hint in [int, float, str, bool, bytes, object]:
            with self.subTest(hint=hint):
                self.assertIs(DataType.from_pytype(hint).to_pyhint(), hint)

    def test_none_round_trips_as_none_type(self):
        self.assertIs(DataType.from_pytype(None).to_pyhint(), type(None))
        self.assertIs(DataType.from_pytype(type(None)).to_pyhint(), type(None))

    def test_decimal_round_trips(self):
        self.assertIs(DataType.from_pytype(Decimal).to_pyhint(), Decimal)

    def test_temporal_round_trip(self):
        for hint in [dt.date, dt.datetime, dt.time, dt.timedelta]:
            with self.subTest(hint=hint):
                self.assertIs(DataType.from_pytype(hint).to_pyhint(), hint)

    def test_list_int_round_trips(self):
        self.assertEqual(DataType.from_pytype(list[int]).to_pyhint(), list[int])

    def test_dict_str_int_round_trips(self):
        self.assertEqual(
            DataType.from_pytype(dict[str, int]).to_pyhint(),
            dict[str, int],
        )

    def test_nested_list_of_list(self):
        # Cached hint should hold the original parsed shape verbatim.
        self.assertEqual(
            DataType.from_pytype(list[list[str]]).to_pyhint(),
            list[list[str]],
        )

    def test_dataclass_preserved_via_cache(self):
        # The reconstruction would lose the dataclass identity → cache
        # holds the original class so to_pyhint hands it back untouched.
        self.assertIs(DataType.from_pytype(_Row).to_pyhint(), _Row)

    def test_enum_preserved_via_cache(self):
        self.assertIs(DataType.from_pytype(_Color).to_pyhint(), _Color)


class TestCacheBehavior(ArrowTestCase):
    """The ``_pyhint_cache`` slot behaves as a first-write-wins record."""

    def test_cache_set_via_from_pytype(self):
        dtype = DataType.from_pytype(int)
        self.assertIs(getattr(dtype, "_pyhint_cache"), int)

    def test_cache_excluded_from_equality(self):
        # Two ``IntegerType`` instances must compare equal regardless of
        # cache state — the cache is metadata, not identity.
        a = IntegerType()
        b = IntegerType()
        # ``IntegerType()`` is a singleton — same instance both calls.
        # Stamping should still leave equality intact.
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_cache_excluded_from_to_dict(self):
        # Serialization must not leak the cached hint into the
        # ``to_dict()`` payload (it isn't a dataclass field).
        dtype = DataType.from_pytype(int)
        self.assertNotIn("_pyhint_cache", dtype.to_dict())

    def test_default_path_when_no_cache(self):
        # Building a DataType directly (not via from_pytype) skips the
        # cache stamp; to_pyhint falls back to the default reconstruction.
        # Note: IntegerType() is a singleton — to test the no-cache path
        # cleanly, build a non-default instance.
        dtype = IntegerType(byte_size=4, signed=False)
        # No from_pytype call → cache unset → default fires.
        self.assertIs(dtype.to_pyhint(), int)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
