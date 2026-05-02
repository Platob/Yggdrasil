"""``DataType.from_pytype`` and ``DataType.from_dataclass``.

Two entry points; one shared design rule: take whatever Python annotation
the caller actually has (a builtin, a typing form, an enum, a dataclass)
and resolve it to the most-specific :class:`DataType` we can.

* Scalars hit a direct mapping.
* Optionals unwrap; literals collapse to their value type.
* Mixed unions degrade to ``StringType``.
* Containers map to nested types: ``list[T]`` → array, ``tuple[T, ...]``
  → array, fixed ``tuple[A, B]`` → struct, ``dict[K, V]`` and TypedDict
  → map.
* Plain annotated classes (``class Foo: a: int``) and dataclasses both
  promote to a struct.
"""
from __future__ import annotations

import datetime as dt
import decimal
import enum
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict

from yggdrasil.arrow.tests import ArrowTestCase
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
    StringType,
    TimeType,
    TimestampType,
)


# ---------------------------------------------------------------------------
# Fixture types — annotated/typed classes used by the tests.
# ---------------------------------------------------------------------------


class _TotalUserPayload(TypedDict):
    id: int
    name: str


class _PartialUserPayload(TypedDict, total=False):
    id: int
    name: str


@dataclass
class _TradeRow:
    ts: dt.datetime
    price: float
    volume: int


class _PlainAnnotatedClass:
    a: int
    b: str


class _Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"


# ---------------------------------------------------------------------------
# Scalar hints
# ---------------------------------------------------------------------------


class TestScalarHints(ArrowTestCase):

    def test_scalar_matrix(self) -> None:
        pa = self.pa
        cases = [
            (None, NullType, pa.null()),
            (bool, BooleanType, pa.bool_()),
            (int, IntegerType, pa.int64()),
            (float, FloatingPointType, pa.float64()),
            (str, StringType, pa.string()),
            (bytes, BinaryType, pa.binary()),
            (bytearray, BinaryType, pa.binary()),
            (memoryview, BinaryType, pa.binary()),
            (decimal.Decimal, DecimalType, pa.decimal128(38, 18)),
            (dt.date, DateType, pa.date32()),
            (dt.time, TimeType, pa.time64("us")),
            (dt.datetime, TimestampType, pa.timestamp("us")),
            (dt.timedelta, DurationType, pa.duration("us")),
            (uuid.UUID, StringType, pa.string()),
            (Any, StringType, pa.string()),
        ]
        for hint, expected_type, expected_arrow in cases:
            with self.subTest(hint=hint):
                dtype = DataType.from_pytype(hint)
                self.assertIsInstance(dtype, expected_type)
                self.assertEqual(dtype.to_arrow(), expected_arrow)


# ---------------------------------------------------------------------------
# Optionals, Literals, Unions
# ---------------------------------------------------------------------------


class TestComplexHints(ArrowTestCase):

    def test_optional_unwraps_to_inner(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(Optional[int])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_literal_with_homogeneous_values(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(Literal[1, 2, 3])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_literal_with_none_collapses_to_inner(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(Literal[1, None])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_mixed_union_falls_back_to_string(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(int | str)

        self.assertIsInstance(dtype, StringType)
        self.assertEqual(dtype.to_arrow(), pa.string())

    def test_enum_uses_member_value_type(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(_Side)

        self.assertIsInstance(dtype, StringType)
        self.assertEqual(dtype.to_arrow(), pa.string())


# ---------------------------------------------------------------------------
# Containers — list / tuple / dict / TypedDict
# ---------------------------------------------------------------------------


class TestContainerHints(ArrowTestCase):

    def test_list_of_int_becomes_array(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(list[int])

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.int64(), nullable=True)),
        )

    def test_plain_list_uses_string_item_fallback(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(list)

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.string(), nullable=True)),
        )

    def test_variadic_tuple_becomes_array(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(tuple[int, ...])

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.int64(), nullable=True)),
        )

    def test_fixed_tuple_becomes_struct_with_positional_names(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(tuple[int, str])

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("_0", pa.int64(), nullable=False),
                    pa.field("_1", pa.string(), nullable=False),
                ]
            ),
        )

    def test_plain_dict_becomes_unsorted_map(self) -> None:
        dtype = DataType.from_pytype(dict[str, int])

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)

    def test_ordered_dict_becomes_sorted_map(self) -> None:
        dtype = DataType.from_pytype(OrderedDict[str, int])

        self.assertIsInstance(dtype, MapType)
        self.assertTrue(dtype.keys_sorted)

    def test_total_typed_dict_resolves_as_map(self) -> None:
        dtype = DataType.from_pytype(_TotalUserPayload)

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)

    def test_partial_typed_dict_resolves_as_map(self) -> None:
        dtype = DataType.from_pytype(_PartialUserPayload)

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)


# ---------------------------------------------------------------------------
# Annotated classes / dataclasses
# ---------------------------------------------------------------------------


class TestAnnotatedClasses(ArrowTestCase):

    def test_dataclass_with_future_annotations_promotes_to_struct(self) -> None:
        pa = self.pa
        dtype = DataType.from_dataclass(_TradeRow)

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("ts", pa.timestamp("us", "UTC"), nullable=True),
                    pa.field("price", pa.float32(), nullable=True),
                    pa.field("volume", pa.int32(), nullable=True),
                ]
            ),
        )

    def test_plain_annotated_class_promotes_to_struct(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(_PlainAnnotatedClass)

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("a", pa.int32(), nullable=True),
                    pa.field("b", pa.string(), nullable=True),
                ]
            ),
        )
