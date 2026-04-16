from __future__ import annotations

import datetime as dt
import decimal
import enum
import unittest
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


class TotalUserPayload(TypedDict):
    id: int
    name: str


class PartialUserPayload(TypedDict, total=False):
    id: int
    name: str


@dataclass
class TradeRow:
    ts: dt.datetime
    price: float
    volume: int


class PlainAnnotatedClass:
    a: int
    b: str


class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class TestFromPytypeScalars(ArrowTestCase):

    def test_from_pytype_scalar_types(self):
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


class TestFromPytypeComplex(ArrowTestCase):

    def test_from_pytype_optional_unwraps_to_inner_type(self):
        pa = self.pa
        dtype = DataType.from_pytype(Optional[int])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_from_pytype_literal_single_concrete_type(self):
        pa = self.pa
        dtype = DataType.from_pytype(Literal[1, 2, 3])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_from_pytype_literal_optional_collapses_to_inner_type(self):
        pa = self.pa
        dtype = DataType.from_pytype(Literal[1, None])

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_from_pytype_union_mixed_types_falls_back_to_string(self):
        pa = self.pa
        dtype = DataType.from_pytype(int | str)

        self.assertIsInstance(dtype, StringType)
        self.assertEqual(dtype.to_arrow(), pa.string())

    def test_from_pytype_enum_uses_member_value_type(self):
        pa = self.pa
        dtype = DataType.from_pytype(Side)

        self.assertIsInstance(dtype, StringType)
        self.assertEqual(dtype.to_arrow(), pa.string())

    def test_from_pytype_list_of_int_becomes_array(self):
        pa = self.pa
        dtype = DataType.from_pytype(list[int])

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.int64(), nullable=True)),
        )

    def test_from_pytype_plain_list_uses_any_item_fallback(self):
        pa = self.pa
        dtype = DataType.from_pytype(list)

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.string(), nullable=True)),
        )

    def test_from_pytype_tuple_variadic_becomes_array(self):
        pa = self.pa
        dtype = DataType.from_pytype(tuple[int, ...])

        self.assertIsInstance(dtype, ArrayType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.list_(pa.field("item", pa.int64(), nullable=True)),
        )

    def test_from_pytype_tuple_fixed_becomes_struct(self):
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

    def test_from_pytype_dict_becomes_map(self):
        dtype = DataType.from_pytype(dict[str, int])

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)

    def test_from_pytype_ordered_dict_becomes_sorted_map(self):
        dtype = DataType.from_pytype(OrderedDict[str, int])

        self.assertIsInstance(dtype, MapType)
        self.assertTrue(dtype.keys_sorted)

    def test_from_pytype_typed_dict_currently_resolves_as_map(self):
        dtype = DataType.from_pytype(TotalUserPayload)

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)

    def test_from_pytype_partial_typed_dict_currently_resolves_as_map(self):
        dtype = DataType.from_pytype(PartialUserPayload)

        self.assertIsInstance(dtype, MapType)
        self.assertFalse(dtype.keys_sorted)

    def test_from_dataclass_with_future_annotations_uses_string_for_datetime_field(self):
        pa = self.pa
        dtype = DataType.from_dataclass(TradeRow)

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("ts", pa.string(), nullable=True),
                    pa.field("price", pa.float32(), nullable=True),
                    pa.field("volume", pa.int32(), nullable=True),
                ]
            ),
        )

    def test_from_pytype_plain_annotated_class_builds_struct(self):
        pa = self.pa
        dtype = DataType.from_pytype(PlainAnnotatedClass)

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
