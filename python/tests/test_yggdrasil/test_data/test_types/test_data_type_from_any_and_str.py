from __future__ import annotations

import json

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types import MapType
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import DecimalType, IntegerType, StringType


class TestFromAny(ArrowTestCase):

    def test_from_any_accepts_existing_dtype_instance(self):
        src = StringType()
        out = DataType.from_any(src)

        self.assertIs(out, src)

    def test_from_any_accepts_dtype_class(self):
        pa = self.pa
        out = DataType.from_any(StringType)

        self.assertIsInstance(out, StringType)
        self.assertEqual(out.to_arrow(), pa.string())

    def test_from_any_accepts_arrow_datatype(self):
        pa = self.pa
        out = DataType.from_any(pa.int32())

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int32())

    def test_from_any_accepts_arrow_field(self):
        pa = self.pa
        out = DataType.from_any(pa.field("x", pa.int16(), nullable=False))

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int16())

    def test_from_any_accepts_arrow_schema(self):
        pa = self.pa
        schema = pa.schema(
            [
                pa.field("a", pa.int64(), nullable=True),
                pa.field("b", pa.string(), nullable=True),
            ]
        )

        out = DataType.from_any(schema)

        self.assertIsInstance(out, StructType)
        self.assertEqual(out.to_arrow(), pa.struct(list(schema)))

    def test_from_any_rejects_unsupported_object(self):
        class Unsupported:
            pass

        with self.assertRaisesRegex(ValueError, "Cannot convert value of type"):
            DataType.from_any(Unsupported())


class TestFromStr(ArrowTestCase):

    def test_from_str_simple_token_matches_current_parser_behavior(self):
        pa = self.pa
        out = DataType.from_str("int64")

        self.assertIsInstance(out, StringType)
        self.assertEqual(out.to_arrow(), pa.string())

    def test_from_str_json_payload_integer(self):
        pa = self.pa
        payload = json.dumps(
            {
                "id": int(DataTypeId.INTEGER),
                "byte_size": 4,
                "signed": True,
            }
        )

        out = DataType.from_str(payload)

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int32())

    def test_from_str_json_payload_decimal(self):
        payload = json.dumps(
            {
                "id": int(DataTypeId.DECIMAL),
                "byte_size": 4,
                "signed": True,
            }
        )

        out = DataType.from_str(payload)

        self.assertIsInstance(out, DecimalType)
        self.assertEqual(out.byte_size, 4)
        self.assertEqual(out.precision, 38)
        self.assertEqual(out.scale, 18)

    def test_from_str_rejects_empty_string(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            DataType.from_str("   ")

    def test_from_databricks_string_map(self):
        pa = self.pa
        parsed = DataType.from_str("MAP<STRING,STRING>")

        self.assertIsInstance(parsed, MapType)
        self.assertEqual(parsed.to_arrow(), pa.map_(pa.string(), pa.string()))
