"""``DataType.from_any`` and ``DataType.from_str`` dispatch.

``from_any`` is the catch-all entry point: anything you might
plausibly hand a yggdrasil API as a "type" argument — an existing
DataType, a class, an Arrow object, a string, a dict, a Python type
hint — should land on the right concrete subclass.

``from_str`` covers two surfaces: bare tokens parsed by the type
parser (``"int64"``, ``"MAP<STRING,STRING>"``) and JSON payloads
that mirror :meth:`DataType.to_dict`.
"""
from __future__ import annotations

import json

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.types import MapType
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import DecimalType, IntegerType, StringType


class TestFromAny(ArrowTestCase):

    def test_existing_dtype_passes_through_identical(self) -> None:
        src = StringType()
        out = DataType.from_any(src)

        self.assertIs(out, src)

    def test_dtype_class_returns_default_instance(self) -> None:
        pa = self.pa
        out = DataType.from_any(StringType)

        self.assertIsInstance(out, StringType)
        self.assertEqual(out.to_arrow(), pa.string())

    def test_arrow_datatype_promotes(self) -> None:
        pa = self.pa
        out = DataType.from_any(pa.int32())

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int32())

    def test_arrow_field_promotes_to_field_dtype(self) -> None:
        pa = self.pa
        out = DataType.from_any(pa.field("x", pa.int16(), nullable=False))

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int16())

    def test_arrow_schema_promotes_to_struct(self) -> None:
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

    def test_unsupported_object_raises_value_error(self) -> None:
        class Unsupported:
            pass

        with self.assertRaisesRegex(ValueError, "Cannot convert value of type"):
            DataType.from_any(Unsupported())


class TestFromStr(ArrowTestCase):

    def test_bare_token_parses_via_type_parser(self) -> None:
        pa = self.pa
        out = DataType.from_str("int64")

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int64())

    def test_json_payload_round_trips_integer(self) -> None:
        pa = self.pa
        payload = json.dumps(
            {"id": int(DataTypeId.INTEGER), "byte_size": 4, "signed": True}
        )

        out = DataType.from_str(payload)

        self.assertIsInstance(out, IntegerType)
        self.assertEqual(out.to_arrow(), pa.int32())

    def test_json_payload_round_trips_decimal_with_defaults(self) -> None:
        payload = json.dumps(
            {"id": int(DataTypeId.DECIMAL), "byte_size": 4, "signed": True}
        )

        out = DataType.from_str(payload)

        self.assertIsInstance(out, DecimalType)
        self.assertEqual(out.byte_size, 4)
        self.assertEqual(out.precision, 38)
        self.assertEqual(out.scale, 18)

    def test_empty_or_blank_string_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            DataType.from_str("   ")

    def test_databricks_map_token(self) -> None:
        pa = self.pa
        parsed = DataType.from_str("MAP<STRING,STRING>")

        self.assertIsInstance(parsed, MapType)
        self.assertEqual(parsed.to_arrow(), pa.map_(pa.string(), pa.string()))

    def test_databricks_struct_token(self) -> None:
        pa = self.pa
        parsed = DataType.from_str("STRUCT<q: TIMESTAMP, v: DOUBLE>")

        self.assertIsInstance(parsed, StructType)
        self.assertEqual(
            parsed.to_arrow(),
            pa.struct(
                [
                    pa.field("q", pa.timestamp("us", "Etc/UTC")),
                    pa.field("v", pa.float64()),
                ]
            ),
        )
