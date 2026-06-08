"""Tests for DataType.short() / Field.short() — the recursive one-line type tag."""
from __future__ import annotations

import unittest

try:
    import pyarrow as pa

    from yggdrasil.data import DataType, Field
    _HAVE = True
except Exception:
    _HAVE = False


@unittest.skipUnless(_HAVE, "requires pyarrow + the data layer")
class TestDataTypeShort(unittest.TestCase):
    def _short(self, arrow_type) -> str:
        return DataType.from_arrow_type(arrow_type).short()

    def test_scalars(self):
        self.assertEqual(self._short(pa.int64()), "i64")
        self.assertEqual(self._short(pa.int32()), "i32")
        self.assertEqual(self._short(pa.uint16()), "u16")
        self.assertEqual(self._short(pa.float64()), "f64")
        self.assertEqual(self._short(pa.string()), "str")
        self.assertEqual(self._short(pa.bool_()), "bool")
        self.assertEqual(self._short(pa.date32()), "date")
        self.assertEqual(self._short(pa.timestamp("us")), "ts")

    def test_nested_recurse(self):
        self.assertEqual(self._short(pa.list_(pa.string())), "list<str>")
        self.assertEqual(self._short(pa.struct([("x", pa.int64()), ("y", pa.string())])),
                         "struct<x:i64, y:str>")
        self.assertEqual(self._short(pa.map_(pa.string(), pa.float64())), "map<str,f64>")
        self.assertEqual(self._short(pa.list_(pa.struct([("k", pa.int64())]))),
                         "list<struct<k:i64>>")

    def test_depth_budget_stops_recursion(self):
        deep = pa.list_(pa.list_(pa.list_(pa.int64())))
        # depth=2 → list<list<list>> (innermost goes flat).
        self.assertEqual(DataType.from_arrow_type(deep).short(depth=2), "list<list<list>>")

    def test_decimal_carries_precision_scale(self):
        self.assertEqual(self._short(pa.decimal128(10, 2)), "dec(10,2)")

    def test_dictionary_categorical_shows_value_type(self):
        self.assertEqual(self._short(pa.dictionary(pa.int8(), pa.string())), "dict<str>")
        self.assertEqual(self._short(pa.dictionary(pa.int32(), pa.int64())), "dict<i64>")

    def test_empty_struct_is_bare(self):
        self.assertEqual(self._short(pa.struct([])), "struct")

    def test_field_short_is_name_colon_dtype(self):
        f = Field(name="age", dtype=DataType.from_arrow_type(pa.int64()))
        self.assertEqual(f.short(), "age:i64")


if __name__ == "__main__":
    unittest.main()
