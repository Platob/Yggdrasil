"""Tests for yggdrasil.arrow.arrow_display — the Arrow text-table preview."""
from __future__ import annotations

import unittest

try:
    import pyarrow as pa

    from yggdrasil.arrow import arrow_display
    from yggdrasil.data import DataType, Field, Schema
    _HAVE = True
except Exception:
    _HAVE = False


@unittest.skipUnless(_HAVE, "requires pyarrow + the data layer")
class TestArrowDisplay(unittest.TestCase):
    def test_schema_derived_when_omitted(self):
        # No schema passed → derived from the Arrow schema: a two-row header
        # (names, then type tags), a │-delimited rule, and a shape footer.
        out = arrow_display(pa.table({"n": [1, 2], "label": ["a", "b"]}))
        lines = out.splitlines()
        self.assertIn("n", lines[0])
        self.assertIn("label", lines[0])
        self.assertIn("i64", lines[1])
        self.assertIn("str", lines[1])
        self.assertTrue(set(lines[2]) <= {"─", "┼"})
        self.assertEqual(lines[-1], "2 rows × 2 cols")

    def test_markers_ride_the_type_row(self):
        # A passed project schema carries markers onto the type row (PK / * /
        # partition); the column order follows the Arrow columns positionally.
        f_id = Field(name="id", dtype=DataType.from_arrow_type(pa.int64()),
                     nullable=False, tags={"primary_key": True})
        f_dt = Field(name="dt", dtype=DataType.from_arrow_type(pa.int32()),
                     tags={"partition_by": True})
        f_city = Field(name="city", dtype=DataType.from_arrow_type(pa.string()))
        schema = Schema.from_fields([f_id, f_dt, f_city])
        tbl = pa.table({"id": [1], "dt": [10], "city": ["Paris"]},
                       schema=schema.to_arrow_schema())
        types_row = arrow_display(tbl, schema).splitlines()[1]
        self.assertIn("i64 PK *", types_row)
        self.assertIn("partition", types_row)

    def test_numeric_right_aligns_text_left(self):
        lines = arrow_display(pa.table({"city": ["Paris"], "pop": [2161]})).splitlines()
        self.assertTrue(lines[0].rstrip().endswith("pop"))     # numeric → right
        self.assertTrue(lines[3].rstrip().endswith("2161"))
        self.assertTrue(lines[3].startswith("Paris"))           # text → left

    def test_over_n_marks_truncation_exactly_n_shows_shape(self):
        self.assertEqual(
            arrow_display(pa.table({"n": list(range(10))}), n=3).splitlines()[-1],
            "… (first 3 rows)")
        self.assertEqual(
            arrow_display(pa.table({"n": [1, 2, 3]}), n=3).splitlines()[-1],
            "3 rows × 1 col")

    def test_nested_values_compact_and_nulls_dot(self):
        tbl = pa.table({"tags": [["a", "b"]], "meta": [{"x": 1}], "v": [None]})
        lines = arrow_display(tbl).splitlines()
        self.assertIn("list<str>", lines[1])
        self.assertIn("struct<x:i64>", lines[1])
        self.assertIn('["a","b"]', lines[3])
        self.assertIn('{"x":1}', lines[3])
        self.assertIn("·", lines[3])

    def test_long_values_clipped_and_wide_glyphs_align(self):
        import unicodedata

        def dwidth(s: str) -> int:
            return sum(0 if unicodedata.combining(c)
                       else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
                       for c in s)

        tbl = pa.table({"city": ["Paris", "東京"], "token": ["x" * 200, "y"]})
        lines = arrow_display(tbl, max_width=24).splitlines()
        self.assertIn("…", "\n".join(lines))                    # clipped
        self.assertTrue(all(len(line) < 60 for line in lines))  # never balloons
        rule = lines[2]                                          # display widths align
        self.assertEqual({dwidth(line) for line in lines[:4]}, {dwidth(rule)})


if __name__ == "__main__":
    unittest.main()
