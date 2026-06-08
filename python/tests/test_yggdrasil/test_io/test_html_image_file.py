"""Tests for the HTML and image Tabular IO leaves."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import polars  # noqa: F401
    import pyarrow  # noqa: F401

    from yggdrasil.io.holder import IO

    _HAVE_STACK = True
except Exception:  # pragma: no cover
    _HAVE_STACK = False


@unittest.skipUnless(_HAVE_STACK, "requires the polars/pyarrow stack")
class TestHtmlImageLeaves(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="ygg-leaf-"))

    def test_html_table_parses_to_frame(self):
        (self.dir / "t.html").write_text(
            "<html><body><table><tr><th>a</th><th>b</th></tr>"
            "<tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr>"
            "</table></body></html>"
        )
        df = IO.from_(str(self.dir / "t.html")).to_polars()
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(df["b"].sum(), 6)

    def test_html_without_table_degrades_to_text(self):
        (self.dir / "p.html").write_text("<html><body><h1>Hi</h1><p>No tables.</p></body></html>")
        rows = IO.from_(str(self.dir / "p.html")).to_polars().to_dicts()
        self.assertEqual(list(rows[0].keys()), ["text"])
        self.assertIn("No tables", rows[0]["text"])

    def test_html_roundtrip_write_read(self):
        import polars as pl

        src = self.dir / "out.html"
        leaf = IO.from_(str(src))
        leaf.write_polars_frame(pl.DataFrame({"x": [10, 20], "y": ["a", "b"]}))
        back = IO.from_(str(src)).to_polars()
        self.assertEqual(back.shape, (2, 2))
        self.assertEqual(set(back.columns), {"x", "y"})

    def test_png_metadata_projection(self):
        from PIL import Image

        Image.new("RGB", (64, 48), "red").save(self.dir / "x.png")
        row = IO.from_(str(self.dir / "x.png")).to_polars().to_dicts()[0]
        self.assertEqual(row["format"], "PNG")
        self.assertEqual((row["width"], row["height"]), (64, 48))
        self.assertEqual(row["mode"], "RGB")
        self.assertGreater(row["bytes"], 0)

    def test_jpeg_uses_same_leaf_via_alias(self):
        from PIL import Image

        from yggdrasil.io.image_file import ImageFile
        from yggdrasil.io.holder import _HOLDER_FORMAT_REGISTRY
        from yggdrasil.enums import MimeTypes

        self.assertIs(_HOLDER_FORMAT_REGISTRY[MimeTypes.JPEG.name], ImageFile)
        Image.new("L", (10, 20)).save(self.dir / "y.jpg")
        row = IO.from_(str(self.dir / "y.jpg")).to_polars().to_dicts()[0]
        self.assertEqual(row["format"], "JPEG")
        self.assertEqual((row["width"], row["height"]), (10, 20))


if __name__ == "__main__":
    unittest.main()


class TestTabularDisplay(unittest.TestCase):
    """Tabular.display() renders a typed, aligned, delimited first-n-rows preview."""

    def test_typed_header_aligned_columns(self):
        try:
            import polars  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "d.csv"
        p.write_text("city,pop\nParis,2161\nTokyo,13960\n")
        lines = IO.from_(str(p)).display().splitlines()
        # Two-row header: names, then type tags, then a │-delimited rule.
        self.assertIn("city", lines[0])
        self.assertIn("pop", lines[0])
        self.assertIn("str", lines[1])
        self.assertIn("i64", lines[1])
        self.assertIn("│", lines[0])
        self.assertTrue(set(lines[2]) <= {"─", "┼"})           # the rule row
        # Numeric columns right-align: the name, the type, and every value in the
        # `pop` column share a right edge (last column → each line ends with it).
        self.assertEqual(len({len(line) for line in lines[:5]}), 1)   # uniform width
        self.assertTrue(lines[0].rstrip().endswith("pop"))
        self.assertTrue(lines[1].rstrip().endswith("i64"))
        self.assertTrue(lines[3].rstrip().endswith("2161"))
        self.assertTrue(lines[4].rstrip().endswith("13960"))
        # …while the text column stays left-aligned.
        self.assertTrue(lines[3].startswith("Paris"))

    def test_long_values_are_clipped_not_ballooned(self):
        try:
            import polars  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "d.csv"
        p.write_text("id,token\n1," + "x" * 200 + "\n")        # a 200-char secret
        lines = IO.from_(str(p)).display(max_width=24).splitlines()
        # No line balloons: every column (and the long value) is bounded.
        self.assertTrue(all(len(line) < 60 for line in lines), [len(line_) for line_ in lines])
        self.assertIn("…", "\n".join(lines))                   # the value was clipped

    def test_closing_rule_footer_and_nulls(self):
        try:
            import polars  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "d.csv"
        p.write_text("city,pop\nParis,2161\nTokyo,\n")   # Tokyo pop is null
        lines = IO.from_(str(p)).display().splitlines()
        # A closing rule (┴) and a shape footer round out the table…
        self.assertTrue(set(lines[-2]) <= {"─", "┴"})
        self.assertEqual(lines[-1], "2 rows × 2 cols")
        # …and a null renders as a clear dot, not blank.
        self.assertIn("·", "\n".join(lines))

    def test_nested_values_are_compacted(self):
        try:
            import polars as pl
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "n.parquet"
        pl.DataFrame({"id": [1], "tags": [["a", "b"]], "meta": [{"x": 1}]}).write_parquet(p)
        lines = IO.from_(str(p)).display().splitlines()
        # Nested columns get a RECURSIVE, bounded type tag on the types row …
        self.assertIn("list<str>", lines[1])
        self.assertIn("struct<x:i64>", lines[1])
        # … and nested values render compactly on one line (first data row).
        self.assertIn('["a","b"]', lines[3])
        self.assertIn('{"x":1}', lines[3])

    def test_nested_type_tags_recurse_and_are_bounded(self):
        try:
            import polars as pl
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "deep.parquet"
        pl.DataFrame({"rows": [[{"k": 1, "v": "a"}]]}).write_parquet(p)
        types_row = IO.from_(str(p)).display().splitlines()[1]   # the type-tags row
        # list<struct<…>> — the recursion shows nested shape, not a flat "list".
        self.assertIn("list<struct<k:i64, v:str>>", types_row)

    def test_deep_type_tag_is_elided(self):
        try:
            import pyarrow as pa
        except Exception:
            self.skipTest("pyarrow not installed")
        # The short tag lives on the project DataType (not engine code), and a
        # wide struct is capped (depth + field cap + length elision).
        from yggdrasil.data import DataType
        from yggdrasil.data.types.base import _SHORT_TAG_MAX

        wide = DataType.from_arrow_type(
            pa.struct([(f"field_number_{i}", pa.int64()) for i in range(10)]))
        tag = wide.short()
        self.assertLessEqual(len(tag), _SHORT_TAG_MAX)
        self.assertTrue(tag.startswith("struct<"))

    def test_limit_marker(self):
        try:
            import polars  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        import tempfile
        from pathlib import Path

        from yggdrasil.io.holder import IO

        p = Path(tempfile.mkdtemp()) / "d.csv"
        p.write_text("n\n" + "\n".join(str(i) for i in range(100)) + "\n")
        out = IO.from_(str(p)).display(5)
        self.assertIn("first 5 rows", out)

    def test_schema_markers_in_type_row(self):
        try:
            import pyarrow as pa
        except Exception:
            self.skipTest("pyarrow not installed")
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.data import DataType, Field, Schema

        # A primary-key, non-nullable id and a partition column — their main
        # schema markers ride along on the type-tags row (PK / * / partition).
        # Built straight from the project schema, displayed in memory.
        f_id = Field(name="id", dtype=DataType.from_arrow_type(pa.int64()),
                     nullable=False, tags={"primary_key": True})
        f_dt = Field(name="dt", dtype=DataType.from_arrow_type(pa.int32()),
                     tags={"partition_by": True})
        f_city = Field(name="city", dtype=DataType.from_arrow_type(pa.string()))
        schema = Schema.from_fields([f_id, f_dt, f_city])
        tbl = pa.table({"id": [1, 2], "dt": [10, 20], "city": ["Paris", "Tokyo"]},
                       schema=schema.to_arrow_schema())
        types_row = ArrowTabular(tbl).display().splitlines()[1]
        self.assertIn("i64 PK *", types_row)        # key + required marker
        self.assertIn("partition", types_row)        # layout marker
        # …and an unmarked column carries only its bare type tag.
        self.assertIn("str", types_row)

    def test_wide_glyphs_align_by_display_width(self):
        try:
            import pyarrow as pa
        except Exception:
            self.skipTest("pyarrow not installed")
        from yggdrasil.arrow.tabular import ArrowTabular

        # CJK glyphs occupy two terminal columns each: "東京" (2 code points,
        # 4 display columns) must align with "Paris" (5), not on len().
        tbl = pa.table({"city": ["Paris", "東京"], "n": [1, 2]})
        lines = ArrowTabular(tbl).display().splitlines()
        # Every rendered row shares one display width (the alignment invariant),
        # even though the byte/char lengths differ because of the wide glyphs.
        import unicodedata

        def dwidth(s: str) -> int:
            return sum(0 if unicodedata.combining(c)
                       else 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
                       for c in s)

        rule = lines[2]
        self.assertEqual({dwidth(line) for line in lines[:5]}, {dwidth(rule)})
