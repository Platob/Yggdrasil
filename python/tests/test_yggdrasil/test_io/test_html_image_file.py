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
        # Header carries short data types and a │ column delimiter.
        self.assertIn("city:str", lines[0])
        self.assertIn("pop:i64", lines[0])
        self.assertIn("│", lines[0])
        # The rule row uses the box-drawing rule, aligned to the columns.
        self.assertTrue(set(lines[1]) <= {"─", "┼"})
        # Columns align: the 2nd column header and its values share an offset.
        self.assertEqual(lines[0].index("pop:i64"), lines[2].index("2161"))

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
        # Nested columns get a flat type tag (not the whole inner schema) …
        self.assertIn("tags:list", lines[0])
        self.assertIn("meta:struct", lines[0])
        # … and nested values render compactly on one line.
        self.assertIn('["a","b"]', lines[2])
        self.assertIn('{"x":1}', lines[2])

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
