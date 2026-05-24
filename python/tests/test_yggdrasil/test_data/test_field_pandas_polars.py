"""``Field.from_pandas`` and ``Field.from_polars``.

Both engines expose Series, DataFrame, and dtype-class shapes.
:meth:`Field.from_pandas` and :meth:`Field.from_polars` resolve them
to a yggdrasil :class:`Field` — series-level inputs preserve the
column name; DataFrame-level inputs lift to a struct field with the
default name.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestFromPandas(PandasTestCase):

    def test_series_keeps_name_and_promotes_to_integer(self) -> None:
        series = self.pd.Series([1, 2, None], name="qty", dtype="Int64")

        out = Field.from_pandas(series)

        self.assertEqual(out.name, "qty")
        self.assertIsInstance(out.dtype, IntegerType)
        self.assertTrue(out.nullable)

    def test_dataframe_lifts_to_struct_with_default_name(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_pandas(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
        self.assertEqual(out.arrow_type.field("a").type, pa.int64())
        # Pandas 3.0+ defaults strings to StringDtype → arrow large_string.
        b_type = out.arrow_type.field("b").type
        self.assertTrue(
            pa.types.is_string(b_type) or pa.types.is_large_string(b_type),
            f"Expected string/large_string, got {b_type!r}",
        )


class TestFromPandasIndexMetadata(PandasTestCase):

    def test_named_index_tags_child_as_indexed(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.index = self.pd.Index([10, 20], name="pk")

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        self.assertIn("pk", children)
        self.assertTrue(children["pk"].indexed)
        self.assertFalse(children["a"].indexed)
        self.assertFalse(children["b"].indexed)

    def test_multi_index_tags_all_levels(self) -> None:
        idx = self.pd.MultiIndex.from_tuples(
            [("x", 1), ("y", 2)], names=["k1", "k2"],
        )
        df = self.pd.DataFrame({"v": [10, 20]}, index=idx)

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        self.assertTrue(children["k1"].indexed)
        self.assertTrue(children["k2"].indexed)
        self.assertFalse(children["v"].indexed)

    def test_default_range_index_has_no_indexed_tag(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2]})

        out = Field.from_pandas(df)

        for child in out.fields:
            self.assertFalse(child.indexed)

    def test_index_object_tagged_as_indexed(self) -> None:
        idx = self.pd.Index([10, 20, 30], name="pk")

        out = Field.from_pandas(idx)

        self.assertTrue(out.indexed)
        self.assertEqual(out.name, "pk")

    def test_unnamed_non_range_index_tags_placeholder(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2]}, index=[10, 20])

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        indexed_children = [f for f in out.fields if f.indexed]
        self.assertEqual(len(indexed_children), 1)
        self.assertFalse(children["a"].indexed)


class TestFromPolars(PolarsTestCase):

    def test_series_keeps_name_and_promotes_to_string(self) -> None:
        series = self.pl.Series("name", ["a", None, "b"])

        out = Field.from_polars(series)

        self.assertEqual(out.name, "name")
        self.assertIsInstance(out.dtype, StringType)
        self.assertTrue(out.nullable)

    def test_dataframe_lifts_to_struct_with_default_name(self) -> None:
        df = self.pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        out = Field.from_polars(df)

        self.assertEqual(out.name, DEFAULT_FIELD_NAME)
        self.assertIsInstance(out.dtype, StructType)
        self.assertFalse(out.nullable)
