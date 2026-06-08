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

    def test_named_index_tags_child_as_index_key(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.index = self.pd.Index([10, 20], name="pk")

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        self.assertIn("pk", children)
        self.assertTrue(children["pk"].index_key)
        self.assertEqual(children["pk"].index_key_level, 0)
        self.assertFalse(children["a"].index_key)
        self.assertFalse(children["b"].index_key)

    def test_multi_index_tags_all_levels(self) -> None:
        idx = self.pd.MultiIndex.from_tuples(
            [("x", 1), ("y", 2)], names=["k1", "k2"],
        )
        df = self.pd.DataFrame({"v": [10, 20]}, index=idx)

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        self.assertTrue(children["k1"].index_key)
        self.assertEqual(children["k1"].index_key_level, 0)
        self.assertTrue(children["k2"].index_key)
        self.assertEqual(children["k2"].index_key_level, 1)
        self.assertFalse(children["v"].index_key)

    def test_default_range_index_has_no_index_key_tag(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2]})

        out = Field.from_pandas(df)

        for child in out.fields:
            self.assertFalse(child.index_key)

    def test_index_object_tagged_as_index_key(self) -> None:
        idx = self.pd.Index([10, 20, 30], name="pk")

        out = Field.from_pandas(idx)

        self.assertTrue(out.index_key)
        self.assertEqual(out.name, "pk")

    def test_unnamed_non_range_index_tags_placeholder(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2]}, index=[10, 20])

        out = Field.from_pandas(df)

        children = {f.name: f for f in out.fields}
        index_children = [f for f in out.fields if f.index_key]
        self.assertEqual(len(index_children), 1)
        self.assertFalse(children["a"].index_key)

    def test_check_pandas_indexes_promotes_index_columns(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2], "pk": [10, 20], "v": [3, 4]})
        field = Field.from_pandas(df)
        for child in field.fields:
            if child.name == "pk":
                child.with_index_key(True, level=0, inplace=True)

        result = field.check_pandas_indexes(df)

        self.assertEqual(result.index.name, "pk")
        self.assertEqual(list(result.index), [10, 20])
        self.assertNotIn("pk", result.columns)

    def test_check_pandas_metadata_from_schema(self) -> None:
        df = self.pd.DataFrame({"v": [10, 20]}, index=self.pd.Index([1, 2], name="pk"))
        table = pa.Table.from_pandas(df)
        field = Field.from_arrow_schema(table.schema.remove_metadata())

        result = field.check_pandas_metadata(table.schema)

        children = {f.name: f for f in result.fields}
        self.assertTrue(children["pk"].index_key)
        self.assertEqual(children["pk"].index_key_level, 0)
        self.assertFalse(children["v"].index_key)

    def test_check_pandas_metadata_from_table(self) -> None:
        df = self.pd.DataFrame({"v": [1]}, index=self.pd.Index([7], name="pk"))
        table = pa.Table.from_pandas(df)
        field = Field.from_arrow_schema(table.schema.remove_metadata())

        result = field.check_pandas_metadata(table)

        self.assertTrue({f.name: f for f in result.fields}["pk"].index_key)

    def test_check_pandas_metadata_from_dict_and_bytes(self) -> None:
        import yggdrasil.pickle.json as ygg_json

        df = self.pd.DataFrame({"v": [1, 2]}, index=self.pd.Index([3, 4], name="pk"))
        table = pa.Table.from_pandas(df)
        blob = table.schema.metadata[b"pandas"]
        parsed = ygg_json.loads(blob)

        # dict shape
        field = Field.from_arrow_schema(table.schema.remove_metadata())
        out_dict = field.check_pandas_metadata(parsed)
        self.assertTrue({f.name: f for f in out_dict.fields}["pk"].index_key)

        # raw bytes shape
        field2 = Field.from_arrow_schema(table.schema.remove_metadata())
        out_bytes = field2.check_pandas_metadata(blob)
        self.assertTrue({f.name: f for f in out_bytes.fields}["pk"].index_key)

    def test_check_pandas_metadata_falls_back_to_self_metadata(self) -> None:
        df = self.pd.DataFrame({"v": [9]}, index=self.pd.Index([2], name="pk"))
        table = pa.Table.from_pandas(df)
        # from_arrow_schema preserves the b"pandas" key onto self.metadata,
        # so no source arg means it should still find the index columns.
        field = Field.from_arrow_schema(table.schema)

        result = field.check_pandas_metadata()

        self.assertTrue({f.name: f for f in result.fields}["pk"].index_key)

    def test_check_pandas_metadata_noop_without_blob(self) -> None:
        df = self.pd.DataFrame({"a": [1, 2]})
        field = Field.from_arrow_schema(pa.Table.from_pandas(df).schema.remove_metadata())

        result = field.check_pandas_metadata(None)

        self.assertIs(result, field)
        self.assertFalse(any(f.index_key for f in result.fields))


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
