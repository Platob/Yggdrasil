"""Fast-path bypass tests for DataType / Field cast_* entry points.

Covers the metadata-only short-circuits in `DataType.cast_arrow_array`,
`cast_polars_series`, `cast_pandas_series`, `cast_spark_column`, the
tabular variants, and the matching methods on `Field`:

- Same source/target dtype → return input unchanged.
- Target is ObjectType → return input unchanged without inspecting values.
- Source is NullType → reinterpret as typed-null of the target dtype without
  a deep value cast.
"""

from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.extensions.obj import ObjectType
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import (
    IntegerType,
    NullType,
    StringType,
)
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestCastArrowBypass(ArrowTestCase):

    def test_same_dtype_returns_input_unchanged(self):
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())

        out = IntegerType(byte_size=8, signed=True).cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_same_chunked_dtype_returns_input_unchanged(self):
        pa = self.pa
        arr = pa.chunked_array([[1, 2], [3]], type=pa.int64())

        out = IntegerType(byte_size=8, signed=True).cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_object_target_returns_input_unchanged(self):
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())

        out = ObjectType().cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_null_source_reinterprets_without_deep_cast(self):
        pa = self.pa
        arr = pa.nulls(4, type=pa.null())

        out = IntegerType(byte_size=8, signed=True).cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(len(out), 4)
        self.assertEqual(out.null_count, 4)

    def test_null_source_chunked_reinterprets(self):
        pa = self.pa
        arr = pa.chunked_array([pa.nulls(2, type=pa.null()), pa.nulls(3, type=pa.null())])

        out = StringType().cast_arrow_array(arr)

        self.assertEqual(out.type, pa.string())
        self.assertEqual(len(out), 5)
        self.assertEqual(out.null_count, 5)


class TestCastPolarsBypass(PolarsTestCase):

    def test_same_dtype_returns_input_unchanged(self):
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)

        out = IntegerType(byte_size=8, signed=True).cast_polars_series(s)

        self.assertIs(out, s)

    def test_object_target_returns_input_unchanged(self):
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)

        out = ObjectType().cast_polars_series(s)

        self.assertIs(out, s)

    def test_null_source_reinterprets_without_deep_cast(self):
        pl = self.pl
        s = pl.Series("x", [None, None, None], dtype=pl.Null)

        out = IntegerType(byte_size=8, signed=True).cast_polars_series(s)

        self.assertEqual(out.dtype, pl.Int64)
        self.assertEqual(out.len(), 3)
        self.assertEqual(out.null_count(), 3)


class TestCastPandasBypass(PandasTestCase):

    def test_same_dtype_returns_input_unchanged(self):
        pd = self.pd
        s = pd.Series([1, 2, 3], name="x", dtype="int64")

        out = IntegerType(byte_size=8, signed=True).cast_pandas_series(s)

        self.assertIs(out, s)

    def test_object_target_returns_input_unchanged(self):
        pd = self.pd
        s = pd.Series(["hello", "world"], name="x")

        out = ObjectType().cast_pandas_series(s)

        self.assertIs(out, s)

    def test_null_source_skips_arrow_roundtrip(self):
        pd = self.pd
        s = pd.Series([None, None, None], name="x", dtype="object")
        # Force the cast options to treat the source as NullType — this is
        # what happens when upstream metadata declares a null-typed source.
        from yggdrasil.data.cast.options import CastOptions

        src = Field(name="x", dtype=NullType(), nullable=True)
        opts = CastOptions(source_field=src)

        out = IntegerType(byte_size=8, signed=True).cast_pandas_series(s, opts)

        self.assertEqual(out.name, "x")
        self.assertEqual(len(out), 3)
        self.assertTrue(out.isna().all())


class TestCastArrowTabularBypass(ArrowTestCase):

    def test_same_schema_returns_table_unchanged(self):
        pa = self.pa
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_arrow_schema(table.schema)

        out = target.cast_arrow_tabular(table)

        self.assertIs(out, table)

    def test_object_target_returns_table_unchanged(self):
        pa = self.pa
        table = pa.table({"a": [1, 2, 3]})

        out = ObjectType().cast_arrow_tabular(table)

        self.assertIs(out, table)


class TestCastPolarsTabularBypass(PolarsTestCase):

    def test_same_schema_returns_frame_unchanged(self):
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_polars_schema(df.schema)

        out = target.cast_polars_tabular(df)

        self.assertIs(out, df)

    def test_object_target_returns_frame_unchanged(self):
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3]})

        out = ObjectType().cast_polars_tabular(df)

        self.assertIs(out, df)


class TestCastPandasTabularBypass(PandasTestCase):

    def test_same_schema_returns_frame_unchanged(self):
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_pandas(df)

        out = target.cast_pandas_tabular(df)

        self.assertIs(out, df)

    def test_object_target_returns_frame_unchanged(self):
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3]})

        out = ObjectType().cast_pandas_tabular(df)

        self.assertIs(out, df)


class TestFieldCastBypassArrow(ArrowTestCase):

    def test_object_field_skips_arrow_array_cast(self):
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_object_field_skips_arrow_tabular_cast(self):
        pa = self.pa
        table = pa.table({"a": [1, 2, 3]})
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_arrow_tabular(table)

        self.assertIs(out, table)


class TestFieldCastBypassPolars(PolarsTestCase):

    def test_object_field_skips_polars_series_cast(self):
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_polars_series(s)

        self.assertIs(out, s)

    def test_object_field_skips_polars_tabular_cast(self):
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3]})
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_polars_tabular(df)

        self.assertIs(out, df)


class TestFieldCastBypassPandas(PandasTestCase):

    def test_object_field_skips_pandas_series_cast(self):
        pd = self.pd
        s = pd.Series([1, 2, 3], name="x")
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_pandas_series(s)

        self.assertIs(out, s)

    def test_object_field_skips_pandas_tabular_cast(self):
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        field = Field(name="x", dtype=ObjectType(), nullable=True)

        out = field.cast_pandas_tabular(df)

        self.assertIs(out, df)
