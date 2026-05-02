"""Cast fast-paths — when does ``cast_*`` skip the actual cast?

Three short-circuits are pinned for every engine:

* **Same dtype** — source matches target, no cast performed; the
  caller's array / series / frame is returned by identity.
* **ObjectType target** — variant target is a structural no-op;
  values pass through untouched regardless of source dtype.
* **NullType source** — Arrow / Polars nulls reinterpret as a typed
  null at the target dtype, no values are touched.

The same contract is tested at three levels: bare ``DataType.cast_*``
methods, ``Field.cast_*`` (Object-field skip), and the tabular
variants on ``StructType``.
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import (
    IntegerType,
    NullType,
    ObjectType,
    StringType,
)
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


_INT64 = IntegerType(byte_size=8, signed=True)


# ---------------------------------------------------------------------------
# DataType.cast_arrow_array
# ---------------------------------------------------------------------------


class TestArrowArrayBypass(ArrowTestCase):

    def test_same_dtype_returns_input_identity(self) -> None:
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())

        out = _INT64.cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_same_chunked_dtype_returns_input_identity(self) -> None:
        pa = self.pa
        arr = pa.chunked_array([[1, 2], [3]], type=pa.int64())

        out = _INT64.cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_object_target_skips_cast(self) -> None:
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())

        out = ObjectType().cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_null_source_reinterprets_as_typed_null(self) -> None:
        pa = self.pa
        arr = pa.nulls(4, type=pa.null())

        out = _INT64.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(len(out), 4)
        self.assertEqual(out.null_count, 4)

    def test_null_chunked_source_reinterprets(self) -> None:
        pa = self.pa
        arr = pa.chunked_array(
            [pa.nulls(2, type=pa.null()), pa.nulls(3, type=pa.null())]
        )

        out = StringType().cast_arrow_array(arr)

        self.assertEqual(out.type, pa.string())
        self.assertEqual(len(out), 5)
        self.assertEqual(out.null_count, 5)


# ---------------------------------------------------------------------------
# DataType.cast_polars_series
# ---------------------------------------------------------------------------


class TestPolarsSeriesBypass(PolarsTestCase):

    def test_same_dtype_returns_input_identity(self) -> None:
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)

        out = _INT64.cast_polars_series(s)

        self.assertIs(out, s)

    def test_object_target_skips_cast(self) -> None:
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)

        out = ObjectType().cast_polars_series(s)

        self.assertIs(out, s)

    def test_null_source_reinterprets_as_typed_null(self) -> None:
        pl = self.pl
        s = pl.Series("x", [None, None, None], dtype=pl.Null)

        out = _INT64.cast_polars_series(s)

        self.assertEqual(out.dtype, pl.Int64)
        self.assertEqual(out.len(), 3)
        self.assertEqual(out.null_count(), 3)


# ---------------------------------------------------------------------------
# DataType.cast_pandas_series
# ---------------------------------------------------------------------------


class TestPandasSeriesBypass(PandasTestCase):

    def test_same_dtype_returns_input_identity(self) -> None:
        pd = self.pd
        s = pd.Series([1, 2, 3], name="x", dtype="int64")

        out = _INT64.cast_pandas_series(s)

        self.assertIs(out, s)

    def test_object_target_skips_cast(self) -> None:
        pd = self.pd
        s = pd.Series(["hello", "world"], name="x")

        out = ObjectType().cast_pandas_series(s)

        self.assertIs(out, s)

    def test_null_source_avoids_arrow_round_trip(self) -> None:
        pd = self.pd
        from yggdrasil.data.options import CastOptions

        # Force the cast to treat the source as NullType (this happens when
        # upstream metadata declares a null-typed source).
        s = pd.Series([None, None, None], name="x", dtype="object")
        opts = CastOptions(source_field=Field(name="x", dtype=NullType(), nullable=True))

        out = _INT64.cast_pandas_series(s, opts)

        self.assertEqual(out.name, "x")
        self.assertEqual(len(out), 3)
        self.assertTrue(out.isna().all())


# ---------------------------------------------------------------------------
# Tabular variants — table / frame returned untouched.
# ---------------------------------------------------------------------------


class TestArrowTabularBypass(ArrowTestCase):

    def test_same_schema_table_passes_through(self) -> None:
        pa = self.pa
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_arrow_schema(table.schema)

        self.assertIs(target.cast_arrow_tabular(table), table)

    def test_object_target_table_passes_through(self) -> None:
        pa = self.pa
        table = pa.table({"a": [1, 2, 3]})

        self.assertIs(ObjectType().cast_arrow_tabular(table), table)


class TestPolarsTabularBypass(PolarsTestCase):

    def test_same_schema_frame_passes_through(self) -> None:
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_polars_schema(df.schema)

        self.assertIs(target.cast_polars_tabular(df), df)

    def test_object_target_frame_passes_through(self) -> None:
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3]})

        self.assertIs(ObjectType().cast_polars_tabular(df), df)


class TestPandasTabularBypass(PandasTestCase):

    def test_same_schema_frame_passes_through(self) -> None:
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        target = StructType.from_pandas(df)

        self.assertIs(target.cast_pandas_tabular(df), df)

    def test_object_target_frame_passes_through(self) -> None:
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3]})

        self.assertIs(ObjectType().cast_pandas_tabular(df), df)


# ---------------------------------------------------------------------------
# Field-level Object skip — same contract, but at the Field surface.
# ---------------------------------------------------------------------------


class TestFieldObjectSkipArrow(ArrowTestCase):

    def test_arrow_array_passthrough(self) -> None:
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int64())
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_arrow_array(arr), arr)

    def test_arrow_table_passthrough(self) -> None:
        pa = self.pa
        table = pa.table({"a": [1, 2, 3]})
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_arrow_tabular(table), table)


class TestFieldObjectSkipPolars(PolarsTestCase):

    def test_polars_series_passthrough(self) -> None:
        pl = self.pl
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_polars_series(s), s)

    def test_polars_frame_passthrough(self) -> None:
        pl = self.pl
        df = pl.DataFrame({"a": [1, 2, 3]})
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_polars_tabular(df), df)


class TestFieldObjectSkipPandas(PandasTestCase):

    def test_pandas_series_passthrough(self) -> None:
        pd = self.pd
        s = pd.Series([1, 2, 3], name="x")
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_pandas_series(s), s)

    def test_pandas_frame_passthrough(self) -> None:
        pd = self.pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        f = Field(name="x", dtype=ObjectType(), nullable=True)

        self.assertIs(f.cast_pandas_tabular(df), df)
