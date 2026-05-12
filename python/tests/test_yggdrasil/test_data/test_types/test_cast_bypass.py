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
        opts = CastOptions(source=Field(name="x", dtype=NullType(), nullable=True))

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


# ---------------------------------------------------------------------------
# Engine-type bypass — when the yggdrasil DataType is "too precise"
# relative to the underlying engine type. Field equality reports
# "differ", engine type comparison says "same"; the cast is no-op work
# and the bypass returns the source unchanged.
# ---------------------------------------------------------------------------


class TestEngineTypeBypassArrow(ArrowTestCase):
    """SJsonType / StringType both lower to ``pa.string()`` — Field-level
    equality flags them as different (semantic dtype subclass), engine
    equality treats them as identical."""

    def _sjson(self):
        from yggdrasil.data.types.primitive.json import SJsonType
        return SJsonType()

    def test_array_passes_through_when_engine_type_matches(self) -> None:
        pa = self.pa
        arr = pa.array(["a", "b", "c"], type=pa.string())

        out = self._sjson().cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_chunked_array_passes_through_when_engine_type_matches(self) -> None:
        pa = self.pa
        arr = pa.chunked_array([["a", "b"], ["c"]], type=pa.string())

        out = self._sjson().cast_arrow_array(arr)

        self.assertIs(out, arr)

    def test_tabular_passes_through_when_engine_schema_matches(self) -> None:
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.schema import Schema

        pa = self.pa
        table = pa.table({"a": pa.array(["x", "y"], type=pa.string())})

        # Target is "string" (engine), source Field carries SJsonType
        # (yggdrasil's "string-backed json" — same arrow type). Field
        # equality says they differ; the engine-level bypass should
        # still return the table unchanged.
        target_field = Schema.from_arrow(table.schema).to_field()
        source_field = Field(
            name=target_field.name,
            dtype=Schema(inner_fields=[
                Field(name="a", dtype=self._sjson(), nullable=True),
            ]).dtype,
            nullable=target_field.nullable,
        )
        opts = CastOptions(source=source_field, target=target_field)

        out = opts.cast_arrow_tabular(table)

        self.assertIs(out, table)


class TestArrowViewFlatBypass(ArrowTestCase):
    """Arrow view layouts (``string_view`` / ``binary_view`` / list views)
    carry the same logical values as their flat counterparts. A cast
    that would force the view into the flat layout is buffer-churn for
    no semantic gain — the bypass keeps the view intact."""

    def _string_view_array(self, values):
        pa = self.pa
        # ``pa.array([...], type=pa.string_view())`` works in pyarrow ≥ 17;
        # fall back to building from string then casting via the cast
        # kernel itself when the constructor route isn't available.
        try:
            return pa.array(values, type=pa.string_view())
        except (TypeError, pa.ArrowNotImplementedError):
            return pa.array(values, type=pa.string()).cast(pa.string_view())

    def test_string_view_target_string_returns_input(self) -> None:
        if not hasattr(self.pa, "string_view"):
            self.skipTest("pyarrow build has no string_view")
        arr = self._string_view_array(["a", "b", "c"])

        out = StringType().cast_arrow_array(arr)

        self.assertIs(out, arr)
        self.assertEqual(out.type, self.pa.string_view())

    def test_string_view_chunked_target_string_returns_input(self) -> None:
        if not hasattr(self.pa, "string_view"):
            self.skipTest("pyarrow build has no string_view")
        pa = self.pa
        arr = pa.chunked_array(
            [self._string_view_array(["a", "b"]), self._string_view_array(["c"])]
        )

        out = StringType().cast_arrow_array(arr)

        self.assertIs(out, arr)
        self.assertEqual(out.type, pa.string_view())

    def test_table_with_string_view_column_passes_through(self) -> None:
        if not hasattr(self.pa, "string_view"):
            self.skipTest("pyarrow build has no string_view")
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.schema import Schema

        pa = self.pa
        table = pa.table({"a": self._string_view_array(["x", "y"])})

        # Target uses the flat ``string`` type; source carries the
        # ``string_view`` layout. Engine-level bypass should still
        # return the table unchanged (view layout preserved).
        flat_schema = pa.schema([pa.field("a", pa.string(), nullable=True)])
        target_field = Schema.from_arrow(flat_schema).to_field()
        opts = CastOptions(target=target_field).check_source(
            obj=table, copy=True,
        )

        out = opts.cast_arrow_tabular(table)

        self.assertIs(out, table)
        self.assertEqual(out.schema.field("a").type, pa.string_view())


class TestEngineTypeBypassPolars(PolarsTestCase):

    def _sjson(self):
        from yggdrasil.data.types.primitive.json import SJsonType
        return SJsonType()

    def test_series_passes_through_when_engine_type_matches(self) -> None:
        pl = self.pl
        s = pl.Series("x", ["a", "b"], dtype=pl.String)

        out = self._sjson().cast_polars_series(s)

        self.assertIs(out, s)

    def test_tabular_passes_through_when_engine_schema_matches(self) -> None:
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.schema import Schema

        pl = self.pl
        df = pl.DataFrame({"a": ["x", "y"]})

        target_field = Schema.from_polars_schema(df.schema).to_field()
        source_field = Field(
            name=target_field.name,
            dtype=Schema(inner_fields=[
                Field(name="a", dtype=self._sjson(), nullable=True),
            ]).dtype,
            nullable=target_field.nullable,
        )
        opts = CastOptions(source=source_field, target=target_field)

        out = opts.cast_polars_tabular(df)

        self.assertIs(out, df)
