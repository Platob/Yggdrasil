"""End-to-end ``DataType`` cast / fill exercises across engines.

The struct of this file mirrors the engine surface ``DataType`` exposes:

* :class:`TestArrowIntegerCast` — ``cast_arrow_array`` /
  ``fill_arrow_array_nulls`` / ``default_arrow_array`` against an
  :class:`IntegerType` target.
* :class:`TestPandasIntegerCast` — pandas ``Series`` cast / fill.
* :class:`TestPolarsIntegerCast` — polars ``Series`` cast / fill,
  including the path that goes through ``Field.cast_polars_series``
  (used by callers that have a target field rather than a bare type).

The Arrow assertions also lock in two empty-coercion contracts that
production code relies on: empty strings and empty bytes both null
out before the integer cast runs (so a ragged CSV column doesn't
explode under pyarrow's ``safe=True`` kernel).
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.primitive import IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


_INT64 = IntegerType(byte_size=8, signed=True)


# ---------------------------------------------------------------------------
# Arrow
# ---------------------------------------------------------------------------


class TestArrowIntegerCast(ArrowTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dtype = _INT64

    def test_from_pytype_optional_int_resolves_to_int64(self) -> None:
        pa = self.pa
        dtype = DataType.from_pytype(int | None)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 8)
        self.assertTrue(dtype.signed)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_default_arrow_array_non_nullable_fills_zero(self) -> None:
        pa = self.pa
        arr = self.dtype.default_arrow_array(nullable=False, size=3)

        self.assertEqual(arr.type, pa.int64())
        self.assertEqual(arr.to_pylist(), [0, 0, 0])

    def test_fill_arrow_array_nulls_replaces_with_zero_when_non_nullable(self) -> None:
        pa = self.pa
        arr = pa.array([1, None, 3], type=pa.int64())

        out = self.dtype.fill_arrow_array_nulls(arr, nullable=False)

        self.assertEqual(out.to_pylist(), [1, 0, 3])

    def test_cast_arrow_array_widens_int32_to_int64(self) -> None:
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int32())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, 3])

    def test_cast_arrow_array_string_to_int_nullifies_empty(self) -> None:
        pa = self.pa
        arr = pa.array(["1", "2", "", "3", None], type=pa.string())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, None, 3, None])

    def test_cast_arrow_array_binary_to_int_nullifies_empty(self) -> None:
        pa = self.pa
        arr = pa.array([b"1", b"2", b"", b"3", None], type=pa.binary())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, None, 3, None])

    def test_cast_arrow_chunked_array_preserves_chunk_layout(self) -> None:
        pa = self.pa
        arr = pa.chunked_array(
            [
                pa.array(["1", "", "3"], type=pa.string()),
                pa.array(["", "5", None], type=pa.string()),
            ]
        )

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, None, 3, None, 5, None])

    def test_cast_arrow_array_float_to_int_truncates(self) -> None:
        pa = self.pa
        arr = pa.array([1.0, 0.0, 3.0], type=pa.float64())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 0, 3])

    def test_string_target_large_string_source_drops_view_and_keeps_empty(self) -> None:
        pa = self.pa
        arr = pa.array(["a", "", "b", None], type=pa.large_string())

        out = StringType().cast_arrow_array(arr)

        self.assertEqual(out.type, pa.string())
        self.assertEqual(out.to_pylist(), ["a", "", "b", None])


# ---------------------------------------------------------------------------
# Pandas
# ---------------------------------------------------------------------------


class TestPandasIntegerCast(PandasTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dtype = _INT64

    def test_cast_pandas_series_keeps_values(self) -> None:
        series = self.pd.Series([1, 2, 3], name="x")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(out.tolist(), [1, 2, 3])
        self.assertEqual(out.name, "x")

    def test_cast_pandas_series_preserves_name(self) -> None:
        series = self.pd.Series([10, 20], name="my_col")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(out.name, "my_col")

    def test_fill_pandas_series_nulls_with_zero_when_non_nullable(self) -> None:
        series = self.pd.Series([1, None, 3], name="x")

        out = self.dtype.fill_pandas_series_nulls(series, nullable=False)

        self.assertEqual(out.tolist(), [1.0, 0.0, 3.0])

    def test_cast_pandas_string_to_int_nullifies_empty_token(self) -> None:
        series = self.pd.Series(["1", "2", "", "3"], name="x")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(out.name, "x")
        self.assertEqual(out.tolist()[:2], [1, 2])
        self.assertEqual(out.tolist()[3], 3)
        self.assertTrue(self.pd.isna(out.iloc[2]))


# ---------------------------------------------------------------------------
# Polars
# ---------------------------------------------------------------------------


class TestPolarsIntegerCast(PolarsTestCase):

    def setUp(self) -> None:
        super().setUp()
        self.dtype = _INT64

    def test_cast_polars_series_keeps_values(self) -> None:
        series = self.pl.Series("x", [1, 2, 3])

        out = self.dtype.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, 3])
        self.assertEqual(out.name, "x")

    def test_cast_polars_series_preserves_name(self) -> None:
        series = self.pl.Series("my_col", [10, 20])

        out = self.dtype.cast_polars_series(series)

        self.assertEqual(out.name, "my_col")

    def test_fill_polars_series_nulls_with_zero_when_non_nullable(self) -> None:
        series = self.pl.Series("x", [1, None, 3])

        out = self.dtype.fill_polars_array_nulls(series, nullable=False)

        self.assertEqual(out.to_list(), [1, 0, 3])

    def test_cast_polars_tabular_scalar_dtype_round_trips(self) -> None:
        df = self.pl.DataFrame({DEFAULT_FIELD_NAME: [1, 2, 3]})

        # Scalar dtype lifts into a single-column struct internally — the
        # call must not raise on a flat frame.
        self.dtype.cast_polars_tabular(df)

    def test_cast_polars_string_to_int_via_field_nullifies_empty(self) -> None:
        target = Field(name="x", dtype=self.dtype, nullable=True)
        series = self.pl.Series("x", ["1", "2", "", "3", None])

        out = target.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, None, 3, None])
        self.assertEqual(out.dtype, self.pl.Int64)

    def test_cast_polars_binary_to_int_via_field_nullifies_empty(self) -> None:
        target = Field(name="x", dtype=self.dtype, nullable=True)
        series = self.pl.Series(
            "x", [b"1", b"2", b"", b"3", None], dtype=self.pl.Binary
        )

        out = target.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, None, 3, None])
        self.assertEqual(out.dtype, self.pl.Int64)
