from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.primitive import BinaryType, IntegerType, StringType
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class _IntegerCastFillMixin:
    """Shared setup for IntegerType cast/fill tests across engines."""

    def _init_dtype(self):
        self.dtype = IntegerType(byte_size=8, signed=True)


class TestDataTypeArrow(_IntegerCastFillMixin, ArrowTestCase):

    def setUp(self):
        super().setUp()
        self._init_dtype()

    def test_from_pytype_optional_int_returns_integer_type(self):
        pa = self.pa
        dtype = DataType.from_pytype(int | None)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.byte_size, 8)
        self.assertTrue(dtype.signed)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_default_arrow_array_non_nullable_int(self):
        pa = self.pa
        arr = self.dtype.default_arrow_array(nullable=False, size=3)

        self.assertEqual(arr.to_pylist(), [0, 0, 0])
        self.assertEqual(arr.type, pa.int64())

    def test_fill_arrow_array_nulls_non_nullable_int(self):
        pa = self.pa
        arr = pa.array([1, None, 3], type=pa.int64())

        out = self.dtype.fill_arrow_array_nulls(arr, nullable=False)

        self.assertEqual(out.to_pylist(), [1, 0, 3])

    def test_cast_arrow_array_int(self):
        pa = self.pa
        arr = pa.array([1, 2, 3], type=pa.int32())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, 3])

    def test_cast_arrow_array_string_to_int_nullifies_empty(self):
        pa = self.pa
        arr = pa.array(["1", "2", "", "3", None], type=pa.string())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, None, 3, None])

    def test_cast_arrow_array_binary_to_int_nullifies_empty(self):
        pa = self.pa
        arr = pa.array([b"1", b"2", b"", b"3", None], type=pa.binary())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 2, None, 3, None])

    def test_cast_arrow_array_large_string_to_string_nullifies_empty(self):
        pa = self.pa
        arr = pa.array(["a", "", "b", None], type=pa.large_string())

        out = StringType().cast_arrow_array(arr)

        self.assertEqual(out.type, pa.string())
        self.assertEqual(out.to_pylist(), ["a", None, "b", None])

    def test_cast_arrow_chunked_array_string_to_int_nullifies_empty(self):
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

    def test_cast_arrow_array_non_string_source_untouched(self):
        pa = self.pa
        arr = pa.array([1.0, 0.0, 3.0], type=pa.float64())

        out = self.dtype.cast_arrow_array(arr)

        self.assertEqual(out.type, pa.int64())
        self.assertEqual(out.to_pylist(), [1, 0, 3])


class TestDataTypePandas(_IntegerCastFillMixin, PandasTestCase):

    def setUp(self):
        super().setUp()
        self._init_dtype()

    def test_cast_pandas_series_to_int(self):
        series = self.pd.Series([1, 2, 3], name="x")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(list(out.tolist()), [1, 2, 3])
        self.assertEqual(out.name, "x")

    def test_fill_pandas_series_nulls_non_nullable_int(self):
        series = self.pd.Series([1, None, 3], name="x")

        out = self.dtype.fill_pandas_series_nulls(series, nullable=False)

        self.assertEqual(out.tolist(), [1.0, 0.0, 3.0])

    def test_cast_pandas_series_preserves_name(self):
        series = self.pd.Series([10, 20], name="my_col")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(out.name, "my_col")

    def test_cast_pandas_series_string_to_int_nullifies_empty(self):
        series = self.pd.Series(["1", "2", "", "3"], name="x")

        out = self.dtype.cast_pandas_series(series)

        self.assertEqual(out.name, "x")
        self.assertEqual(out.tolist()[:2], [1, 2])
        self.assertEqual(out.tolist()[3], 3)
        self.assertTrue(self.pd.isna(out.iloc[2]))


class TestDataTypePolars(_IntegerCastFillMixin, PolarsTestCase):

    def setUp(self):
        super().setUp()
        self._init_dtype()

    def test_cast_polars_series_to_int(self):
        series = self.pl.Series("x", [1, 2, 3])

        out = self.dtype.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, 3])
        self.assertEqual(out.name, "x")

    def test_fill_polars_series_nulls_non_nullable_int(self):
        series = self.pl.Series("x", [1, None, 3])

        out = self.dtype.fill_polars_array_nulls(series, nullable=False)

        self.assertEqual(out.to_list(), [1, 0, 3])

    def test_cast_polars_tabular_scalar_dtype_delegates_to_struct(self):
        df = self.pl.DataFrame({DEFAULT_FIELD_NAME: [1, 2, 3]})

        self.dtype.cast_polars_tabular(df)

    def test_cast_polars_series_preserves_name(self):
        series = self.pl.Series("my_col", [10, 20])

        out = self.dtype.cast_polars_series(series)

        self.assertEqual(out.name, "my_col")

    def test_cast_polars_series_string_to_int_nullifies_empty(self):
        target = Field(name="x", dtype=self.dtype, nullable=True)
        series = self.pl.Series("x", ["1", "2", "", "3", None])

        out = target.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, None, 3, None])
        self.assertEqual(out.dtype, self.pl.Int64)

    def test_cast_polars_series_binary_to_int_nullifies_empty(self):
        target = Field(name="x", dtype=self.dtype, nullable=True)
        series = self.pl.Series(
            "x", [b"1", b"2", b"", b"3", None], dtype=self.pl.Binary
        )

        out = target.cast_polars_series(series)

        self.assertEqual(out.to_list(), [1, 2, None, 3, None])
        self.assertEqual(out.dtype, self.pl.Int64)
