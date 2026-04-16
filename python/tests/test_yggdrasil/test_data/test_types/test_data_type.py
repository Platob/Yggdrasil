from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.primitive import IntegerType
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
