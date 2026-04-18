from __future__ import annotations

import pyarrow as pa

from yggdrasil.data.types.base import DataType
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import (
    BooleanType,
    DurationType,
    FloatingPointType,
    IntegerType,
    StringType,
    TimestampType,
)
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.tests import PolarsTestCase


class TestDataTypePandas(PandasTestCase):

    def test_from_pandas_series_dtype_nullable_integer(self):
        series = self.pd.Series([1, 2, None], dtype="Int32")

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int32())

    def test_from_pandas_timestamp(self):
        value = self.pd.Timestamp("2025-01-01T12:00:00Z")

        dtype = DataType.from_pandas(value)

        self.assertIsInstance(dtype, TimestampType)
        self.assertEqual(dtype.unit, "ns")
        self.assertIsNotNone(dtype.tz)

    def test_from_pandas_timedelta(self):
        value = self.pd.Timedelta("1 day")

        dtype = DataType.from_pandas(value)

        self.assertIsInstance(dtype, DurationType)
        self.assertEqual(dtype.unit, "ns")

    def test_from_pandas_dataframe_returns_struct(self):
        df = self.pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dtype = DataType.from_pandas(df)

        self.assertIsInstance(dtype, StructType)
        arrow_struct = dtype.to_arrow()
        self.assertEqual(arrow_struct.field("a").type, pa.int64())
        # pyarrow converts pandas string columns to large_string on modern
        # pandas (3.0+), plain string on older pandas — both are valid.
        b_type = arrow_struct.field("b").type
        self.assertTrue(
            pa.types.is_string(b_type) or pa.types.is_large_string(b_type),
            f"Expected string/large_string, got {b_type!r}",
        )

    def test_from_pandas_series_float(self):
        series = self.pd.Series([1.5, 2.5, 3.5])

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, FloatingPointType)

    def test_from_pandas_series_string_object(self):
        series = self.pd.Series(["a", "b", "c"])

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, StringType)

    def test_from_pandas_series_bool_numpy(self):
        series = self.pd.Series([True, False, True])

        dtype = DataType.from_pandas(series.dtype)

        self.assertIsInstance(dtype, BooleanType)


class TestDataTypePolars(PolarsTestCase):

    def test_from_polars_series(self):
        s = self.pl.Series("a", [1, 2, 3], dtype=self.pl.Int64)

        dtype = DataType.from_polars(s)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int64())

    def test_from_polars_dataframe(self):
        df = self.pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

        dtype = DataType.from_polars(df)

        self.assertIsInstance(dtype, StructType)
        self.assertEqual(
            dtype.to_arrow(),
            pa.struct(
                [
                    pa.field("a", pa.int64(), nullable=True),
                    pa.field("b", pa.string(), nullable=True),
                ]
            ),
        )

    def test_from_polars_series_float(self):
        s = self.pl.Series("f", [1.0, 2.0], dtype=self.pl.Float64)

        dtype = DataType.from_polars(s)

        self.assertIsInstance(dtype, FloatingPointType)
        self.assertEqual(dtype.to_arrow(), pa.float64())

    def test_from_polars_series_string(self):
        s = self.pl.Series("s", ["a", "b"], dtype=self.pl.Utf8)

        dtype = DataType.from_polars(s)

        self.assertIsInstance(dtype, StringType)

    def test_from_polars_dtype_direct(self):
        dtype = DataType.from_polars(self.pl.Int32)

        self.assertIsInstance(dtype, IntegerType)
        self.assertEqual(dtype.to_arrow(), pa.int32())
